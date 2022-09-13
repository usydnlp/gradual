import torch
import torch.nn as nn
import os
import json
import numpy as np

from . import data_parallel
from ..utils.logger import get_logger
from .imgenc import get_image_encoder, get_img_pooling
from .similarity.factory import get_similarity_object
from .similarity.similarity import Similarity
from .txtenc import get_text_encoder, get_txt_pooling

logger = get_logger()


class Retrieval_BISG(nn.Module):

    def __init__(
        self, txt_enc={}, img_enc={}, similarity={},
        ml_similarity={}, tokenizers=None, latent_size=1024, sg_type='txt', txt_pooling=False, txt_pooling_type='max', img_pooling=False,img_pooling_type='max',
        **kwargs
    ):
        super().__init__()
        print("[IMGSG] Initializing Retrieval_IMGSG...")

        self.master = True
        self.sg_type = sg_type
        self.txt_pooling=txt_pooling
        self.img_pooling=img_pooling
        self.txt_pooling_type=txt_pooling_type
        self.img_pooling_type=img_pooling_type

        self.latent_size = latent_size
        self.img_enc = get_image_encoder(
            name=img_enc.name,
            latent_size=latent_size,
            **img_enc.params
        )

        logger.info((
            'Image encoder created: '
            f'{img_enc.name,}'
        ))

        self.txt_enc = get_text_encoder(
            name = txt_enc.name,
            latent_size=latent_size,
            tokenizers=tokenizers,
            **txt_enc.params,
        )

        self.tokenizers = tokenizers

        self.txt_pool = get_txt_pooling(txt_enc.pooling)
        self.img_pool = get_img_pooling(img_enc.pooling)

        logger.info((
            'Text encoder created: '
            f'{txt_enc.name}'
        ))

        # dimensions for test, TODO: make this configurable
        if sg_type == "txt":
            q1_size = 1024
            q2_size = 1200 if not self.txt_pooling else 900  
            v1_size = 1024
            v2_size = None
        elif sg_type == 'img':
            q1_size = 1024
            q2_size = None
            v1_size = 1024
            v2_size = 800 if not self.img_pooling else 400
        elif sg_type == "bi_adapt":
            q1_size = 1024
            q2_size = 1200 if not self.txt_pooling else 900
            v1_size = 1024
            v2_size = 800 if not self.img_pooling else 400
        elif sg_type =="bi_adapt_atsg":
            q1_size = 1200 if not self.txt_pooling else 900
            q2_size = None
            v1_size = 800 if not self.img_pooling else 400
            v2_size = 800 if not self.img_pooling else 400
        else: # bi_concat
            q1_size = 1024
            q2_size = 1200 if not self.txt_pooling else 900
            v1_size = 1824 if not self.img_pooling else 1424
            v2_size = None

        sim_obj = get_similarity_object(
            similarity.name,
            q1_size,
            q2_size,
            v1_size,
            v2_size,
            **similarity.params
        )

        self.similarity = Similarity(
            similarity_object=sim_obj,
            device=similarity.device,
            #latent_size=latent_size,
            **kwargs
        )

        self.ml_similarity = nn.Identity()
        if ml_similarity is not None:
            self.ml_similarity = self.similarity

            if ml_similarity != {}:
                ml_sim_obj = get_similarity_object(
                    ml_similarity.name,
                    **ml_similarity.params
                )

                self.ml_similarity = Similarity(
                    similarity_object=ml_sim_obj,
                    device=similarity.device,
                    latent_size=latent_size,
                    **kwargs
                )

        self.init_attn(self.sg_type)

        self.set_devices_()
    def initialize_sg_encoders(self, data_dir):
        if 'bi' in self.sg_type  or self.sg_type == 'txt':
            self.initialize_txtsg_encoders_(data_dir)

        if 'bi' in self.sg_type or self.sg_type == 'img':
            self.initialize_imgsg_encoders_(data_dir)


    def init_attn(self, sg_type):
        if 'bi' in self.sg_type or sg_type =='txt':
            sg_dimension=1200 if not self.txt_pooling else 900
            self.txtatten_sg = SG_Attention(sg_dim=sg_dimension)
            self.txtatten_sg.cuda()
            print("## TXTSG attn intialized")

        if 'bi' in self.sg_type or sg_type == 'img':
            sg_dimension=800 if not self.img_pooling else 400
            self.imgatten_sg = SG_Attention(sg_dim=sg_dimension)
            self.imgatten_sg.cuda()
            print("## IMGSG attn intialized")



    def initialize_txtsg_encoders_(self, data_dir):
        if "coco" in str(data_dir):
            dt_name="COCO"
        else:
            dt_name="Flickr"
        print("Initializing txtsg encoder for " + dt_name)
        GCN_file = os.path.join(data_dir, dt_name + '_basic_graph_pool.json')
        index2objvec_txtsg, _ = self.build_sg_vocab("[TXTSG]", GCN_file)

        self.sg_embedding = nn.Embedding.from_pretrained(torch.from_numpy(index2objvec_txtsg))

        index2vec_inside, _, index2vec_outside, _, \
        index2vec_left, _, index2vec_right, _, \
        index2vec_above, _, index2vec_below, _ = self.build_pos_sg(data_dir)

        self.inside_embedding = nn.Embedding.from_pretrained(torch.from_numpy(index2vec_inside))
        self.outside_embedding = nn.Embedding.from_pretrained(torch.from_numpy(index2vec_outside))
        self.left_embedding = nn.Embedding.from_pretrained(torch.from_numpy(index2vec_left))
        self.right_embedding = nn.Embedding.from_pretrained(torch.from_numpy(index2vec_right))
        self.above_embedding = nn.Embedding.from_pretrained(torch.from_numpy(index2vec_above))
        self.below_embedding = nn.Embedding.from_pretrained(torch.from_numpy(index2vec_below))

        self.sg_embedding.cuda()

        self.inside_embedding.cuda()
        self.outside_embedding.cuda()
        self.left_embedding.cuda()
        self.right_embedding.cuda()
        self.above_embedding.cuda()
        self.below_embedding.cuda()
        print("## TXT_SG encoder initialized")



    def build_pos_sg(self, data_dir):
        if "coco" in str(data_dir):
            dt_name="COCO"
        else:
            dt_name="Flickr"

        GCN_inside_file = os.path.join(data_dir, dt_name+'_position_inside.json')
        GCN_outside_file = os.path.join(data_dir, dt_name+'_position_outside.json')
        GCN_left_file = os.path.join(data_dir, dt_name+'_position_left.json')
        GCN_right_file = os.path.join(data_dir, dt_name+'_position_right.json')
        GCN_above_file = os.path.join(data_dir, dt_name+'_position_above.json')
        GCN_below_file = os.path.join(data_dir, dt_name+'_position_below.json')

        if not os.path.isfile(GCN_inside_file):
            print('[TXTSG_ERROR]: No gcn embedding file found in %s ' % GCN_inside_file)
        if not os.path.isfile(GCN_outside_file):
            print('[TXTSG_ERROR]: No gcn embedding file found in %s ' % GCN_outside_file)
        elif not os.path.isfile(GCN_left_file):
            print('[TXTSG_ERROR]: No gcn embedding file found in %s ' % GCN_left_file)
        elif not os.path.isfile(GCN_right_file):
            print('[TXTSG_ERROR]: No gcn embedding file found in %s ' % GCN_right_file)
        elif not os.path.isfile(GCN_above_file):
            print('[TXTSG_ERROR]: No gcn embedding file found in %s ' % GCN_above_file)
        elif not os.path.isfile(GCN_below_file):
            print('[TXTSG_ERROR]: No gcn embedding file found in %s ' % GCN_below_file)
        else:
            print('[TXTSG_ERROR] Loading Positional Graph embedding vocab')
            index2vec_inside, obj2index_inside = self.get_vocab(GCN_inside_file)
            index2vec_outside, obj2index_outside = self.get_vocab(GCN_outside_file)
            index2vec_left, obj2index_left = self.get_vocab(GCN_left_file)
            index2vec_right, obj2index_right = self.get_vocab(GCN_right_file)
            index2vec_above, obj2index_above = self.get_vocab(GCN_above_file)
            index2vec_below, obj2index_below = self.get_vocab(GCN_below_file)

        return index2vec_inside, obj2index_inside, index2vec_outside, obj2index_outside, \
               index2vec_left, obj2index_left, index2vec_right, obj2index_right, \
               index2vec_above, obj2index_above, index2vec_below, obj2index_below

    def initialize_imgsg_encoders_(self, data_dir):
        if "coco" in str(data_dir):
            dt_name="COCO"
        else:
            dt_name="Flickr"
        print("Initializing imgsg encoder for "+dt_name)
        GCN_file = os.path.join(data_dir, dt_name + '_basic_graph_img.json')
        index2objvec_imgsg, _ = self.build_sg_vocab("[IMGSG]", GCN_file)

        index2vec_imgcg, _ = self.build_contexual_sg(data_dir, dt_name)

        # [IMGSG] Basic Graph node embedding
        self.imgsg_embedding = nn.Embedding.from_pretrained(torch.from_numpy(index2objvec_imgsg))

        # [IMGSG] Contextual Graph node embedding
        self.imgcg_embedding = nn.Embedding.from_pretrained(torch.from_numpy(index2vec_imgcg))

        self.imgsg_embedding.cuda()

        self.imgcg_embedding.cuda()
        print("## IMG_SG encoder initialized")


    def build_contexual_sg(self, data_dir, dt_name):
        GCN_contexual_file = os.path.join(data_dir, dt_name+'_contextual_graph_img.json')


        if not os.path.isfile(GCN_contexual_file):
            print('[IMGSG_ERROR]: No gcn embedding file found in %s ' % GCN_contexual_file)
        else:
            print('[IMGSG] Loading Contextual Graph embedding vocab')
            index2vec, obj2index = self.get_vocab(GCN_contexual_file)


        return index2vec, obj2index

    def get_vocab(self, file):
        with open(file, 'r') as f:
            embeddings = json.load(f)
        dim = len(embeddings[list(embeddings.keys())[0]])
        padding = np.zeros((dim,))
        index2vec = []
        index2vec.append(padding)
        obj2index = {}
        count = 1
        for word in embeddings:
            obj2index[word] = count
            index2vec.append(embeddings[word])
            count += 1
        return np.array(index2vec).astype(np.float32), obj2index

    def build_sg_vocab(self, sg_type, GCN_file):

        if not os.path.isfile(GCN_file):
            print(sg_type+' [ERROR] No GCN embedding file found in %s ' % GCN_file)
        else:
            print(sg_type+' Loading Basic Graph embedding vocab from %s ...' % GCN_file)
            with open(GCN_file, 'r') as f:
                GCN_embed = json.load(f)

            dim=len(GCN_embed[list(GCN_embed.keys())[0]])
            padding = np.zeros((dim,))
            index2vec = []
            index2vec.append(padding)
            sg2index = {}
            count = 1
            for word in GCN_embed:
                sg2index[word] = count
                index2vec.append(GCN_embed[word])
                count += 1
        return np.array(index2vec).astype(np.float32), sg2index


    def set_devices_(self, txt_devices=['cuda'], img_devices=['cuda'], loss_device='cuda'):
        if len(txt_devices) > 1:
            self.txt_enc = data_parallel.DataParallel(self.txt_enc)
            self.txt_enc.device = torch.device('cuda')
        elif len(txt_devices) == 1:
            self.txt_enc.to(txt_devices[0])
            self.txt_enc.device = torch.device(txt_devices[0])

        if len(img_devices) > 1:
            self.img_enc = data_parallel.DataParallel(self.img_device)
            self.img_enc.device = torch.device('cuda')
        elif len(img_devices) == 1:
            self.img_enc.to(img_devices[0])
            self.img_enc.device = torch.device(img_devices[0])




        self.loss_device = torch.device(loss_device)

        self.similarity = self.similarity.to(self.loss_device)
        self.ml_similarity = self.ml_similarity.to(self.loss_device)

        logger.info((
            f'Setting devices: '
            f'img: {self.img_enc.device},'
            f'txt: {self.txt_enc.device}, '
            f'loss: {self.loss_device}'
        ))

    def embed_caption_features(self, cap_features, lengths):
        return self.txt_pool(cap_features, lengths)

    def embed_image_features(self, img_features):
        return self.img_pool(img_features)

    def embed_images(self, batch):
        img_tensor = self.img_enc(batch)
        img_embed  = self.embed_image_features(img_tensor)
        return img_embed

    def embed_captions(self, batch):
        txt_tensor, lengths = self.txt_enc(batch)
        txt_embed = self.embed_caption_features(txt_tensor, lengths)
        return txt_embed

    def forward_batch(self, batch):
        img_embed = self.embed_images(batch)
        txt_embed = self.embed_captions(batch)
        if "bi" in self.sg_type:
            txtsg_embed, imgsg_embed = self.embed_sg(batch, img_sg=True,txt_sg=True)
        elif self.sg_type=="txt":
            txtsg_embed = self.embed_sg(batch, img_sg=False, txt_sg=True)
        else:
            imgsg_embed = self.embed_sg(batch, img_sg=True, txt_sg=False)

        if "bi" in self.sg_type or self.sg_type=="txt":
            batch_size = len(batch['txtsg5_obj_count'])
            sg_mask = batch['txtsg5_obj_count'].view(batch_size, -1)
            sg_mask_valid = (sg_mask == 1)
            sg_mask_valid = sg_mask_valid.cuda()
            self.txtatten_sg.applyMask(sg_mask_valid)
            txtsg_cap, _ = self.txtatten_sg(txt_embed, txtsg_embed)
            txtsg_cap_glb = torch.mean(txtsg_cap, -2)

        if "bi" in self.sg_type or self.sg_type=="img":
            batch_size = len(batch['imgsg5_obj_count'])
            sg_mask = batch['imgsg5_obj_count'].view(batch_size, -1)
            sg_mask_valid = (sg_mask == 1)
            sg_mask_valid = sg_mask_valid.cuda()
            self.imgatten_sg.applyMask(sg_mask_valid)
            imgsg_cap, _ = self.imgatten_sg(img_embed, imgsg_embed)
            imgsg_cap_glb = torch.mean(imgsg_cap, -2)

        if self.sg_type=="txt":
            return txt_embed, txtsg_cap_glb, img_embed, None
        elif self.sg_type=="img":
            return txt_embed, None, img_embed, imgsg_cap_glb
        elif self.sg_type=="bi_adapt":
            return txt_embed, txtsg_cap_glb, img_embed, imgsg_cap_glb
        elif self.sg_type=="bi_adapt_atsg":
            return txtsg_cap, None, imgsg_cap, imgsg_cap_glb 
        else: # bi_concat
            concat_img_embed=torch.cat((img_embed, imgsg_cap), -1)
            return txt_embed, txtsg_cap_glb, concat_img_embed, None



    def embed_sg(self, batch, img_sg=True, txt_sg=False):

        if img_sg:
            imgsg5_obj = batch['imgsg5_obj'].cuda()
            imgsg5_rel = batch['imgsg5_rel'].cuda()
            imgsg5_obj_cg = batch['imgsg5_obj_cg'].cuda()
            imgsg5_rel_cg = batch['imgsg5_rel_cg'].cuda()

            #obj embedding
            imgsg5_obj_cg_embed = self.imgcg_embedding(imgsg5_obj_cg)  # [16, 20, 50]
            imgsg5_obj_embed = self.imgsg_embedding(imgsg5_obj)  # [16, 5, d]

            #rel embedding
            imgsg5_rel_cg_shape = list(imgsg5_rel_cg.size())  # [16, 20, 20]
            imgsg5_rel_cg_embed = self.imgcg_embedding(imgsg5_rel_cg.view(-1, imgsg5_rel_cg_shape[1] * imgsg5_rel_cg_shape[2]))
            imgsg5_rel_shape = list(imgsg5_rel.size())
            imgsg5_rel_embed = self.imgsg_embedding(imgsg5_rel.view(-1, imgsg5_rel_shape[1] * imgsg5_rel_shape[2]))

            # [IMGSG] contextual graph
            if self.img_pooling:
                imgsg5_obj_cg_embed=torch.unsqueeze(imgsg5_obj_cg_embed,dim=-2)
                imgsg5_obj_embed=torch.unsqueeze(imgsg5_obj_embed,dim=-2)
                imgsg_obj_all=torch.cat((imgsg5_obj_cg_embed,imgsg5_obj_embed),-2)
                imgsg5_rel_cg_embed=torch.unsqueeze(imgsg5_rel_cg_embed,dim=-2)
                imgsg5_rel_embed=torch.unsqueeze(imgsg5_rel_embed,dim=-2)
                imgrel_embed_pro=torch.cat((imgsg5_rel_cg_embed,imgsg5_rel_embed),-2)
                


                if self.img_pooling_type=='max':
                    #print("[IMGSG_POOLING - MAX]")
                    imgsg_obj_all=torch.max(imgsg_obj_all,-2)[0]
                    imgrel_embed_pro=torch.max(imgrel_embed_pro,-2)[0]
                elif self.img_pooling_type=='avr':
                    #print("[IMGSG_POOLING - AVR]")
                    imgsg_obj_all=torch.mean(imgsg_obj_all,-2)
                    imgrel_embed_pro=torch.mean(imgrel_embed_pro,-2)
                else:
                    #print("[IMGSG_POOLING - MIN]")
                    imgsg_obj_all=torch.min(imgsg_obj_all,-2)[0]
                    imgrel_embed_pro=torch.min(imgrel_embed_pro,-2)[0]
                imgrel_embed_pro = imgrel_embed_pro.view(imgsg5_rel_cg_shape[0], imgsg5_rel_cg_shape[1],imgsg5_rel_cg_shape[2], -1)
                imgsg_rel_all=torch.mean(imgrel_embed_pro, -2) 



            else:
                imgsg5_rel_cg_embed = imgsg5_rel_cg_embed.view(imgsg5_rel_cg_shape[0], imgsg5_rel_cg_shape[1],
                                                             imgsg5_rel_cg_shape[2], -1)  # [16, 20, 20, 50]
                imgrel_cg_embed_pro = torch.mean(imgsg5_rel_cg_embed, -2)  # [16, 20, 50]

                imgsg5_rel_embed = imgsg5_rel_embed.view(imgsg5_rel_shape[0], imgsg5_rel_shape[1], imgsg5_rel_shape[2], -1)

                imgrel_embed_pro = torch.mean(imgsg5_rel_embed, -2)  # [14, 5, 300]

                imgsg_obj_all = torch.cat((imgsg5_obj_embed, imgsg5_obj_cg_embed), -1)

                
                imgsg_rel_all = torch.cat((imgrel_embed_pro, imgrel_cg_embed_pro), -1)
                del imgrel_cg_embed_pro

            del imgsg5_obj_embed, imgsg5_obj_cg_embed, imgrel_embed_pro
            imgsg_embed = torch.cat((imgsg_obj_all, imgsg_rel_all), -1)
            del imgsg_obj_all, imgsg_rel_all


        if txt_sg:
            txtsg5_obj = batch['txtsg5_obj'].cuda()
            txtsg5_att = batch['txtsg5_att'].cuda()
            txtsg5_rel = batch['txtsg5_rel'].cuda()
            txtsg5_obj_inside = batch['txtsg5_obj_inside'].cuda()
            txtsg5_rel_inside = batch['txtsg5_rel_inside'].cuda()
            txtsg5_obj_outside = batch['txtsg5_obj_outside'].cuda()
            txtsg5_rel_outside = batch['txtsg5_rel_outside'].cuda()
            txtsg5_obj_left = batch['txtsg5_obj_left'].cuda()
            txtsg5_rel_left = batch['txtsg5_rel_left'].cuda()
            txtsg5_obj_right = batch['txtsg5_obj_right'].cuda()
            txtsg5_rel_right = batch['txtsg5_rel_right'].cuda()
            txtsg5_obj_above = batch['txtsg5_obj_above'].cuda()
            txtsg5_rel_above = batch['txtsg5_rel_above'].cuda()
            txtsg5_obj_below = batch['txtsg5_obj_below'].cuda()
            txtsg5_rel_below = batch['txtsg5_rel_below'].cuda()


            #obj embedding
            txtsg5_obj_embed = self.sg_embedding(txtsg5_obj)  # [16, 5, d]
            
            txtsg5_obj_inside_embed = self.inside_embedding(txtsg5_obj_inside)  # [16, 20, 50]
            txtsg5_obj_outside_embed = self.outside_embedding(txtsg5_obj_outside)
            txtsg5_obj_left_embed = self.left_embedding(txtsg5_obj_left)
            txtsg5_obj_right_embed = self.right_embedding(txtsg5_obj_right)
            txtsg5_obj_above_embed = self.above_embedding(txtsg5_obj_above)
            txtsg5_obj_below_embed = self.below_embedding(txtsg5_obj_below)

            txtposition_obj = torch.cat(
                    (txtsg5_obj_inside_embed, txtsg5_obj_outside_embed, txtsg5_obj_left_embed, txtsg5_obj_right_embed,
                     txtsg5_obj_above_embed, txtsg5_obj_below_embed), -1)
            del txtsg5_obj_inside_embed, txtsg5_obj_outside_embed, txtsg5_obj_left_embed, txtsg5_obj_right_embed, txtsg5_obj_above_embed, txtsg5_obj_below_embed

            #rel embedding
            txtsg5_rel_shape = list(txtsg5_rel.size())
            txtsg5_rel_embed = self.sg_embedding(txtsg5_rel.view(-1, txtsg5_rel_shape[1] * txtsg5_rel_shape[2]))

            txtsg5_rel_inside_shape = list(txtsg5_rel_inside.size())  # [16, 20, 20]
            txtsg5_rel_inside_embed = self.inside_embedding(txtsg5_rel_inside.view(-1, txtsg5_rel_inside_shape[1] * txtsg5_rel_inside_shape[2]))
            
            txtsg5_rel_outside_shape = list(txtsg5_rel_outside.size())  # [16, 20, 20]
            txtsg5_rel_outside_embed = self.outside_embedding(txtsg5_rel_outside.view(-1, txtsg5_rel_outside_shape[1] * txtsg5_rel_outside_shape[2]))

            txtsg5_rel_left_shape = list(txtsg5_rel_left.size())
            txtsg5_rel_left_embed = self.left_embedding(txtsg5_rel_left.view(-1, txtsg5_rel_left_shape[1] * txtsg5_rel_left_shape[2]))

            txtsg5_rel_right_shape = list(txtsg5_rel_right.size())
            txtsg5_rel_right_embed = self.right_embedding(txtsg5_rel_right.view(-1, txtsg5_rel_right_shape[1] * txtsg5_rel_right_shape[2]))

            txtsg5_rel_above_shape = list(txtsg5_rel_above.size())
            txtsg5_rel_above_embed = self.above_embedding(txtsg5_rel_above.view(-1, txtsg5_rel_above_shape[1] * txtsg5_rel_above_shape[2]))

            txtsg5_rel_below_shape = list(txtsg5_rel_below.size())
            txtsg5_rel_below_embed = self.below_embedding(txtsg5_rel_below.view(-1, txtsg5_rel_below_shape[1] * txtsg5_rel_below_shape[2]))

            #att embedding
            txtsg5_att_shape = list(txtsg5_att.size())
            txtsg5_att_embed = self.sg_embedding(txtsg5_att.view(-1, txtsg5_att_shape[1] * txtsg5_att_shape[2]))
            txtsg5_att_embed = txtsg5_att_embed.view(txtsg5_att_shape[0], txtsg5_att_shape[1], txtsg5_att_shape[2],-1)  # [14, 5, 10, d]
            txtatt_embed_pro = torch.mean(txtsg5_att_embed, -2)  # [14, 5, 300]


            if self.txt_pooling:

                txtposition_rel = torch.cat(
                    (txtsg5_rel_inside_embed, txtsg5_rel_outside_embed, txtsg5_rel_left_embed, txtsg5_rel_right_embed,
                     txtsg5_rel_above_embed, txtsg5_rel_below_embed), -1)

                txtsg5_obj_embed=torch.unsqueeze(txtsg5_obj_embed,dim=-2)
                txtposition_obj=torch.unsqueeze(txtposition_obj,dim=-2)
                txtsg_obj_all=torch.cat((txtsg5_obj_embed,txtposition_obj),-2)
                
                txtsg5_rel_embed=torch.unsqueeze(txtsg5_rel_embed,dim=-2)
                txtposition_rel=torch.unsqueeze(txtposition_rel,dim=-2)
                txtrel_embed_pro =torch.cat((txtsg5_rel_embed,txtposition_rel),-2)


                


                if self.txt_pooling_type=='max':
                    #print("[IMGSG_POOLING - MAX]")
                    txtsg_obj_all=torch.max(txtsg_obj_all,-2)[0]
                    txtrel_embed_pro=torch.max(txtrel_embed_pro,-2)[0]
                elif self.txt_pooling_type=='avr':
                    #print("[IMGSG_POOLING - AVR]")
                    txtsg_obj_all=torch.mean(txtsg_obj_all,-2)
                    txtrel_embed_pro=torch.mean(txtrel_embed_pro,-2)
                else:
                    #print("[IMGSG_POOLING - MIN]")
                    txtsg_obj_all=torch.min(txtsg_obj_all,-2)[0]
                    txtrel_embed_pro=torch.min(txtrel_embed_pro,-2)[0]

                txtrel_embed_pro = txtrel_embed_pro.view(txtsg5_rel_above_shape[0], txtsg5_rel_above_shape[1],txtsg5_rel_above_shape[2], -1)
                txtsg_rel_all=torch.mean(txtrel_embed_pro, -2) 

            # [TXTSG] positional graph - inside
            else:
                
                txtsg5_rel_inside_embed = txtsg5_rel_inside_embed.view(txtsg5_rel_inside_shape[0], txtsg5_rel_inside_shape[1],txtsg5_rel_inside_shape[2], -1)  # [16, 20, 20, 50]
                txtrel_inside_embed_pro = torch.mean(txtsg5_rel_inside_embed, -2)  # [16, 20, 50]

                # [TXTSG] positional graph - outside
                txtsg5_rel_outside_embed = txtsg5_rel_outside_embed.view(txtsg5_rel_outside_shape[0], txtsg5_rel_outside_shape[1],txtsg5_rel_outside_shape[2], -1)
                txtrel_outside_embed_pro = torch.mean(txtsg5_rel_outside_embed, -2)

                # [TXTSG] positional graph - left
                txtsg5_rel_left_embed = txtsg5_rel_left_embed.view(txtsg5_rel_left_shape[0], txtsg5_rel_left_shape[1],txtsg5_rel_left_shape[2], -1)
                txtrel_left_embed_pro = torch.mean(txtsg5_rel_left_embed, -2)

                # [TXTSG] positional graph - right

                txtsg5_rel_right_embed = txtsg5_rel_right_embed.view(txtsg5_rel_right_shape[0], txtsg5_rel_right_shape[1],txtsg5_rel_right_shape[2], -1)
                txtrel_right_embed_pro = torch.mean(txtsg5_rel_right_embed, -2)

                # [TXTSG] positional graph - above
                txtsg5_rel_above_embed = txtsg5_rel_above_embed.view(txtsg5_rel_above_shape[0], txtsg5_rel_above_shape[1],txtsg5_rel_above_shape[2], -1)
                txtrel_above_embed_pro = torch.mean(txtsg5_rel_above_embed, -2)

                # [TXTSG] positional graph - below
                txtsg5_rel_below_embed = txtsg5_rel_below_embed.view(txtsg5_rel_below_shape[0], txtsg5_rel_below_shape[1],txtsg5_rel_below_shape[2], -1)
                txtrel_below_embed_pro = torch.mean(txtsg5_rel_below_embed, -2)

                txtposition_rel = torch.cat(
                    (txtrel_inside_embed_pro, txtrel_outside_embed_pro, txtrel_left_embed_pro, txtrel_right_embed_pro,
                     txtrel_above_embed_pro, txtrel_below_embed_pro), -1)
                del txtrel_inside_embed_pro, txtrel_outside_embed_pro, txtrel_left_embed_pro, txtrel_right_embed_pro, txtrel_above_embed_pro, txtrel_below_embed_pro

                txtsg5_rel_embed = txtsg5_rel_embed.view(txtsg5_rel_shape[0], txtsg5_rel_shape[1], txtsg5_rel_shape[2], -1)


                txtrel_embed_pro = torch.mean(txtsg5_rel_embed, -2)  # [14, 5, 300]

                txtsg_obj_all = torch.cat((txtsg5_obj_embed, txtposition_obj), -1)
                
                txtsg_rel_all = torch.cat((txtrel_embed_pro, txtposition_rel), -1)
                

            txtsg_embed = torch.cat((txtsg_obj_all, txtsg_rel_all, txtatt_embed_pro), -1)
            del txtsg5_rel_inside_embed, txtsg5_rel_outside_embed, txtsg5_rel_left_embed, txtsg5_rel_right_embed, txtsg5_rel_above_embed, txtsg5_rel_below_embed
            del txtsg5_obj_embed, txtposition_obj
            del txtrel_embed_pro, txtposition_rel, txtsg_obj_all, txtsg_rel_all, txtatt_embed_pro


        if img_sg and not txt_sg:
            return imgsg_embed

        elif txt_sg and not img_sg:
            return txtsg_embed

        else:
            return txtsg_embed, imgsg_embed




    def get_sim_matrix(self, q1_embed, q2_embed, v1_embed, v2_embed, lens=None):
        return self.similarity(q1_embed, q2_embed, v1_embed, v2_embed, lens)

    def get_ml_sim_matrix(self, embed_a, embed_b, lens=None):
        return self.ml_similarity(embed_a, embed_b, lens)

    def get_sim_matrix_shared(self, q1_emb=None, q2_emb=None, v1_emb=None, v2_emb=None, lens=None, shared_size=128):
        return self.similarity.forward_shared(
            q1_emb, q2_emb, v1_emb,  v2_emb, lens,
            shared_size=shared_size
        )




class SG_Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    *modified based on IBM's initial implementation of :class:`Attention`.
    Here is their `License:
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`
    """

    def __init__(self, query_dim=1024, sg_dim=1200, attention_type='general'):
        super(SG_Attention, self).__init__()
        print("### sg_dim "+str(sg_dim))

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(query_dim, sg_dim, bias=False)

        #self.linear_out = nn.Linear(query_dim+sg_dim, query_dim, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        # self.tanh = nn.Tanh()

        self.init_trainable_weights()
        self.mask = None

    def init_trainable_weights(self):
        initrange = 0.1
        if self.attention_type == 'general':
            self.linear_in.weight.data.uniform_(-initrange, initrange)
        #self.linear_out.weight.data.uniform_(-initrange, initrange)

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, query_in, context_in):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """

        # query = self.linear_trans(query_in) #c[16, 20, 1200] q[16, 12, 256]
        batch_size, output_len, query_dim = query_in.size(0), query_in.size(1), query_in.size(2)
        dimensions = context_in.size(2)
        query_len = context_in.size(1)
        context = context_in.view(batch_size, query_len, dimensions)

        if self.attention_type == "general":
            query = query_in.view(batch_size * output_len, -1)
            query = self.linear_in(query)
            query = query.view(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)  # 16*12, 20
        # [14, 12, 100]

        if self.mask is not None:
            mask = self.mask.repeat(output_len, 1)  # 16, 20
            attention_scores.data.masked_fill_(mask.bool(), -float('inf'))

        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        return mix, attention_weights
