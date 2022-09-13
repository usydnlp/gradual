import os
import torch
import pickle
import numpy as np
import json
from PIL import Image
from addict import Dict
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from ..utils.logger import get_logger
from ..utils.file_utils import read_txt, load_pickle
from .preprocessing import get_transform
import nltk
nltk.download('punkt')

logger = get_logger()


class Birds(Dataset):
    def __init__(self, data_path, data_name, transform=None,
                target_transform=None, data_split='train',
                tokenizer=None, lang=None):

        self.data_path = data_path
        self.data_name = data_name
        self.tokenizer = tokenizer
        self.target_transform = target_transform
        self.data_split = data_split
        self.__dataset_path = Path(self.data_path) / self.data_name / data_split

        self.embeddings = load_pickle(
            path=os.path.join(self.__dataset_path, 'char-CNN-RNN-embeddings.pickle'),
            encoding='latin1'
        )

        self.fnames = load_pickle(
            path=os.path.join(self.__dataset_path, 'filenames.pickle'),
            encoding='latin1'
        )

        self.images = load_pickle(
            path=os.path.join(self.__dataset_path, '304images.pickle'),
            encoding='latin1'
        )

        self.class_info = load_pickle(
            path=os.path.join(self.__dataset_path, 'class_info.pickle'),
            encoding='latin1'
        )

        self.captions = self._get_captions()
        self.transform = get_transform(data_split)

        self.captions_per_image = 10

        logger.info(f'[Birds] #Captions {len(self.captions)}')
        logger.info(f'[Birds] #Images {len(self.images)}')
        self.n = self._set_len_captions(data_split)

    def __getitem__(self, ix):
        img = Image.fromarray(self.images[ix//self.captions_per_image]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        fname = self.fnames[ix//self.captions_per_image]
        caption = self.captions[ix]
        tokens = self.tokenizer(caption)
        return img, tokens, ix, fname

    def __len__(self):
        return self.n

    def __repr__(self):
        return f'Birds.{self.data_name}.{self.data_split}'

    def __str__(self):
        return f'{self.data_name}.{self.data_split}'

    def _get_captions(self):
        captions = []
        for fname in self.fnames:
            cap_file = Path(self.data_path) / self.data_name / 'text_c10' / f'{fname}.txt'
            with open(cap_file, 'r') as f:
                cap = f.readlines()
                captions.extend(cap)
        return captions

    def _set_len_captions(self, data_split):
        n = len(self.captions)
        if data_split in ['test', 'dev']:
            n = 10000
        return n


class PrecompDataset(Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(
        self, data_path, data_name,
        data_split, tokenizers, lang='en',
    ):
        logger.debug(f'Precomp dataset\n {[data_path, data_split, tokenizers, lang]}')
        self.tokenizers = tokenizers
        self.lang = lang
        self.data_split = '.'.join([data_split, lang])
        self.data_path = data_path = Path(data_path)
        self.data_name = Path(data_name)
        self.full_path = self.data_path / self.data_name
        # Load Captions
        caption_file = self.full_path / f'{data_split}_caps.{lang}.txt'
        self.captions = read_txt(caption_file)
        logger.debug(f'Read captions. Found: {len(self.captions)}')

        # Load Image features
        img_features_file = self.full_path / f'{data_split}_ims.npy'
        self.images = np.load(img_features_file)
        self.length = len(self.captions)
        self.ids = np.loadtxt(data_path/ data_name / f'{data_split}_ids.txt', dtype=int)

        self.captions_per_image = 5

        logger.debug(f'Read feature file. Shape: {len(self.images.shape)}')

        # Each image must have five captions
        assert (
            self.images.shape[0] == len(self.captions)
            or self.images.shape[0]*5 == len(self.captions)
        )

        if self.images.shape[0] != len(self.captions):
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        # if data_split == 'dev' and self.length > 5000:
        #     self.length = 5000

        print('Image div', self.im_div)

        logger.info('Precomputing captions')
        # self.precomp_captions =  [
        #     self.tokenizer(x)
        #     for x in self.captions
        # ]

        # self.maxlen = max([len(x) for x in self.precomp_captions])
        # logger.info(f'Maxlen {self.maxlen}')

        logger.info((
            f'Loaded PrecompDataset {self.data_name}/{self.data_split} with '
            f'images: {self.images.shape} and captions: {self.length}.'
        ))

    def get_img_dim(self):
        return self.images.shape[-1]

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index//self.im_div
        image = self.images[img_id]
        image = torch.FloatTensor(image)

        # caption = self.precomp_captions[index]
        caption = self.captions[index]

        ret_caption = []
        for tokenizer in self.tokenizers:
            tokens = tokenizer(caption)
            ret_caption.append(tokens)

        batch = Dict(
            image=image,
            caption=ret_caption,
            index=index,
            img_id=img_id,
        )

        return batch

    def __len__(self):
        return self.length

    def __repr__(self):
        return f'PrecompDataset.{self.data_name}.{self.data_split}'

    def __str__(self):
        return f'{self.data_name}.{self.data_split}'




class PrecompBISGDataset(Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(
        self, data_path, data_name,
        data_split, tokenizers, lang='en', sg_type='img_sg', txt_pooling=False,txt_pooling_type='max',img_pooling=False,img_pooling_type='max',
    ):
        print("## Initializing PrecompIMGSGDataset")
        logger.debug(f'Precomp dataset\n {[data_path, data_split, tokenizers, lang]}')
        self.tokenizers = tokenizers
        self.lang = lang
        self.data_split = '.'.join([data_split, lang])
        self.data_path = data_path = Path(data_path)
        self.data_name = Path(data_name) ## coco_precomp
        self.full_path = self.data_path / self.data_name
        self.data_dir = self.full_path
        self.split=data_split

        self.imgsg="[IMGSG]"
        self.txtsg="[TXTSG]"

        if "coco" in data_name:
            self.dt_name="COCO"
        else:
            self.dt_name="Flickr"

        

        # Load Captions
        #caption_file = self.full_path / f'{data_split}_caps.{lang}.txt'
        caption_file = self.full_path / f'{data_split}_caps.txt'
        print("## load captions from ")
        print(caption_file)

        self.captions = read_txt(caption_file)
        logger.debug(f'Read captions. Found: {len(self.captions)}')
        print("Loaded "+str(len(self.captions))+" captions.") # 5N

        # Load IMG SG embedding
        # [IMGSG] Basic graph
        GCN_file = os.path.join(self.data_dir, self.dt_name + '_basic_graph_img.json')
        _, self.sg2index_imgsg = self.build_sg_vocab(self.imgsg, GCN_file)
        self.imgsg_object, self.imgsg_relationship, self.imgsg_obj_count = self.load_imgsg_index(self.split, 'basic_sg')
        del self.sg2index_imgsg

        # [IMGSG] Contextual graph
        _, self.name2index_imgcg = self.build_contexual_sg(
            self.data_dir)

        self.imgcg_object, self.imgcg_relationship, _ = self.load_imgsg_index(self.split, 'contextual_sg')
        print("IMG SG initialized...")

        # Load TXT SG embedding
        # [TXTSG] Basic graph
        GCN_file = os.path.join(self.data_dir, self.dt_name + '_basic_graph_pool.json')
        _, self.sg2index_txtsg = self.build_sg_vocab(self.txtsg, GCN_file)
        self.txtsg_object, self.txtsg_attribute, self.txtsg_relationship, self.txtsg_obj_count = self.load_txtsg_index(self.split,
                                                                                                     'basic_sg')
        del self.sg2index_txtsg

        self.index2vec_inside, self.name2index_inside, self.index2vec_outside, self.name2index_outside, \
        self.index2vec_left, self.name2index_left, self.index2vec_right, self.name2index_right, \
        self.index2vec_above, self.name2index_above, self.index2vec_below, self.name2index_below = self.build_pos_sg(
            self.data_dir)
        self.sg_object_inside, self.sg_relationship_inside, _ = self.load_txtsg_index(self.split, 'pos_inside')
        del self.name2index_inside
        self.sg_object_outside, self.sg_relationship_outside, _ = self.load_txtsg_index(self.split, 'pos_outside')
        del self.name2index_outside
        self.sg_object_left, self.sg_relationship_left, _ = self.load_txtsg_index(self.split, 'pos_left')
        del self.name2index_left
        self.sg_object_right, self.sg_relationship_right, _ = self.load_txtsg_index(self.split, 'pos_right')
        del self.name2index_right
        self.sg_object_above, self.sg_relationship_above, _ = self.load_txtsg_index(self.split, 'pos_above')
        del self.name2index_above
        self.sg_object_below, self.sg_relationship_below, _ = self.load_txtsg_index(self.split, 'pos_below')
        del self.name2index_below

        print("TXT SG initialized...")


        # Load Image features
        print("## Loading img features of shape: ")
        img_features_file = self.full_path / f'{data_split}_ims.npy'
        self.images = np.load(img_features_file)
        print(self.images.shape)  # (N, 36, 2048)

        self.ids = np.loadtxt(data_path/ data_name / f'{data_split}_ids.txt', dtype=int)
        #self.ids = self.ids[:10] ### => will delete later
        self.length = len(self.captions)
        print("## dataset length: "+str(self.length))  #N

        self.captions_per_image = 5

        logger.debug(f'Read feature file. Shape: {len(self.images.shape)}')


        # Each image must have five captions
        assert (
            self.images.shape[0] == len(self.captions)
            or self.images.shape[0]*5 == len(self.captions)
        )

        if self.images.shape[0] != len(self.captions):
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        # if data_split == 'dev' and self.length > 5000:
        #     self.length = 5000

        print('Image div', self.im_div)

        logger.info('Precomputing captions')
        # self.precomp_captions =  [
        #     self.tokenizer(x)
        #     for x in self.captions
        # ]

        # self.maxlen = max([len(x) for x in self.precomp_captions])
        # logger.info(f'Maxlen {self.maxlen}')

        logger.info((
            f'Loaded PrecompDataset {self.data_name}/{self.data_split} with '
            f'images: {self.images.shape} and captions: {self.length}.'
        ))

    # VICTR_added
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

    def build_contexual_sg(self, data_dir):
        GCN_contexual_file = os.path.join(data_dir, self.dt_name+'_contextual_graph_img.json')


        if not os.path.isfile(GCN_contexual_file):
            print('[IMGSG_ERROR]: No gcn embedding file found in %s ' % GCN_contexual_file)
        else:
            print('[IMGSG] Loading Contextual Graph embedding vocab')
            index2vec, obj2index = self.get_vocab(GCN_contexual_file)


        return index2vec, obj2index

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
            print('[TXTSG] Loading Positional Graph embedding vocab')
            index2vec_inside, obj2index_inside = self.get_vocab(GCN_inside_file)
            index2vec_outside, obj2index_outside = self.get_vocab(GCN_outside_file)
            index2vec_left, obj2index_left = self.get_vocab(GCN_left_file)
            index2vec_right, obj2index_right = self.get_vocab(GCN_right_file)
            index2vec_above, obj2index_above = self.get_vocab(GCN_above_file)
            index2vec_below, obj2index_below = self.get_vocab(GCN_below_file)

        return index2vec_inside, obj2index_inside, index2vec_outside, obj2index_outside, \
               index2vec_left, obj2index_left, index2vec_right, obj2index_right, \
               index2vec_above, obj2index_above, index2vec_below, obj2index_below


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

    def load_imgsg_index(self, split, sg_embedding):
        data_dir = self.data_dir
        #sg_path = os.path.join(data_dir, split, 'coco_sg.json')
        sg_path = os.path.join(data_dir, split, 'img_sg_cleared_'+split+".pickle")

        if sg_embedding == 'basic_sg':
            sg_filepath = os.path.join(data_dir, split, 'img_sg_cleared_basic.pickle')

            obj2index, rel2index, att2index = self.sg2index_imgsg, self.sg2index_imgsg, self.sg2index_imgsg
            #return self.process_sg(sg_path, sg_filepath, obj2index, rel2index, att2index)
        elif sg_embedding == 'contextual_sg':
            sg_filepath = os.path.join(data_dir, split, 'img_sg_cleared_contextual.pickle')

            obj2index, rel2index = self.name2index_imgcg, self.name2index_imgcg

        else:
            print('[IMGSG_ERROR]: Wrong img sg naming ')

        return self.process_sg(self.imgsg, sg_path, sg_filepath, obj2index, rel2index, MAX_OBJECTS=20, MAX_RELATIONSHIPS=50)


    def load_txtsg_index(self, split, sg_embedding):
        data_dir = self.data_dir

        sg_path = os.path.join(data_dir, split, 'sg_'+split+".pickle")

        if sg_embedding == 'basic_sg':
            sg_filepath = os.path.join(data_dir, split, 'sg_basic_pool.pickle')

            obj2index, rel2index, att2index = self.sg2index_txtsg, self.sg2index_txtsg, self.sg2index_txtsg
            return self.process_sg(self.txtsg, sg_path, sg_filepath, obj2index, rel2index, att2index)
        elif sg_embedding == 'pos_inside':
            sg_filepath = os.path.join(data_dir, split, 'sg_inside.pickle')

            obj2index, rel2index = self.name2index_inside, self.name2index_inside
        elif sg_embedding == 'pos_outside':
            sg_filepath = os.path.join(data_dir, split, 'sg_outside.pickle')

            obj2index, rel2index = self.name2index_outside, self.name2index_outside
        elif sg_embedding == 'pos_left':
            sg_filepath = os.path.join(data_dir, split, 'sg_left.pickle')

            obj2index, rel2index = self.name2index_left, self.name2index_left
        elif sg_embedding == 'pos_right':
            sg_filepath = os.path.join(data_dir, split, 'sg_right.pickle')

            obj2index, rel2index = self.name2index_right, self.name2index_right
        elif sg_embedding == 'pos_above':
            sg_filepath = os.path.join(data_dir, split, 'sg_above.pickle')

            obj2index, rel2index = self.name2index_above, self.name2index_above
        elif sg_embedding == 'pos_below':
            sg_filepath = os.path.join(data_dir, split, 'sg_below.pickle')

            obj2index, rel2index = self.name2index_below, self.name2index_below
        else:
            sg_filepath = os.path.join(data_dir, split, 'sg_index.pickle')
            obj2index, rel2index, att2index = self.obj2index, self.rel2index, self.att2index
            return self.process_sg(self.txtsg, sg_path, sg_filepath, obj2index, rel2index, att2index)

        return self.process_sg(self.txtsg, sg_path, sg_filepath, obj2index, rel2index, MAX_OBJECTS=5, MAX_RELATIONSHIPS=4)



    def process_sg(self, sg_type, sg_path, sg_filepath, obj2index, rel2index, att2index=None, MAX_OBJECTS=5, MAX_ATTRIBUTES=2, MAX_RELATIONSHIPS=4):
        if not os.path.isfile(sg_filepath):
            with open(sg_path, 'rb') as  f:
                sg_all = pickle.load(f)
            # sg = unicode_convert(sg)

            print(sg_type+" Indexing scene graph for ", len(sg_all), " images")


            new_sg_obj = []
            new_sg_att = []
            new_sg_rel = []
            new_sg_obj_count = []

            for a_sg in sg_all:
                mask_sg = [1] * MAX_OBJECTS  #
                obj_list = [0] * MAX_OBJECTS  #
                rel_pad = [0] * MAX_RELATIONSHIPS
                rel_list = [rel_pad] * MAX_OBJECTS
                att_pad = [0] * MAX_ATTRIBUTES
                att_list = [att_pad] * MAX_OBJECTS

                if len(a_sg['objects']) > 0:
                    if len(a_sg['objects']) < MAX_OBJECTS:
                        mask_sg[:len(a_sg['objects'])] = [0] * len(a_sg['objects'])
                    else:
                        mask_sg[:] = [0] * MAX_OBJECTS

                    get_rel = len(a_sg['relationships']) > 0
                    try:
                        get_att = len(a_sg['attributes']) > 0
                    except KeyError:
                        get_att = False

                    # processing objects
                    for i, obj in enumerate(a_sg['objects']):
                        if i > MAX_OBJECTS - 1:
                            break
                        try:
                            obj_list[i] = obj2index[obj]
                        except KeyError:
                            obj_list[i] = 0

                        # processing relationships for this object
                        if get_rel:
                            rel_pad = [0] * MAX_RELATIONSHIPS
                            rel_ctr = []
                            for rel in a_sg['relationships']:
                                if rel[0] == i or rel[2] == i:
                                    try:
                                        rel_ctr.append(rel2index[rel[1]])
                                    except KeyError:
                                        rel_ctr.append(0)
                            if len(rel_ctr) < MAX_RELATIONSHIPS:
                                rel_pad[:len(rel_ctr)] = rel_ctr
                            else:
                                rel_pad = rel_ctr[:MAX_RELATIONSHIPS]
                            rel_list[i] = rel_pad

                        # processing attribute for this object
                        if get_att and att2index is not None:
                            att_pad = [0] * MAX_ATTRIBUTES
                            att_ctr = []
                            for att in a_sg['attributes']:
                                if att[0] == i:
                                    try:
                                        att_ctr.append(att2index[att[2]])
                                    except KeyError:
                                        att_ctr.append(0)
                            if len(att_ctr) < MAX_ATTRIBUTES:
                                att_pad[:len(att_ctr)] = att_ctr
                            else:
                                att_pad = att_ctr[:MAX_ATTRIBUTES]
                            att_list[i] = att_pad
                else:
                    obj_list[0]=0
                    mask_sg[0]=0

                assert len(mask_sg) == MAX_OBJECTS
                assert len(obj_list) == MAX_OBJECTS
                assert len(rel_list) == MAX_OBJECTS

                new_sg_obj_count.append(mask_sg)
                new_sg_obj.append(obj_list)
                new_sg_rel.append(rel_list)
                if att2index is not None:
                    assert len(att_list) == MAX_OBJECTS
                    new_sg_att.append(att_list)



            if att2index is not None:
                with open(sg_filepath, 'wb') as f:
                    pickle.dump([new_sg_obj, new_sg_att,
                                 new_sg_rel, new_sg_obj_count], f, protocol=2)
                    print('Save to: ', sg_filepath)

            else:
                with open(sg_filepath, 'wb') as f:
                    pickle.dump([new_sg_obj,
                                 new_sg_rel, new_sg_obj_count], f, protocol=2)
                    print('Save to: ', sg_filepath)

        else:

            print(sg_type+' Loading indexed scene graph from: ', sg_filepath)

            if att2index is not None:
                with open(sg_filepath, 'rb') as f:
                    x = pickle.load(f)
                    new_sg_obj, new_sg_att = x[0], x[1]
                    new_sg_rel, new_sg_obj_count = x[2], x[3]
                    del x
            else:
                with open(sg_filepath, 'rb') as f:
                    x = pickle.load(f)
                    new_sg_obj, new_sg_rel, new_sg_obj_count = x[0], x[1], x[2]
                    del x

            print(sg_type+" Loaded indexed scene graph for ", len(new_sg_obj), " images")

        if att2index is None:
            return new_sg_obj, new_sg_rel, new_sg_obj_count
        else:

            return new_sg_obj, new_sg_att, new_sg_rel, new_sg_obj_count


    def get_img_dim(self):
        return self.images.shape[-1]


    def get_imgsg(self, sent_ix):

        sg5_obj = np.asarray(self.imgsg_object[sent_ix]).astype('int64')
        sg5_rel = np.asarray(self.imgsg_relationship[sent_ix]).astype('int64')
        sg5_obj_count=np.asarray(self.imgsg_obj_count[sent_ix]).astype('int64')


        sg5_obj_cg = np.asarray(self.imgcg_object[sent_ix]).astype('int64')
        sg5_rel_cg = np.asarray(self.imgcg_relationship[sent_ix]).astype('int64')

        return sg5_obj,sg5_rel,sg5_obj_count, \
               sg5_obj_cg,sg5_rel_cg

    def get_txtsg(self, sent_ix):

        sg5_obj = np.asarray(self.txtsg_object[sent_ix]).astype('int64')
        sg5_att = np.asarray(self.txtsg_attribute[sent_ix]).astype('int64')
        sg5_rel = np.asarray(self.txtsg_relationship[sent_ix]).astype('int64')
        sg5_obj_count=np.asarray(self.txtsg_obj_count[sent_ix]).astype('int64')

        sg5_obj_inside = np.asarray(self.sg_object_inside[sent_ix]).astype('int64')
        sg5_rel_inside = np.asarray(self.sg_relationship_inside[sent_ix]).astype('int64')

        sg5_obj_outside = np.asarray(self.sg_object_outside[sent_ix]).astype('int64')
        sg5_rel_outside = np.asarray(self.sg_relationship_outside[sent_ix]).astype('int64')

        sg5_obj_left = np.asarray(self.sg_object_left[sent_ix]).astype('int64')
        sg5_rel_left = np.asarray(self.sg_relationship_left[sent_ix]).astype('int64')

        sg5_obj_right = np.asarray(self.sg_object_right[sent_ix]).astype('int64')
        sg5_rel_right = np.asarray(self.sg_relationship_right[sent_ix]).astype('int64')

        sg5_obj_above = np.asarray(self.sg_object_above[sent_ix]).astype('int64')
        sg5_rel_above = np.asarray(self.sg_relationship_above[sent_ix]).astype('int64')

        sg5_obj_below = np.asarray(self.sg_object_below[sent_ix]).astype('int64')
        sg5_rel_below = np.asarray(self.sg_relationship_below[sent_ix]).astype('int64')

        return sg5_obj,sg5_att,sg5_rel,sg5_obj_count, \
               sg5_obj_inside,sg5_rel_inside, \
               sg5_obj_outside,sg5_rel_outside, \
               sg5_obj_left,sg5_rel_left, \
               sg5_obj_right,sg5_rel_right, \
               sg5_obj_above,sg5_rel_above, \
               sg5_obj_below,sg5_rel_below

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index//self.im_div
        #img_id = index
        image = self.images[img_id]
        image = torch.FloatTensor(image)

        imgsg_id=img_id
        txtsg_id=index

        ## get imgsg
        imgsg5_obj, imgsg5_rel,imgsg5_obj_count, \
        imgsg5_obj_cg,imgsg5_rel_cg= self.get_imgsg(imgsg_id)

        ## get txtsg
        txtsg5_obj, txtsg5_att, txtsg5_rel, txtsg5_obj_count, \
        txtsg5_obj_inside, txtsg5_rel_inside, \
        txtsg5_obj_outside, txtsg5_rel_outside, \
        txtsg5_obj_left, txtsg5_rel_left, \
        txtsg5_obj_right, txtsg5_rel_right, \
        txtsg5_obj_above, txtsg5_rel_above, \
        txtsg5_obj_below, txtsg5_rel_below = self.get_txtsg(txtsg_id)


        caption = self.captions[index]

        ret_caption = []
        for tokenizer in self.tokenizers:
            tokens = tokenizer(caption)
            ret_caption.append(tokens)

        batch = Dict(
            image=image,
            caption=ret_caption,
            index=index,
            img_id=img_id,
            imgsg5_obj=imgsg5_obj,
            imgsg5_rel=imgsg5_rel,
            imgsg5_obj_count=imgsg5_obj_count,
            imgsg5_obj_cg=imgsg5_obj_cg,
            imgsg5_rel_cg=imgsg5_rel_cg,
            txtsg5_obj=txtsg5_obj,
            txtsg5_att=txtsg5_att,
            txtsg5_rel=txtsg5_rel,
            txtsg5_obj_count=txtsg5_obj_count,
            txtsg5_obj_inside=txtsg5_obj_inside,
            txtsg5_rel_inside=txtsg5_rel_inside,
            txtsg5_obj_outside=txtsg5_obj_outside,
            txtsg5_rel_outside=txtsg5_rel_outside,
            txtsg5_obj_left=txtsg5_obj_left,
            txtsg5_rel_left=txtsg5_rel_left,
            txtsg5_obj_right=txtsg5_obj_right,
            txtsg5_rel_right=txtsg5_rel_right,
            txtsg5_obj_above=txtsg5_obj_above,
            txtsg5_rel_above=txtsg5_rel_above,
            txtsg5_obj_below=txtsg5_obj_below,
            txtsg5_rel_below=txtsg5_rel_below
        )

        return batch

    def __len__(self):
        return self.length

    def __repr__(self):
        return f'PrecompDataset.{self.data_name}.{self.data_split}'

    def __str__(self):
        return f'{self.data_name}.{self.data_split}'




class DummyDataset(Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(
        self, data_path, data_name,
        data_split, tokenizer, lang='en'
    ):
        logger.debug(f'Precomp dataset\n {[data_path, data_split, tokenizer, lang]}')
        self.tokenizer = tokenizer

        self.captions = np.random.randint(0, 1000, size=(5000, 50))
        logger.debug(f'Read captions. Found: {len(self.captions)}')

        # Load Image features
        self.images = np.random.uniform(size=(1000, 36, 2048))
        self.length = 5000

        logger.debug(f'Read feature file. Shape: {len(self.images.shape)}')

        # Each image must have five captions
        assert (
            self.images.shape[0] == len(self.captions)
            or self.images.shape[0]*5 == len(self.captions)
        )

        if self.images.shape[0] != len(self.captions):
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000
        print('Image div', self.im_div)

        # self.precomp_captions =  [
        #     self.tokenizer(x)
        #     for x in self.captions
        # ]

        # self.maxlen = max([len(x) for x in self.precomp_captions])
        # logger.info(f'Maxlen {self.maxlen}')

    def get_img_dim(self):
        return self.images.shape[-1]

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index//self.im_div
        image = self.images[img_id]
        image = torch.FloatTensor(image)
        # caption = self.precomp_captions[index]
        caption = torch.LongTensor(self.captions[index])

        return image, caption, index, img_id

    def __len__(self):
        return self.length


class CrossLanguageLoader(Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(
        self, data_path, data_name, data_split,
        tokenizers, lang='en-de',
    ):
        logger.debug((
            'CrossLanguageLoader dataset\n '
            f'{[data_path, data_split, tokenizers, lang]}'
        ))

        self.data_path = Path(data_path)
        self.data_name = Path(data_name)
        self.full_path = self.data_path / self.data_name
        self.data_split = '.'.join([data_split, lang])

        self.lang = lang

        assert len(tokenizers) == 1 # TODO: implement multi-tokenizer

        self.tokenizer = tokenizers[0]

        lang_base, lang_target = lang.split('-')
        base_filename = f'{data_split}_caps.{lang_base}.txt'
        target_filename = f'{data_split}_caps.{lang_target}.txt'

        base_file = self.full_path / base_filename
        target_file = self.full_path / target_filename

        logger.debug(f'Base: {base_file} - Target: {target_file}')
        # Paired files
        self.lang_a = read_txt(base_file)
        self.lang_b = read_txt(target_file)

        logger.debug(f'Base and target size: {(len(self.lang_a), len(self.lang_b))}')
        self.length = len(self.lang_a)
        assert len(self.lang_a) == len(self.lang_b)

        logger.info((
            f'Loaded CrossLangDataset {self.data_name}/{self.data_split} with '
            f'captions: {self.length}'
        ))

    def __getitem__(self, index):
        caption_a = self.lang_a[index]
        caption_b = self.lang_b[index]

        target_a = self.tokenizer(caption_a)
        target_b = self.tokenizer(caption_b)

        return target_a, target_b, index

    def __len__(self):
        return self.length

    def __str__(self):
        return f'{self.data_name}.{self.data_split}'


class ImageDataset(Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(
        self, data_path, data_name,
        data_split, tokenizer, lang='en',
        resize_to=256, crop_size=224,
    ):
        from .adapters import Flickr, Coco

        logger.debug(f'ImageDataset\n {[data_path, data_split, tokenizer, lang]}')
        self.tokenizer = tokenizer
        self.lang = lang
        self.data_split = data_split
        self.split = '.'.join([data_split, lang])
        self.data_path = Path(data_path)
        self.data_name = Path(data_name)
        self.full_path = self.data_path / self.data_name

        self.data_wrapper = (
            Flickr(
                self.full_path,
                data_split=data_split,
            ) if 'f30k' in data_name
            else Coco(
                self.full_path,
                data_split=data_split,
            )
        )

        self._fetch_captions()
        self.length = len(self.ids)

        self.transform = get_transform(
            data_split, resize_to=resize_to, crop_size=crop_size
        )

        self.captions_per_image = 5

        if data_split == 'dev' and len(self.length) > 5000:
            self.length = 5000

        logger.debug(f'Split size: {len(self.ids)}')

    def _fetch_captions(self,):
        self.captions = []
        for image_id in sorted(self.data_wrapper.image_ids):
            self.captions.extend(
                self.data_wrapper.get_captions_by_image_id(image_id)[:5]
            )

        self.ids = range(len(self.captions))
        logger.debug(f'Loaded {len(self.captions)} captions')

    def load_img(self, image_id):

        filename = self.data_wrapper.get_filename_by_image_id(image_id)
        feat_path = self.full_path / filename
        try:
            image = default_loader(feat_path)
            image = self.transform(image)
        except OSError:
            print('Error to load image: ', feat_path)
            image = torch.zeros(3, 224, 224,)

        return image

    def __getitem__(self, index):
        # handle the image redundancy
        seq_id = self.ids[index]
        image_id = self.data_wrapper.image_ids[seq_id//5]

        image = self.load_img(image_id)

        caption = self.captions[index]
        cap_tokens = self.tokenizer(caption)

        return image, cap_tokens, index, image_id

    def __len__(self):
        return self.length

    def __repr__(self):
        return f'ImageDataset.{self.data_name}.{self.split}'

    def __str__(self):
        return f'{self.data_name}.{self.split}'
