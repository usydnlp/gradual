import os
import sys
import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np

import params
from run import load_model, get_tokenizers
from retrieval.data.loaders import get_loader
from retrieval.model import model
from retrieval.train.train import Trainer
from retrieval.utils import file_utils, helper
from retrieval.utils.logger import create_logger
from run import load_yaml_opts, parse_loader_name, get_data_path
from retrieval.train import evaluation

if __name__ == '__main__':
    args = params.get_test_params()
    opt = load_yaml_opts(args.options)
    logger = create_logger(level='debug' if opt.engine.debug else 'info')

    logger.info(f'Used args   : \n{args}')
    logger.info(f'Used options: \n{opt}')

    device = torch.device(args.device)

    data_path = get_data_path(opt)
    data_path = Path(data_path)
    data_name = Path(opt.dataset.train.data.split('.')[0])
    data_dir = data_path / data_name

    loaders = []
    
    for data_info in opt.dataset.val.data:
        _, lang = parse_loader_name(data_info)
        loaders.append(
            get_loader(
                data_split=args.data_split,
                data_path=data_path,
                data_info=data_info,
                loader_name=opt.dataset.loader_name,
                local_rank=args.local_rank,
                text_repr=opt.dataset.text_repr,
                vocab_paths=opt.dataset.vocab_paths,
                ngpu=torch.cuda.device_count(),
                **opt.dataset.val
            )
        )


    tokenizers = get_tokenizers(loaders[0])
    #model = helper.load_model(f'{opt.exp.outpath}/best_model.pkl')
    evaluate_all=False

    filename='logs_flickr_attnsg_img_min_t2i/f30k_precomp/adapt_i2t/checkpoint_10_12000.pkl'
    print("Evaluating "+filename)
    model = helper.load_model(filename, sg_type=args.sg_type, data_dir=data_dir, txt_pooling=args.txt_pooling, txt_pooling_type=args.txt_pooling_type, img_pooling=args.img_pooling,img_pooling_type=args.img_pooling_type)
    print_fn = (lambda x: x) if not model.master else tqdm.write


    q1_emb, q2_emb, v1_emb, v2_emb, lens = evaluation.predict_loader(model, loaders[0], device)
    _, sim_matrix = evaluation.evaluate(model=model, q1_emb=q1_emb, q2_emb=q2_emb,v1_emb=v1_emb, v2_emb=v2_emb, lengths=lens, device=device, shared_size=128, return_sims=True)

    i2t_metrics = evaluation.i2t(sim_matrix)
    t2i_metrics = evaluation.t2i(sim_matrix)

    _metrics_ = ('r1', 'r5', 'r10', 'medr', 'meanr')
    metrics = {}

    rsum = np.sum(i2t_metrics[:3]) + np.sum(t2i_metrics[:3])

    i2t_metrics = {
        f'i2t_{k}': float(v) for k, v in zip(_metrics_, i2t_metrics)
    }
    t2i_metrics = {
        f't2i_{k}': float(v) for k, v in zip(_metrics_, t2i_metrics)
    }

    metrics.update(i2t_metrics)
    metrics.update(t2i_metrics)
    metrics['rsum'] = rsum
    logger.info(metrics)

    np.save('sim.npy',sim_matrix)


