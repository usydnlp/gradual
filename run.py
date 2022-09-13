import os
import torch
from tqdm import tqdm
from pathlib import Path
import params
from retrieval.train import train
from retrieval.utils import helper
from retrieval.model import loss
from retrieval.model.model import Retrieval, Retrieval_IMGSG, Retrieval_BISG
from retrieval.data.loaders import get_loaders
from retrieval.utils.logger import create_logger
from retrieval.utils.helper import load_model
from retrieval.utils.file_utils import load_yaml_opts, parse_loader_name


def get_data_path(opt):
    if 'DATA_PATH' not in os.environ:
        if not opt.dataset.data_path:
            raise Exception('''
                DATA_PATH not specified.
                Please, run "$ export DATA_PATH=/path/to/dataset"
                or add path to yaml file
            ''')
        return opt.dataset.data_path
    else:
        return os.environ['DATA_PATH']


def get_tokenizers(train_loader):
    tokenizers = train_loader.dataset.tokenizers
    if type(tokenizers) != list:
        tokenizers = [tokenizers]
    return tokenizers


def set_criterion(opt, model):
    if 'name' in opt.criterion:
        logger.info(opt.criterion)
        multimodal_criterion = loss.get_loss(**opt.criterion)
        multilanguage_criterion = loss.get_loss(**opt.criterion)
    else:
        multimodal_criterion = loss.ContrastiveLoss(**opt.criterion)
        multilanguage_criterion = loss.ContrastiveLoss(**opt.ml_criterion)
    set_model_criterion(opt, model, multilanguage_criterion, multimodal_criterion)
    


def set_model_criterion(opt, model, multilanguage_criterion, multimodal_criterion):
    model.mm_criterion = multimodal_criterion
    model.ml_criterion = None
    if len(opt.dataset.adapt.data) > 0:
        model.ml_criterion = multilanguage_criterion


if __name__ == '__main__':
    args = params.get_train_params()
    opt = load_yaml_opts(args.options)
    logger = create_logger(level='debug' if opt.engine.debug else 'info')

    logger.info(f'Used args   : \n{args}')
    logger.info(f'Used options: \n{opt}')

    data_path = get_data_path(opt)
    print("## data path:")
    print(data_path)
    opt.dataset.train.data

    train_loader, val_loaders, adapt_loaders = get_loaders(data_path, args.local_rank, opt)

    tokenizers = get_tokenizers(train_loader)



    model = Retrieval_BISG(**opt.model, tokenizers=tokenizers, sg_type=args.sg_type, txt_pooling=args.txt_pooling, txt_pooling_type=args.txt_pooling_type, img_pooling=args.img_pooling,img_pooling_type=args.img_pooling_type)

    data_path = Path(data_path)
    data_name = Path(opt.dataset.train.data.split('.')[0])
    data_dir = data_path / data_name
    print("## data_dir: "+str(data_dir))
    model.initialize_sg_encoders(data_dir)



    # TODO: Implement complete resume of training
    if opt.exp.resume:
        print("Resuming from saved checkpoint...")
        model = helper.load_model(opt.exp.resume)
    #     model, optimizer = restore_checkpoint(opt, tokenizers)
        print(model)

    print_fn = (lambda x: x) if not model.master else tqdm.write

    set_criterion(opt, model)


    trainer = train.Trainer(
        model=model,
        args=opt,
        sysoutlog=print_fn,
        path=opt.exp.outpath,
        world_size=1 # TODO
    )

    trainer.setup_optim(
        lr=opt.optimizer.lr,
        lr_scheduler=opt.optimizer.lr_scheduler,
        clip_grad=opt.optimizer.grad_clip,
        log_grad_norm=False,
        log_histograms=False,
        optimizer=opt.optimizer,
        freeze_modules=opt.model.freeze_modules
    )

    if opt.engine.eval_before_training:
        result, rs = trainer.evaluate_loaders(
            val_loaders
        )

    trainer.fit(
        train_loader=train_loader,
        valid_loaders=val_loaders,
        lang_loaders=adapt_loaders,
        nb_epochs=opt.engine.nb_epochs,
        valid_interval=opt.engine.valid_interval,
        log_interval=opt.engine.print_freq
    )
