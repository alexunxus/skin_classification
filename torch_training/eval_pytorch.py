import os
import time
from tqdm import tqdm
import random
import argparse
from mpi4py import MPI

# torch library
import torch
from torch.utils.data import DataLoader
from torch import nn

# customized libraries
from pytorch_model.dataloader import Dataset, skin_augment_fn, imagenet_preproc
from pytorch_model.config     import get_cfg_defaults
from pytorch_model.pipeline   import train, validation, test
from pytorch_model.util       import Metric, cross_valid, check_train_not_zero
from pytorch_model.loss       import get_bceloss, Callback, FocalLoss2d
from pytorch_model.model_zoo  import CustomModel, build_optimizer, build_scheduler

if __name__ == "__main__":
    # get config variables
    cfg = get_cfg_defaults()

    # assign GPU
    device = ','.join(str(i) for i in cfg.SYSTEM.DEVICES)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = device

    #torch.backends.cudnn.enabled = False
    #torch.backends.cudnn.benchmark = True

    # test slides
    train_slide, valid_slide =  cfg.DATASET.TRAIN_SLIDE, cfg.DATASET.VALID_SLIDE
    if cfg.DATASET.USE_CROSS_VALID:
        total_fold_num = len(cfg.SYSTEM.DEVICES)
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        assert total_fold_num == size, f'Communicator world size({size}) != GPU number({total_fold_num})'
        
    test_dataset = Dataset(slide_dir=cfg.DATASET.SLIDE_DIR,
                            target_slide_names = cfg.DATASET.TEST_SLIDE,
                            label_path = cfg.DATASET.LABEL_PATH,
                            bbox_dir= cfg.DATASET.BBOX_PATH,
                            num_slide_hold=cfg.DATASET.NUM_SLIDE,
                            aug_fn=None,
                            preproc_fn=imagenet_preproc,
                            debug = cfg.SOURCE.DEBUG,
                            )
    
    test_loader  = DataLoader(test_dataset , 
                              batch_size=cfg.MODEL.BATCH_SIZE, 
                              shuffle=False, 
                              num_workers=4,
                              drop_last=False,
                              )

    # prepare for checkpoint info and callback
    if not os.path.isdir(cfg.MODEL.CHECKPOINT_DIR):
        os.makedirs(cfg.MODEL.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_prefix = f"{cfg.MODEL.BACKBONE}_{cfg.DATASET.PATCH_SIZE}_"
    if cfg.DATASET.USE_CROSS_VALID:
        checkpoint_prefix += f'fold_{rank}_'
    
    # prepare resnet50, resume from checkpoint
    if cfg.DATASET.USE_CROSS_VALID and rank == 0:
        print("==============Building model=================")
    model = CustomModel(backbone=cfg.MODEL.BACKBONE, 
                        num_cls=len(cfg.DATASET.INT_TO_CLASS), 
                        resume_from=cfg.MODEL.RESULT_DIR+checkpoint_prefix+'best_loss_acc89.pth',
                        norm=cfg.MODEL.NORM_USE)
    #if torch.cuda.device_count() > 1:
    #    print(f"Using {torch.cuda.device_count()} GPUs...")
    #    model = nn.DataParallel(model)
    if torch.cuda.device_count() > 1:
        model = model.cuda(rank)
    else:
        model = model.cuda()
    
    # criterion: BCE loss for multilabel tensor with shape (BATCH_SIZE, 10)
    criterion = FocalLoss2d(weight=torch.tensor([float(item[2]) for item in cfg.DATASET.CLASS_MAP]).cuda())
    
    # training pipeline
    if cfg.DATASET.USE_CROSS_VALID and rank == 0:
        print("==============Start testing==================")
    
    # test pipeline
    test(cfg, test_loader, model, criterion)

    if cfg.DATASET.USE_CROSS_VALID and rank == 0:
        print('Finished Testing')
    
    