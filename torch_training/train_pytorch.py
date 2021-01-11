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

    # split train/valid
    train_slide, valid_slide =  cfg.DATASET.TRAIN_SLIDE, cfg.DATASET.VALID_SLIDE
    if cfg.DATASET.USE_CROSS_VALID:
        total_fold_num = len(cfg.SYSTEM.DEVICES)
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        assert total_fold_num == size, f'Communicator world size({size}) != GPU number({total_fold_num})'

        fold = cross_valid(train_val_list=cfg.DATASET.TRAIN_SLIDE, 
                            json_path=cfg.DATASET.LABEL_PATH, 
                            num_cls=len(cfg.DATASET.CLASS_MAP), 
                            id_map={li[0]: li[1:] for li in cfg.DATASET.CLASS_MAP},
                            split_ratio=1./total_fold_num)
        fold_n = rank
        train_slide, valid_slide = fold[int(fold_n)]
        print(f'[{rank}] Valid_slide: {valid_slide}')
    
    train_dataset = Dataset(slide_dir=cfg.DATASET.SLIDE_DIR,
                            target_slide_names = train_slide,
                            label_path = cfg.DATASET.LABEL_PATH,
                            bbox_dir= cfg.DATASET.BBOX_PATH,
                            num_slide_hold=cfg.DATASET.NUM_SLIDE,
                            aug_fn=skin_augment_fn,
                            preproc_fn=imagenet_preproc,
                            debug = cfg.SOURCE.DEBUG,
                            )
    valid_dataset = Dataset(slide_dir=cfg.DATASET.SLIDE_DIR,
                            target_slide_names = valid_slide,
                            label_path = cfg.DATASET.LABEL_PATH,
                            bbox_dir= cfg.DATASET.BBOX_PATH,
                            aug_fn=None,
                            preproc_fn=imagenet_preproc,
                            debug = cfg.SOURCE.DEBUG,
                            )
                        
    train_loader = DataLoader(train_dataset, 
                              batch_size=cfg.MODEL.BATCH_SIZE, 
                              shuffle=True,
                              num_workers=4,
                              drop_last=False,)
    valid_loader  = DataLoader(valid_dataset , 
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
        checkpoint_prefix += f'fold_{fold_n}_'
    early_stopping_callback = Callback(model_selection_criterion=cfg.METRIC.MODEL_SELECTION_CRITERION, 
                                       checkpoint_prefix=checkpoint_prefix)

    # prepare resnet50, resume from checkpoint
    if cfg.DATASET.USE_CROSS_VALID and rank == 0:
        print("==============Building model=================")
    model = CustomModel(backbone=cfg.MODEL.BACKBONE, 
                        num_cls=len(cfg.DATASET.INT_TO_CLASS), 
                        resume_from=cfg.MODEL.RESUME_FROM,
                        norm=cfg.MODEL.NORM_USE)
    #if torch.cuda.device_count() > 1:
    #    print(f"Using {torch.cuda.device_count()} GPUs...")
    #    model = nn.DataParallel(model)
    if torch.cuda.device_count() > 1:
        model = model.cuda(rank)
    else:
        model = model.cuda()

    # prepare optimizer: Adam is suggested in this case.
    optimizer = build_optimizer(type=cfg.MODEL.OPTIMIZER, 
                                model=model, 
                                lr=cfg.MODEL.LEARNING_RATE)
    scheduler = build_scheduler(type='step', optimizer=optimizer, cfg=cfg)
    
    # criterion: BCE loss or multilabel focal loss
    # criterion = get_bceloss()
    criterion = FocalLoss2d(weight=torch.tensor([float(item[2]) for item in cfg.DATASET.CLASS_MAP]).cuda())
    
    # prepare training and testing loss
    loss_acc_metric   = Metric(cfg.METRIC.KEYS)
    csv_path          = os.path.join(cfg.MODEL.CHECKPOINT_DIR, checkpoint_prefix+"_loss.csv")
    best_criterion, best_loss, resume_from_epoch = loss_acc_metric.load_metrics(csv_path, 
                                                resume=cfg.MODEL.LOAD_CSV, 
                                                model_selection_criterion=cfg.METRIC.MODEL_SELECTION_CRITERION)
    early_stopping_callback.update_best(best_loss, best_criterion)

    # this zero gradient update is needed to avoid a warning message, issue #8.
    optimizer.zero_grad()
    optimizer.step()
    
    # training pipeline
    if cfg.DATASET.USE_CROSS_VALID and rank == 0:
        print("==============Start training=================")
    
    for epoch in range(0, cfg.MODEL.EPOCHS):  # loop over the dataset multiple times
        # update scheduler  
        if epoch < resume_from_epoch:
            scheduler.step()
            optimizer.step()
            continue
        scheduler.step()

        cuda_num = 0
        if cfg.DATASET.USE_CROSS_VALID:
            cuda_num = comm.Get_rank()
        
        with torch.cuda.device(cuda_num):
            train(cfg, model, optimizer, train_loader, criterion, loss_acc_metric, epoch)
            train_dataset.fetch_new_slide()

            validation(cfg, model, valid_loader, criterion, epoch, loss_acc_metric, optimizer=optimizer)

        if early_stopping_callback.on_epoch_end(cfg=cfg, metrics=loss_acc_metric, csv_path=csv_path, model=model,
                                                opt=optimizer, scheduler=scheduler, epoch=epoch):
            break

    
    if cfg.DATASET.USE_CROSS_VALID and rank == 0:
        print('Finished Training')