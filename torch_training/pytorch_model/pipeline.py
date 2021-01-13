import time
import os
from tqdm import tqdm
import typing
from typing import Callable, Tuple, Union
import numpy as np
from mpi4py import MPI

import torch
from torch import nn
from torch.utils.data import dataloader

from .loss import kappa_metric, correct, flatten_list_tensor_to_numpy, get_auc_precision_recall_AP_f1
from .util import Metric

def train(cfg, model: nn.Module, optimizer: object, train_loader: dataloader, criterion: Callable, 
          loss_acc_metric: object, epoch: int) -> None:
    # ===============================================================================================
    #                                 Train for loop block
    # ===============================================================================================
    model.train()

    total_loss    = 0.0
    train_correct = 0.
    predictions   = []
    groundtruth   = []

    if cfg.DATASET.USE_CROSS_VALID and MPI.COMM_WORLD.Get_rank() != 0:
        pbar = enumerate(train_loader, 0)
    else:
        pbar = tqdm(enumerate(train_loader, 0))
    
    # tracking data time and GPU time and print them on tqdm bar.
    end_time = time.time()

    model.train()
    for i, data in pbar:

        # get the inputs; data is a list of [inputs, labels], put them in cuda
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        data_time = time.time()-end_time # data time
        end_time = time.time()
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # feed forward
        outputs = model(inputs)

        # compute loss and backpropagation
        loss = criterion(outputs, labels)
        loss.backward()
        loss = loss.detach()

        optimizer.step()
        gpu_time = time.time()-end_time # gpu time

        # collect statistics
        predictions.append(outputs.detach().cpu())
        groundtruth.append(labels.detach().cpu())

        running_loss  =  loss.item()
        total_loss    += running_loss
        train_correct += correct(predictions[-1], groundtruth[-1])

        if (cfg.DATASET.USE_CROSS_VALID and MPI.COMM_WORLD.Get_rank() == 0) or not cfg.DATASET.USE_CROSS_VALID:
            pbar.set_postfix_str(f"[{epoch}/{cfg.MODEL.EPOCHS}] [{i+1}/{len(train_loader)} "+
                                 f"training loss={running_loss:.4f}, data time = {data_time:.4f}, gpu time = {gpu_time:.4f}")
        end_time = time.time()

    groundtruth = flatten_list_tensor_to_numpy(groundtruth,len(cfg.DATASET.INT_TO_CLASS))
    predictions = flatten_list_tensor_to_numpy(predictions,len(cfg.DATASET.INT_TO_CLASS))
    loss_acc_metric.push_loss_acc(loss=total_loss/len(train_loader), 
                                  acc=train_correct/(len(train_loader)*cfg.MODEL.BATCH_SIZE), 
                                  train=True,
                                 )
    
    
def validation(cfg, model: nn.Module, test_loader: dataloader, criterion: Callable, epoch: int, loss_acc_metric: object, 
               writer: object =None, optimizer: object = None, save_pred: bool = False) -> None:
    # ===============================================================================================
    #                                     TEST for loop block
    # ===============================================================================================
    model.eval()

    test_total_loss = 0.0
    predictions = []
    groundtruth = []
    model.eval()

    pbar = test_loader
    if not cfg.DATASET.USE_CROSS_VALID or MPI.COMM_WORLD.Get_rank == 0:
        pbar = tqdm(test_loader)
    
    for imgs, labels in pbar:
        with torch.no_grad():
            imgs          = imgs.cuda()
            labels        = labels.cuda()
            outputs       = model(imgs)
            test_loss     = criterion(outputs, labels).item()
            test_total_loss += test_loss

            predictions.append(outputs.cpu())
            groundtruth.append(labels.cpu())
    
    groundtruth = flatten_list_tensor_to_numpy(groundtruth, len(cfg.DATASET.INT_TO_CLASS))
    predictions = flatten_list_tensor_to_numpy(predictions, len(cfg.DATASET.INT_TO_CLASS))

    if save_pred:
        gt_pred_arr = np.stack([groundtruth, predictions], axis=0)
        save_name = f"{cfg.MODEL.BACKBONE}_{cfg.DATASET.PATCH_SIZE}_"
        if cfg.DATASET.USE_CROSS_VALID:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            save_name += f'fold{rank}_'
        save_name += f'test_loss.npy'
        with open(cfg.MODEL.CHECKPOINT_DIR + save_name, 'wb') as f:
            np.save(f, gt_pred_arr)

    test_epoch_loss = test_total_loss/len(test_loader)
    loss_acc_metric.push_loss_acc(loss=test_epoch_loss,
                                  acc=correct(groundtruth, predictions)/(predictions.shape[0]), 
                                  train=False,
                                  )
    
    loss_acc_metric.push_auc_precision_recall_AP_f1(*get_auc_precision_recall_AP_f1(groundtruth, predictions))
    loss_acc_metric.print_summary(epoch=epoch, total_epoch=cfg.MODEL.EPOCHS, 
                                  lr= optimizer.param_groups[0]['lr'] if optimizer else -1)
    
    return groundtruth, predictions
    

def test(cfg, test_loader: dataloader, model: nn.Module, criterion: Callable)->None:
    checkpoint_prefix = f"{cfg.MODEL.BACKBONE}_{cfg.DATASET.PATCH_SIZE}_"
    if cfg.DATASET.USE_CROSS_VALID:
        comm = MPI.COMM_WORLD
        checkpoint_prefix += f'{comm.Get_rank()}_'

    model = model.cuda()

    # prepare training and testing loss
    loss_acc_metric = Metric([key for key in cfg.METRIC.KEYS if 'train' not in key])
    csv_path               = os.path.join(cfg.MODEL.CHECKPOINT_DIR, checkpoint_prefix+"test_loss.csv")
    
     # test pipeline
    groundtruth, predictions = validation(cfg, model, test_loader, criterion, 0, loss_acc_metric, save_pred=True)
    loss_acc_metric.save_metrics(csv_path)

    return groundtruth, predictions
