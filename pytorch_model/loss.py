import os
import numpy as np
import torch
from torch import Tensor
from torch.nn import BCELoss
from torch import nn
from sklearn.metrics import cohen_kappa_score, confusion_matrix, average_precision_score, auc, roc_curve

from yacs.config import CfgNode
import typing
from typing import Tuple, Union

def get_bceloss():
    # binary cross entropy loss
    return BCELoss()

def kappa_score(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    # quadratic weighted kappa
    return cohen_kappa_score(gt, pred, weights='quadratic')

@torch.no_grad()
def kappa_metric(gts: np.ndarray, preds: np.ndarray) -> np.ndarray:
    '''
    gts:   nparray with shape (N, 5)
    preds: nparray with shape (N, 5)
    '''
    #gts   = np.concatenate([tensor.numpy()>0.5 for tensor in gts  ], axis=0).reshape((-1, last_dim))
    #preds = np.concatenate([tensor.numpy()>0.5 for tensor in preds], axis=0).reshape((-1, last_dim))
    gts   = gts.round().astype(np.int32).sum(1)
    preds = preds.round().astype(np.int32).sum(1)
    k     = kappa_score(gts, preds)
    conf  = confusion_matrix(gts, preds)
    print(f"Kappa score = {k}")
    print("Confusion matrix:\n", conf)
    return k

def correct(gt: Union[np.ndarray, Tensor], pred: Union[np.ndarray, Tensor]) -> np.ndarray:
    '''
    gt    is a (N, num_class) ndarray or tensor
    label is a (N, num_class) ndarray or tensor
    '''
    if isinstance(gt, Tensor):
        gt, pred = gt.numpy(), pred.numpy()
    gt   = np.argmax(gt, axis=-1)
    pred = np.argmax(pred, axis=-1)
    return (gt == pred).sum()

def get_auc_precision_recall_AP_f1(gt: np.ndarray, pred: np.ndarray) -> \
        Tuple[np.float32, np.float32, np.float32, np.float32]:
    # flatten array
    gt   = gt.reshape(-1)
    pred = pred.reshape(-1)
    
    AP   = average_precision_score(gt, pred)

    fpr, tpr, thresholds = roc_curve(gt, pred, pos_label=1)
    auc_ = auc(fpr, tpr)

    Tp = np.logical_and(gt == 1, pred >= 0.5).sum()
    Fp = np.logical_and(gt == 0, pred >= 0.5).sum()
    Fn = np.logical_and(gt == 1, pred <= 0.5).sum()

    precision = Tp/(Tp+Fp+1e-7)
    recall    = Tp/(Tp+Fn+1e-7)
    f1        = 2*precision*recall/(precision + recall + 1e-7)
    return   auc_, precision, recall, AP, f1

def flatten_list_tensor_to_numpy(list_tensor: list, last_dim: int) -> np.ndarray:
    return np.concatenate([tensor.numpy() for tensor in list_tensor], axis=0).reshape((-1, last_dim))

class Callback:
    def __init__(self, model_selection_criterion: str, checkpoint_prefix:str) -> None:
        self.model_selection_criterion = model_selection_criterion
        self.checkpoint_prefix = checkpoint_prefix
        self.patience=0
        self.best_loss=100
        self.best_criterion=-100
    
    def update_best(self, best_loss: np.ndarray, best_criterion: np.ndarray) -> None:
        self.best_criterion = best_criterion
        self.best_loss = best_loss
    
    def get_best_loss(self):
        return self.best_loss

    def on_epoch_end(self, cfg: CfgNode, metrics: object, csv_path: str, model:nn.Module, opt: object=None,
                     scheduler:object=None, epoch: int=None) -> int:
        # ===============================================================================================
        #                                     Callback block
        # ===============================================================================================
        # 1. Save best weight: 
        #   If for one epoch, the test loss or kappa is better than current best kappa and best loss, then
        #   I reset the patience and save the model
        #   Otherwise, patience <- patience+1, and the model weight is not saved.
        # 2. Early stopping: 
        #   If the patience >= the patience limit, then break the epoch for loop and finish training.
        # 3. Saving training curve:
        #   For each epoch, update the loss, kappa dictionary and save them.
        update_loss      = False
        update_criterion = False
        exit_signal = 0

        if cfg.SOURCE.DEBUG:
            print("Debugging, not saving...")
            metrics.save_metrics(csv_path=csv_path, debug=True)
        else:
            print('======Saving training curves=======')
            metrics.save_metrics(csv_path=csv_path, debug=False)
        
        test_epoch_loss = metrics.metric_dict['test_losses'][-1]
        if test_epoch_loss < self.best_loss:
            self.best_loss = test_epoch_loss
            if not cfg.SOURCE.DEBUG:
                torch.save(model.state_dict(), os.path.join(cfg.MODEL.CHECKPOINT_DIR, self.checkpoint_prefix+"best_loss.pth"))
            self.patience = 0
            update_loss = True
        criterion = metrics.metric_dict[self.model_selection_criterion][-1]
        if criterion >= self.best_criterion:
            self.best_criterion = criterion
            if not cfg.SOURCE.DEBUG:
                torch.save({'model_state_dict': model.state_dict(),
                            'opt_state_dict': opt.state_dict() if opt else {},
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else {},
                            'epoch': epoch if epoch else 0,
                            }, 
                           os.path.join(cfg.MODEL.CHECKPOINT_DIR, 
                           self.checkpoint_prefix+f"best_{self.model_selection_criterion}.pth"))
            self.patience = 0
            update_criterion = True
        if not update_loss and not update_criterion:
            self.patience += 1
            print(f"Patience = {self.patience}")
            if self.patience >= cfg.MODEL.PATIENCE:
                print(f"Early stopping at epoch {len(metrics.metric_dict[self.model_selection_criterion])}")
                exit_signal = 1

        print(f"best loss={self.best_loss}, best {self.model_selection_criterion}={self.best_criterion}")
        return exit_signal