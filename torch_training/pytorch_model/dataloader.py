import matplotlib.pyplot as plt
import numpy as np
import os
import json
import typing
from typing import Callable, Tuple, Union
import time
from joblib import Parallel, delayed
import albumentations
import cv2

# torch
from torch import nn, Tensor
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# customized
from hephaestus.data.ndpwrapper_v2 import Slide_ndpread

skin_augment_fn =  albumentations.Compose([
                           albumentations.Transpose(p=0.5),
                           albumentations.VerticalFlip(p=0.5),
                           albumentations.HorizontalFlip(p=0.5),
                           albumentations.transforms.Rotate(limit=45, border_mode=cv2.BORDER_WRAP, p=0.5),
                           albumentations.imgaug.transforms.IAAAdditiveGaussianNoise(p=0.1),
                           #albumentations.augmentations.transforms.MultiplicativeNoise (multiplier=(0.95, 1.05), elementwise=True, p=0.5),
                           albumentations.augmentations.transforms.HueSaturationValue(hue_shift_limit=15, 
                                                                                      sat_shift_limit=15, 
                                                                                      val_shift_limit=15, p=0.3)
                           ])

imagenet_preproc = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

class Dataset:
    def __init__(self, slide_dir: str, 
                 target_slide_names: list, 
                 label_path: str, 
                 extension: str = '.ndpi',
                 img_sz:int = 512,
                 num_cls:int = 10,
                 num_slide_hold:int = 5,
                 interest_id_list:list = None,
                 bbox_dir: str = None,
                 aug_fn: Callable=None, 
                 preproc_fn: Callable=None,
                 debug = False):
        self.slide_dir          = slide_dir
        self.target_slide_names = target_slide_names
        self.label_path         = label_path
        self.extension          = extension
        self.img_sz             = img_sz
        self.num_cls            = num_cls
        self.num_slide_hold     = min(num_slide_hold, len(target_slide_names))
        self.interest_id_list   = interest_id_list
        self.bbox_dir           = bbox_dir
        self.aug_fn             = aug_fn
        self.preproc_fn         = preproc_fn
        self.debug              = debug
        
        self.cur_slide_pos = 0
        self.opened_slides = []
        
        self.slide_list = [] # which No. on the holden slides e.g. 0~4 
        self.bboxs_list = []
        self.label_list = []
        
        self._load_bbox_and_label()
        self._open_slide()
    
    
    def _load_bbox_and_label(self):
        for i in range(self.num_slide_hold):
            name = self.target_slide_names[(self.cur_slide_pos + i) % len(self.target_slide_names)]
            bbox_path = os.path.join(self.bbox_dir, name.split(self.extension)[0]+".npy")
            if not os.path.isfile(bbox_path):
                raise ValueError(f'{bbox_path} doesn\'t exist, please generate bbox first.')
            self._get_one_slide_bbox_label(bbox_path, i)
    
    def _open_slide(self):
        for slide in self.opened_slides:
            slide.close()
            del slide
        self.opened_slides.clear()
        for i in range(self.num_slide_hold):
            name = self.target_slide_names[(self.cur_slide_pos + i) % len(self.target_slide_names)]
            self.opened_slides.append(Slide_ndpread(os.path.join(self.slide_dir, name), show_info=False))
        print(f'Holding {len(self.opened_slides)} new slides.')
    
    def fetch_new_slide(self):
        self.cur_slide_pos = (self.cur_slide_pos +self.num_slide_hold) %len(self.target_slide_names)
        self.slide_list.clear()
        self.bboxs_list.clear()
        self.label_list.clear()
        
        self._load_bbox_and_label()
        self._open_slide()
        
    
    def _get_one_slide_bbox_label(self, bbox_path, idx):
        bbox_label_info = np.load(bbox_path)
        for i in range(bbox_label_info.shape[0]): 
            if bbox_label_info[i, 2] < self.num_cls:
                self.slide_list.append(idx)
                self.bboxs_list.append(tuple(bbox_label_info[i, :2]))
                self.label_list.append(bbox_label_info[i, 2])
    
    def __getitem__(self, idx):
        assert idx < len(self), f'Index out of bound, {idx} is larger than {len(self)}!'
        slide = self.opened_slides[self.slide_list[idx]]
        x, y  = self.bboxs_list[idx]
        label = self.label_list[idx]
        
        img = slide.get_patch_at_level((x, y), (self.img_sz, self.img_sz))
        
        if self.aug_fn is not None:
            img = self.aug_fn(image=img)['image']
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)/255.
        label = np.eye(self.num_cls, dtype='uint8')[label]
        img, label = torch.from_numpy(img), torch.tensor(label, requires_grad=False).float()
        
        if self.preproc_fn is not None:
            img = self.preproc_fn(img)
        
        return img, label
    
    def __len__(self):
        if self.debug:
            return 16
        return len(self.bboxs_list)

if __name__ == '__main__':
    from .config import get_cfg_defaults
    cfg = get_cfg_defaults()
    train_dataset = Dataset(slide_dir=cfg.DATASET.SLIDE_DIR,
                            target_slide_names = cfg.DATASET.TRAIN_SLIDE,
                            label_path = cfg.DATASET.LABEL_PATH,
                            bbox_dir= cfg.DATASET.BBOX_PATH,
                            aug_fn=skin_augment_fn,
                            preproc_fn=imagenet_preproc,
                            debug = cfg.SOURCE.DEBUG,
                            )
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=cfg.MODEL.BATCH_SIZE, 
                              shuffle=True,
                              num_workers=4,
                              drop_last=False,)
    
    for i in range(16):
        print(train_dataset[i][1])
    for imgs, labels in train_loader:
        print(labels)
        break