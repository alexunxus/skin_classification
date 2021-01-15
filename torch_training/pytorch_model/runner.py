from skimage.color import rgb2hsv
import numpy as np
import time
import os
from tqdm import tqdm
import math
import typing
from typing import Tuple, Callable, List, Optional
from scipy import ndimage

# torch
from torch.utils.data import DataLoader
import torch
from torch import nn

# customized libraries
from hephaestus.data.ndpwrapper_v2 import Slide_ndpread

class InfDataset:
    def __init__(self,
                 slide_dir: str,
                 slide_name: List[str],
                 patch_size: Tuple[int],
                 preproc_fn: Callable, 
                 hsv_threshold: Optional[float]=0.05):
        '''
        Arg: slide_dir: string, slide directory
             slide_name: string, slide name
             patch_size: tuple of int, (512, 512)
             hsv_threshold: float, the threshold for filtering background in thumbnail
        The inference dataset will collect the patches which is not judged as background in orthodoxical
        mesh coordinate and shifted coordinate(with a 0.5* patch_size shift, who is sized h+1, w+1).
        The image will first be resized to (h=H//512, w=W//512) and use hsv channel to filter out 
        the background (so does the shifted mesh coordinated by 0.5*patch_size), the non-backgorund 
        patches will be loaded together with the resized coordinate(h, w). 

        '''
        self.this_slide = Slide_ndpread(os.path.join(slide_dir, slide_name))
        self.hsv_threshold = hsv_threshold
        self.patch_size = patch_size
        self.preproc_fn = preproc_fn
        
        W, H = self.this_slide.get_size()
        w, h = patch_size[0], patch_size[1]
        self.W = W
        self.H = H
        
        self.w_stride = W//w if W%w==0 else W//w+1
        self.h_stride = H//h if H%h==0 else H//h+1
        
        assert W//w*w <= W and H//h*h <= H
        self.tiny_slide = self.this_slide.get_patch_with_resize(coord=(0,0),
                                                                src_sz = (W//w*w, H//h*h),
                                                                dst_sz = (W//w, H//h)
                                                               )
        self.tiny_slide = np.pad(self.tiny_slide, 
                                 ((0, self.h_stride-self.tiny_slide.shape[0]),
                                  (0, self.w_stride-self.tiny_slide.shape[1]), 
                                  (0, 0)), 
                                 mode="reflect")
        
        # get non-background patches and their
        self.object_mask     = None
        self.five_crop_mask  = None
        self._judge_bg()
        self._get_crop_mask()
        
        # collect all patches that will be inferenced on
        # labels: (hi, wi)
        self.coords = None
        self._collect_label()
        
        # current position
        self.cur_pos = 0
    
    def _collect_label(self):
        xs,  ys  = np.where(self.object_mask > 0)
        xss, yss = np.where(self.five_crop_mask > 0)
        xss = xss.astype(np.float32) - 0.5
        yss = yss.astype(np.float32) - 0.5
        xs  = np.concatenate([xs.astype(np.float32), xss], axis=-1)
        ys  = np.concatenate([ys.astype(np.float32), yss], axis=-1)
        self.coords = list(zip(xs, ys))
        
    def _judge_bg(self, expand=True):
        hsv_img = rgb2hsv(self.tiny_slide)
        saturation_img = hsv_img[:, :, 1]
        boolean_mask = np.array(saturation_img>self.hsv_threshold, dtype=np.int32)
        if not expand:
            self.object_mask = boolean_mask
            return
        # dilate the mask
        struct2 = ndimage.generate_binary_structure(2, 2)
        self.object_mask = ndimage.morphology.binary_dilation(boolean_mask, struct2).astype(np.bool)
        
    
    def _get_crop_mask(self):
        '''
        Get the object mask in shifted coordinate, which can be fast acquired by shifting the
        original object masks by (0, 0), (1, 0)(0, 1)(1, 1), and then superimposing them. 
        '''
        self.five_crop_mask = np.zeros((self.object_mask.shape[0]+1, self.object_mask.shape[1]+1))
        dirs = [((0, 1), (0, 1)), ((0, 1), (1, 0)), ((1, 0), (0, 1)), ((1, 0), (1, 0))]
        for dir in dirs:
            self.five_crop_mask += np.pad(self.object_mask, dir, 'constant', constant_values=(0, 0))
    
    def __getitem__(self, idx):
        h, w = int(self.coords[idx][0]*self.patch_size[1]), int(self.coords[idx][1]*self.patch_size[0])
        img = self.this_slide.get_patch_at_level((w, h), self.patch_size)

        img = np.transpose(img, (2, 0, 1)).astype(np.float32)/255.
        img = torch.from_numpy(img)
        if self.preproc_fn is not None:
            img = self.preproc_fn(img)        
        return img, np.array(list(self.coords[idx]))
    
    def get_coords(self):
        ''' return the coordinate list'''
        return self.coords
    
    def get_mask(self):
        ''' return the mask numpy array sized (h, w), dtype=int'''
        return np.copy(self.object_mask)
    
    def __next__(self):
        ''' get next element '''
        return self.__getitem__(self.cur_pos)
        
    def __len__(self):
        ''' return the length of object patches'''
        return len(self.coords)
    
    def get_shape(self):
        ''' return the shape of the original slide'''
        return self.W, self.H

class InferenceRunner:
    def __init__(self, dataset, model, center_weight=2, batch_size=32):
        '''
        Arg: dataset: inference dataset
             model: deep learning framework
             center_weigth: the center weight during performing five-crop
             batch_size: the batch size during inferencing
        function: Will produce a (h, w, num_class) np array which satisfy the PRPrun format
        I use five-crop method during inferencing time to eliminate the outlier in a large area of 
        the same tissue. If the center weight is large, the outcome will approach not using the 
        five-crop augmentation. On the other hand, if the center weight is small, the inference outcome
        of the four corners neighboring the target patch will smooth the inference outcome. e.g. provide
        more spatial information and optimize the disparation of har follicles and epidermis.
        '''
        inference_dataset = dataset
        inference_loader  = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
        self.output = InferenceRunner.predict(model, inference_loader)
        self.coords = inference_dataset.get_coords()
        
        self.center_weight = center_weight
        self.object_mask = inference_dataset.object_mask
        self.H = self.object_mask.shape[0]
        self.W = self.object_mask.shape[1]
        self.dirs = [(0, 0), (1, 0), (0, 1), (1, 1)]
        assert self.output.shape[0] == len(self.coords)
        self.heatmap = np.zeros((self.H, self.W, self.output.shape[1]), dtype=np.float64)
        background_mask = np.logical_not(self.object_mask)
        self.heatmap[background_mask, 0] = 1
        self.fill_in_probability()
    
    @staticmethod
    def predict(model: nn.Module, loader: DataLoader):
        ret = []
        with torch.no_grad():
            for imgs, labels in tqdm(loader):
                imgs = imgs.cuda()
                pred = model(imgs)
                out  = pred.detach().cpu().numpy()
                ret.append(out)
        ret = np.concatenate(ret, axis=0)
        return ret
        
    def fill_in_probability(self):
        '''
        fill in the probability map from inference result, weight the center patch with 
        patches centering on the four corners by center weight = self.center_weight
        '''
        corner_heatmap = np.zeros((self.object_mask.shape[0]+1, 
                                   self.object_mask.shape[1]+1, 
                                   self.output.shape[1]), dtype=np.float64)
        for i, (h, w) in enumerate(self.coords):
            h2 = int(h*2)
            w2 = int(w*2)
            if (h2)%2 == 0 and (w2)%2 == 0:
                h, w = h2//2, w2//2
                assert self.heatmap[h, w, 2] == 0
                self.heatmap[h, w] = np.copy(self.output[i])
            elif (h2)%2 == 1 and (w2)%2 == 1:
                h, w = (h2+1)//2, (w2+1)//2
                assert corner_heatmap[h, w, 0] == 0
                corner_heatmap[h, w] = np.copy(self.output[i])
            else:
                print(f"Coordinate {h}, {w} is not found!")
                raise ValueError
        hs, ws = np.where(self.object_mask > 0)
        for i in range(hs.shape[0]):
            h, w = hs[i], ws[i]
            self.heatmap[h, w] *= self.center_weight
            for dh, dw in self.dirs:
                self.heatmap[h, w] += corner_heatmap[h+dh, w+dw]
            self.heatmap[h, w] /= (self.center_weight+4)
    
    def get_heatmap(self):
        ''' return a np array of heatmap (h, w, num_class)'''
        return self.heatmap
        #return self.heatmap.reshape((-1, self.heatmap.shape[-1]))
    