from hephaestus.data.ndpwrapper_v2 import Slide_ndpread
from skimage.color import rgb2hsv
import numpy as np
import time
import os
from tqdm import tqdm

class SlidePredictor:
    def __init__(self, 
                 bbox_shape, 
                 slide_dir,
                 slide_name,
                 histologic_name,
                 classifier,
                 class_map,
                 batch_size=32,
                 fast=True,
                 five_crop = True,
                ):
        self.bbox_shape = bbox_shape
        self.slide_dir  = slide_dir
        self.slide_name = slide_name
        self.histologic_name = histologic_name
        self.classifier = classifier
        self.batch_size = batch_size
        self.five_crop  = five_crop
        self.colormap   = [(1, 1, 1),
                           (0.7098039215686275, 0.5333333333333333, 0.09411764705882353), 
                           (1.0, 0.3411764705882353, 0.13333333333333333), 
                           (0.9137254901960784, 0.11764705882352941, 0.38823529411764707), 
                           (0.803921568627451, 0.8627450980392157, 0.2235294117647059), 
                           (0.0, 0.0, 0.6), 
                           (0.2980392156862745, 0.6862745098039216, 0.3137254901960784), 
                           (0.611764705882353, 0.15294117647058825, 0.6901960784313725), 
                           (0.011764705882352941, 0.6627450980392157, 0.9568627450980393), 
                           (1, 0, 0),
                           (0.7098039215686275, 0.5333333333333333, 0.09411764705882353)
                           ]
        self.class_map=class_map

        self.this_slide = Slide_ndpread(os.path.join(slide_dir, slide_name))
        W, H = self.this_slide.get_size()
        w, h = bbox_shape[0], bbox_shape[1]
        self.W = W
        self.H = H
        
        self.w_stride = W//w if W%w==0 else W//w+1
        self.h_stride = H//h if H%h==0 else H//h+1
        self.prob_map = np.zeros((self.w_stride, self.h_stride, len(self.class_map)), dtype=np.float32)
        
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
        self.background_mask = np.zeros((self.h_stride, self.w_stride))
        self.fast = fast
        if fast:
            self.background_mask = self._judge_bg()
        self._get_prob_map()
        self._class_heatmap = np.argmax(self.prob_map, axis=-1)
    
    def _judge_bg(self, expand=True):
        hsv_img = rgb2hsv(self.tiny_slide)
        saturation_img = hsv_img[:, :, 1]
        boolean_mask = np.array(saturation_img<0.05, dtype=np.int32)
        if not expand:
            return boolean_mask
        for w in range(boolean_mask.shape[1]):
            meet = 0
            for h in range(boolean_mask.shape[0]):
                if boolean_mask[h, w] == 0:
                    meet = 2
                elif meet > 0:
                    boolean_mask[h, w] = 0
                    meet -= 1
            meet = 0
            for h in range(boolean_mask.shape[0])[::-1]:
                if boolean_mask[h, w] == 0:
                    meet = 2
                elif meet > 0:
                    boolean_mask[h, w] = 0
                    meet -= 1
        return boolean_mask
        
    def crop(self, x, y):
        # perform five crop at coordinate x, y (left-upper vertex)
        # return: 5 np-image list[center, LUQ, RUQ, LLQ, RLQ]
        patch_size = self.bbox_shape[0]
        dirs = [(0, 0), (-patch_size//2, -patch_size//2), (patch_size//2, -patch_size//2), (-patch_size//2, patch_size//2), (patch_size//2, patch_size//2)]
        ret = []
        for dx, dy in dirs:
            ret.append(self.this_slide.get_patch_at_level((x+dx, y+dy), self.bbox_shape)/255.)
        return ret

    def _get_prob_map(self):
        # 277*76
        patches = []
        coords  = []

        for i in tqdm(range(self.w_stride)):
            begin_time = time.time()
            for j in range(self.h_stride):
                if self.fast and self.background_mask[j, i] != 0:
                    continue
                else:
                    try:
                        # if the slide ndpi file is corrupted, getting some patches may raise error
                        if self.five_crop:
                            patch = self.crop(self.bbox_shape[0]*i, self.bbox_shape[1]*j)
                        else:
                            patch = self.this_slide.get_patch_at_level((512*i, 512*j), 
                                                                       self.bbox_shape)/255.
                    except:
                        self.background_mask[j, i] = 1
                    if not self.fast:
                        saturation = rgb2hsv(patch[0] if self.five_crop else patch)[..., 1].mean()
                        if saturation < 0.05:
                            self.background_mask[j, i] = 1
                if self.background_mask[j, i] == 0:
                    if isinstance(patch, list):
                        patches.extend(patch)
                        coords.extend([(i, j) for k in range(len(patch))])
                    else:
                        patches.append(patch)
                        coords.append((i, j))
                    if len(patches) >= self.batch_size:
                    #if len(patches)%self.batch_size==self.batch_size-1:
                        preds = self.classifier(np.array(patches, dtype=np.float32)).numpy()
                        if self.five_crop:
                            for idx, coord in enumerate(coords[::5]):
                                self.prob_map[coord[0], coord[1]] = np.copy(np.mean(preds[5*idx:5*(idx+1)], axis=0))
                        else:
                            for idx, coord in enumerate(coords):
                                self.prob_map[coord[0], coord[1]] = np.copy(preds[idx])
                        patches.clear()
                        coords.clear()
            end_time = time.time()
            #print("Column {:}/{:}: time elapse {:}".format(i, self.w_stride, end_time-begin_time), end='\r')
        
        if len(patches):
            preds = self.classifier(np.array(patches, dtype=np.float32)).numpy()
            if self.five_crop:
                for idx, coord in enumerate(coords[::5]):
                    self.prob_map[coord[0], coord[1]] = np.copy(np.mean(preds[5*idx:5*(idx+1)], axis=0))
            else:
                for idx, coord in enumerate(coords):
                    self.prob_map[coord[0], coord[1]] = np.copy(preds[idx])
            patches.clear()
            coords.clear()
        self.prob_map[self.background_mask.transpose().astype(bool), 0] = 1
    
    def _postprocess(self):
        '''
        Handle adipose tissue that is adjacent to epidermis
        '''
        adipose_mask = np.where(self._class_heatmap == 2)
            
    def visualize_predict(self, axis):
        predicted_cls = np.argmax(self.prob_map, axis=-1)
        # heatmap in shape[h, w, 3]
        heatmap = np.zeros((predicted_cls.shape[1], predicted_cls.shape[0], 3))
        for w in range(predicted_cls.shape[0]):
            for h in range(predicted_cls.shape[1]):
                heatmap[h, w] = np.array(self.colormap[int(predicted_cls[w][h])])
        heatmap = 0.5*heatmap + 0.5*self.tiny_slide/255.
        axis.imshow(heatmap)
    
    def get_np_pred(self):
        '''
        self.prob_map is in shape (w, h, 10)
        however, the PRP run session require flattened (h, w)
        '''
        return self.prob_map.transpose((1,0,2)).reshape((-1, self.prob_map.shape[2]))
    
    def get_shape(self):
        return self.W, self.H

if __name__ == "__main__":
    myPredictor = SlidePredictor(bbox_shape=(512,512),
                             slide_dir = "/mnt/cephrbd/data/A19001_NCKU_SKIN/Image/20191106/",
                             #slide_name = slide_name,
                             slide_name = "2019-10-30 02.23.07.ndpi",
                             histologic_name = type_dictionary,
                             classifier = model,
                             class_map = class_map
                            )
