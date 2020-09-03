from hephaestus.data.ndpwrapper_v2 import Slide_ndpread
from skimage.color import rgb2hsv
import numpy as np
import time
import os

class SlidePredictor:
    def __init__(self, 
                 bbox_shape, 
                 slide_dir,
                 slide_name,
                 histologic_name,
                 classifier,
                 class_map,
                 batch_size=32
                ):
        self.bbox_shape = bbox_shape
        self.slide_dir  = slide_dir
        self.slide_name = slide_name
        self.histologic_name = histologic_name
        self.classifier = classifier
        self.batch_size = batch_size
        self.colormap   = [
            "tomato",
            "chocolate",
            "tan",
            "gold",
            "olive",
            "palegreen",
            "cyan",
            "navy",
            "white"
        ]
        if class_map is None:
            self.class_map = {
                80: 0, 
                56: 1, 
                43: 2, 
                53: 3, 
                45: 4, 
                42: 5,
                54: 6, 
                41: 7
                #(55, 8)
            }
        else:
            self.class_map=class_map
        
        self.this_slide = Slide_ndpread(os.path.join(slide_dir, slide_name))
        W, H = self.this_slide.get_size()
        w, h = bbox_shape[0], bbox_shape[1]
        self.W = W
        self.H = H
        
        self.w_stride = W//w if W%w==0 else W//w+1
        self.h_stride = H//h if H%h==0 else H//h+1
        self.prob_map = np.zeros((self.w_stride, self.h_stride, len(self.class_map)), dtype=np.float32)
        self.tiny_slide = self.this_slide.get_patch_with_resize(coord=(0, 0), 
                                            src_sz=(W, H),
                                            dst_sz=(self.w_stride, self.h_stride))
        self.background_mask = self._judge_bg()
        
        #self._background_test()
        self._get_prob_map()
    
    def _judge_bg(self):
        hsv_img = rgb2hsv(self.tiny_slide)
        saturation_img = hsv_img[:, :, 1]
        return np.array(saturation_img<0.05)
        
    
    def _get_prob_map(self):
        # 277*76
        patches = []
        coords  = []
        for i in range(self.w_stride):
            begin_time = time.time()
            for j in range(self.h_stride):
                patch = self.this_slide.get_patch_at_level((512*i, 512*j), 
                                                                  self.bbox_shape)/255.
                if not self.background_mask[j, i]:
                    patches.append(patch)
                    coords.append((i, j))
                    if len(patches)%self.batch_size==31:
                        preds = self.classifier(np.array(patches, dtype=np.float32)).numpy()
                        for idx, coord in enumerate(coords):
                            self.prob_map[coord[0], coord[1]] = np.copy(preds[idx])
                        patches.clear()
                        coords.clear()
            end_time = time.time()
            print("Column {:}/{:}: time elapse {:}".format(i, self.w_stride, end_time-begin_time), end='\r')
        
        if len(patches):
            preds = self.classifier(np.array(patches, dtype=np.float32)).numpy()
            for idx, coord in enumerate(coords):
                self.prob_map[coord[0], coord[1]] = np.copy(preds[idx])
            patches.clear()
            coords.clear()
        
        predicted_cls = np.array([[np.argmax(self.prob_map[i,j]) for i in range(self.prob_map.shape[0])] 
                         for j in range(self.prob_map.shape[1])])
        predicted_cls[self.background_mask] = len(self.class_map)
        
    def _background_test(self):
        '''
            normalized to 0~1:
            background std: 0.0017
            background mean: 0.93
        '''
        std  = []
        mean = []
        for i in range(0, self.w_stride, 10):
            for j in range(self.h_stride):
                patch = self.this_slide.get_patch_at_level((512*i, 512*j), self.bbox_shape)/255.
                std.append(np.std(patch))
                mean.append(np.mean(patch))
        print("{:}: std median {:04f}, mean median {: 03f}".format(self.slide_name,
                                                                  np.median(np.array(std)),
                                                                  np.median(np.array(mean))))
        
            
    def visualize_predict(self, axis):
        predicted_cls = np.array([[np.argmax(self.prob_map[i,j]) for i in range(self.prob_map.shape[0])] 
                         for j in range(self.prob_map.shape[1])])
        predicted_cls[self.background_mask] = len(self.class_map)
        
        w_s, h_s = self.w_stride, self.h_stride
        x = np.array([i for j in range(h_s)for i in range(w_s)])
        y = np.array([j for j in range(h_s)for i in range(w_s)])
        assert len(x) == len(y)
        classes = [predicted_cls[y[i], x[i]] for i in range(len(x))]
        for cls in np.unique(classes):
            ix = np.where(classes == cls)
            axis.scatter(x[ix], y[ix], c = self.colormap[cls], 
                         label = self.histologic_name[cls], marker='s', s=8)
        axis.legend(loc="lower right", bbox_to_anchor=(1,1), ncol=4, fontsize=10, markerscale=2)
        #axis.scatter(x, y, color=color)
        axis.set_xlim([0, w_s])
        axis.set_ylim([0, h_s])
    
    def get_np_pred(self):
        '''
        self.pro_map is in shape (w, h, 9)
        however, the PRP run session require flattened (h, w)
        '''
        back  = np.zeros((self.w_stride, self.h_stride, 1))
        back[self.background_mask.transpose()] = 1
        ret = np.concatenate([self.prob_map, back], axis=2)
        return ret.transpose((1,0,2)).reshape((-1, ret.shape[2]))
    
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
