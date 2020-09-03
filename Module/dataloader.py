# simple loader
import os
import random
from hephaestus.data.ndpwrapper_v2 import Slide_ndpread
import time
import math
import json
import numpy as np
try:
    from Module.util import *
except:
    from util import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
from joblib import Parallel, delayed


class SlideDataSet:
    def __init__(self,
                 slide_path,
                 slide_name,
                 label_path,
                 frequency_dict,
                 class_map,
                 preproc=None,
                 augment=None,
                 shuffle=True,
                 patch_size=(512,512),
                 num_worker=10
                ):
        self.slide_path=slide_path
        self.slide_name=slide_name
        self.label_path=label_path
        self.frequency_dict=frequency_dict
        self.class_map=class_map
        self.preproc_fn = preproc
        self.augment_fn = augment
        self.shuffle = shuffle
        self.patch_size=patch_size
        self.num_worker = num_worker
        self.num_class = len(class_map)
        
        self.cur_pos    = 0
        self.this_slide =Slide_ndpread(os.path.join(slide_path, slide_name), show_info=False) 
        self.slide_w, self.slide_h = self.this_slide.get_size()
        
        # get bboxs list
        self.bboxs_list=[]
        self.label_list=[]
        self._get_bboxs_list(verbose=True)
        
        if shuffle:
            self._shuffle()
        
    
    def _get_bboxs_list(self, n_worker=10, verbose=False):
        _labels   = []
        _contours = []
        with open(self.label_path) as f:
            _data = json.load(f)
            key = self.slide_name.split(".ndpi")[0]
            if verbose:
                count = 0
                start_time = time.time()
            for i in range(len(_data[key][0]['targets'])):
                _target=_data[key][0]['targets'][i]
                if len(_target['labels'])==0:
                    continue
                if len(_target['segments'])<10:
                    continue
                if _target['labels'][0]['label'] == 55:
                    # ignore blood vessel class due to its small size
                    continue
                else:
                    _labels.append( _target['labels'][0]['label'])
                    _contours.append( np.array(_target['segments'], dtype=np.int32))
                    if verbose:
                        count += self.frequency_dict[_target['labels'][0]['label']]
        
        results = Parallel(n_jobs=self.num_worker)(delayed(self._random_sample_from_contour)(_contours[i], self.class_map[_labels[i]], 
                                 self.frequency_dict[_labels[i]]) for i in range(len(_labels)))
        for bboxs_, labels_ in results:
            self.bboxs_list.extend(bboxs_)
            self.label_list.extend(labels_)
        if verbose:
            end_time = time.time()
            print("Time elapse: generate {:4d} random patches: {:.4f} sec".format(count, end_time-start_time))
        
    def __getitem__(self, index):
        # should refill all the Contour object 
        # if traverse a cycle of contour_list, should reset all flag of contour
        bbox =self.bboxs_list[index]
        label=self.label_list[index]
        img = self.this_slide.get_patch_at_level(coord=(bbox[0],bbox[1]), sz=self.patch_size, level = 0)
        img = img.astype(np.float32)
        img /= 255.
        if self.augment_fn is not None:
            img = self.augment_fn.augment_image(img)

        if self.preproc_fn is not None:
            img = self.preproc_fn(img)
        self.cur_pos = (self.cur_pos+1)%len(self)
        return img, to_categorical(label, self.num_class)
            
    def __len__(self):
        return len(self.bboxs_list)
    
    def __next__(self):
        return self.__getitem__(self.cur_pos)
    
    def _shuffle(self):
        tmp = list(zip(self.bboxs_list, self.label_list))
        random.shuffle(tmp)
        self.bboxs_list, self.label_list=zip(*tmp)
    
    def _random_sample_from_contour(self, contour, label, num=1):
        if not isinstance(contour, np.int32):
            contour = contour.astype(np.int32)
        # matplotlib version: 16 sec for random generating 93 images
        p_list = get_point_in_polygon(contour, num, ww=self.slide_w, hh=self.slide_h, x=0, y=0)
        bboxs  = get_bounding_box(p_list, (self.this_slide.get_size()), size=(512,512))
        return (bboxs, [label for i in range(num)])
    
    def close(self):
        self.close_flag = True
        self.this_slide.close()
        self.cur_pos    = 0
    
    def residual_length(self):
        return self.__len__()-self.cur_pos


class DataLoader(Sequence):
    def __init__(self, 
                 datasets_dir, 
                 valid_slides,
                 label_path, 
                 frequency_dict, 
                 class_map, 
                 preproc_fn=None,
                 augment_fn=None,
                 batch_size=32, 
                 num_slide=5,
                 num_worker=10):
        """"
        Arg: 
            datasets_dir: a lists of datasets names
            valid_slides: only choose slide from this list of string
            label_path: the label json file path
            batch_size: default 32 per batch
            num_slide: randomly get 5 datasets from directory
            class_map: map from class num to (0~8)
            preproc_fn: preprocessing function
            augment_fn: augmentation function
        """
        self.datasets_dir  = datasets_dir
        self.valid_slides  = valid_slides
        self.label_path    = label_path
        self.datasets      = []
        self.batch_size    = batch_size
        self.num_of_slide_hold = num_slide
        self.frequency_dict= frequency_dict
        self.class_map     = class_map
        self.num_worker    = num_worker
        self.preproc_fn    = preproc_fn
        self.augment_fn    = augment_fn
        
        #generate datasets from valid slides
        self._gen_datasets()
        self._shuffle_datasets()

    def _gen_datasets(self):
        '''
        Function: generate all the available patches from each valid slides 
        Arg: nil
        Return: nil
        '''
        self.datasets=[SlideDataSet(slide_path=self.datasets_dir,
                                    slide_name=self.valid_slides[i],
                                    label_path=self.label_path,
                                    frequency_dict=self.frequency_dict,
                                    class_map=self.class_map,
                                    num_worker=self.num_worker,
                                    preproc=self.preproc_fn,
                                    augment=self.augment_fn) for i in range(len(self.valid_slides))]
        
    def __getitem__(self, index):
        '''
        Function: 
            1. It randomly selects 32 patches and 32 labels from datasets and 
               return them in a batch.
        Arg: 
            index -- no role in this function since the dataloader return data 
        '''
        # return batches of images and labels
        batch_x = []
        batch_y = []
        for i in range(self.batch_size):
            img, label = self._random_select_patch_()
            batch_x.append(img)
            batch_y.append(label)
        img_list = np.array(batch_x)
        lab_list = np.array(batch_y)
        return img_list, lab_list

    def _shuffle_datasets(self):
        random.shuffle(self.datasets)
        
    def on_epoch_end(self):
        """
        Method called at the end of every epoch.
        """
        self._shuffle_datasets()
    
    def pack_data(self):
        '''
        Return:
            imgs: np array with shape (N, 512, 512, 3) RGB
            label: np array with shape(N, 9), have been categorical
        '''
        imgs   = []
        labels = []
        for dataset in self.datasets:
            for img, label in dataset:
                imgs.append(img)
                labels.append(label)
        return np.array(imgs), np.array(labels)
        
    def _random_select_patch_(self, verbose=False):
        '''
        Function: 
            1. it randomly selects a (patch, label) pair from datasets
        '''
        n_slide = random.randint(0, len(self.datasets)-1)
        img, label = next(self.datasets[n_slide])
        
        return img, label
    
    def __len__(self):
        '''
        Return: (total patch * num_of_slide_hold / num_of_datasets) / 32
        '''
        return int(self.num_of_patches() * self.num_of_slide_hold/len(self.datasets)/self.batch_size)

    def _residual_length(self):
        '''
        Return: the sum of the residual number of patches
        '''
        ret = 0
        for dataset in self.datasets:
            ret += dataset.residual_length()
        return ret
    
    def num_of_patches(self):
        '''
        Return: total number of patches in datasets
        '''
        return sum([len(dataset) for dataset in self.datasets])

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    class_map      = { 80: 0, 56: 1, 43: 2, 53: 3, 45: 4, 42: 5, 54: 6, 41: 7, 55: 8 }
    frequency_dict = { 80: 2, 56: 1, 43: 1, 53: 1, 45: 1, 42: 1, 54: 1, 41: 1, 55: 7 }
    train_slides   = ["2019-10-30 02.05.46.ndpi",
                      "2019-10-30 02.04.50.ndpi",
                      "2019-10-30 02.09.05.ndpi"]
    
    valid_slides   = ["2019-10-30 02.03.40.ndpi",
                      "2019-10-30 02.05.46.ndpi"]

    testloader = DataLoader(datasets_dir="/mnt/cephrbd/data/A19001_NCKU_SKIN/Image/20191106/", 
                            valid_slides=train_slides,
                            label_path="/mnt/cephrbd/data/A19001_NCKU_SKIN/Meta/key-image-locations.json",
                            frequency_dict=frequency_dict,
                            class_map=class_map,
                            batch_size=32, num_slide=2)
    
    def iterate_one_epoch(loader):
        start_time = time.time()
        for i in range(len(testloader)):
            imgs, labels = testloader[i]
            print(imgs.shape)
            print(np.argmax(labels, axis=1),end='\r')
        loader.on_epoch_end()
        end_time = time.time()
        print("\nTime elapse for iterating dataloader: {:.4f}\n".format(end_time-start_time))
    
    #for i in range(10):
    #    iterate_one_epoch(testloader)


    imgs, labels = testloader.pack_data()
    print(imgs.shape, labels.shape)

    iterate_one_epoch(testloader)
    