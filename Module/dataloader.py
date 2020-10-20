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
                 patch_size,
                 interest=None,
                 preproc=None,
                 augment=None,
                 shuffle=True,
                 num_worker=10,
                 save_bbox = True,
                 multiscale=0
                ):
        ''' Arg: slide_path: string, the parent folder of slide
                 slide_name: string, the name of the slide
                 label_path: string, the json file containing the contour, label information
                 frequency_dict: a list of int, indicating sampling frequency of each class
                 class_map: a dictionary, from json label(a randomly assigned number) to class(0~9)
                 patch_size: a 2-tuple(512, 512)
                 interest: list of int, the label of interest, this model will omit some label e.g. ROI, smooth muscle
                 preproc: preprocessing function, such as resnet_preprocess
                 augment: list, a combination of random flip, translation, rotation
                 shuffle: bool, whether to shuffle the bboxes and labels
                 num_worker: int, the num_worker used in parallelize generating bbox
                 save_bbox: bool, whether to save bbox or not
                 multiscale: concatenate 512*512 image and resized 4096*4096 image at color channel
        '''
        self.slide_path=slide_path
        self.slide_name=slide_name
        self.label_path=label_path
        self.frequency_dict=frequency_dict
        self.class_map=class_map
        self.patch_size=patch_size
        self.interest=interest
        self.preproc_fn = preproc
        self.augment_fn = augment
        self.shuffle = shuffle
        self.num_worker = num_worker
        self.num_class = len(class_map)
        self.save_bbox = save_bbox
        self.multiscale=multiscale
        
        self.cur_pos    = 0
        self.this_slide =Slide_ndpread(os.path.join(slide_path, slide_name), show_info=False) 
        self.slide_w, self.slide_h = self.this_slide.get_size()
        
        # get bboxs list
        '''
        bboxs_list: [(x, y), (x2, y2)......]
        label_list: [1     , 1       ......]
        '''
        self.bboxs_list=[]
        self.label_list=[]

        save_dir  = "/workspace/skin/bbox/"
        this_name = self.slide_name.split(".ndpi")[0] if self.slide_name.endswith(".ndpi") else self.slide_name
        bbox_isload = False
        if os.path.isfile(os.path.join(save_dir, this_name+".npy")):
            print("==============Loading bboxs==============")
            tmp = np.load(os.path.join(save_dir, this_name+".npy"))
            self.bboxs_list = [tuple(tmp[i, :2]) for i in range(tmp.shape[0]) if tmp[i, 2] < self.num_class]
            self.label_list = [tmp[i, 2] for i in range(tmp.shape[0]) if tmp[i, 2] < self.num_class]
            bbox_isload=True
        else:
            self._get_bboxs_list(verbose=True)

        if save_bbox and not bbox_isload:
            print("===============Saving bboxs===============")
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            save_list = [list(self.bboxs_list[i]) for i in range(len(self.bboxs_list))]
            for i in range(len(save_list)):
                save_list[i].append(self.label_list[i])
            save_list = np.array(save_list)
            save_path = os.path.join(save_dir, this_name+".npy")
            np.save(save_path, save_list)
        
        if shuffle:
            self._shuffle()
        
    
    def _get_bboxs_list(self, verbose=False):
        ''' get all contours and labels first
            generate random bboxes for each contour according to frequency map
             using joblib parallel
        '''
        _labels   = []
        _contours = []
        with open(self.label_path) as f:
            _data = json.load(f)
            key = self.slide_name.split(".ndpi")[0]
            if verbose:
                count = 0
                start_time = time.time()
            num_seg = len(_data[key][0]['targets']) if type(_data[key]) is list else len(_data[key]['targets'])
            for i in range(num_seg):
                _target=_data[key][0]['targets'][i] if type(_data[key]) is list else _data[key]['targets'][i]
                if len(_target['labels'])==0:
                    continue
                if len(_target['segments'])<10: # and len(_target['segments']) != 4:
                    continue
                if self.interest is not None and _target['labels'][0]['label'] not in self.interest:
                    # ignore non-interested class due to its small size
                    continue
                else:
                    _labels.append( _target['labels'][0]['label'])
                    _contours.append( np.array(_target['segments'], dtype=np.int32))
                    if verbose:
                        count += self.frequency_dict[_target['labels'][0]['label']]
        
        # results is a tuple of lists that returned after each parallel function calls
        # each list contains several bboxs/labels from the same contour
        results = Parallel(n_jobs=self.num_worker)(delayed(self._random_sample_from_contour)(_contours[i], self.class_map[_labels[i]], 
                                  self.patch_size, self.frequency_dict[_labels[i]]) for i in range(len(_labels)))
        for bboxs_, labels_ in results:
            self.bboxs_list.extend(bboxs_)
            self.label_list.extend(labels_)
        if verbose:
            end_time = time.time()
            print("Time elapse: generate {:4d} random patches in {:.4f} sec".format(count, end_time-start_time))
        
    def __getitem__(self, index):
        # get the image from index-th bbox and label
        bbox =self.bboxs_list[index]
        label=self.label_list[index]
        img = self.this_slide.get_patch_at_level(coord=(bbox[0],bbox[1]), sz=self.patch_size, level = 0)
        if self.multiscale != 0:
            multiscale = self.multiscale
            assert multiscale > self.patch_size[0] and multiscale > self.patch_size[1]
            thumbnail = self.this_slide.get_patch_with_resize(coord=(bbox[0]+self.patch_size[0]//2-multiscale, bbox[1]+self.patch_size[1]//2-multiscale), 
                                                              src_sz=(multiscale, multiscale), 
                                                              dst_sz=self.patch_size)
            img = np.concatenate((img, thumbnail), axis=-1)
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
    
    def _random_sample_from_contour(self, contour, label, size, num, robust=True):
        '''
        Arg: contour: list of tuple
             label: 0~9
             size: (512, 512)
             num: number of bbox sampled from this contour
             robust: bool, if False, will the inner point by interpolating two random contour points
        '''
        if not isinstance(contour, np.int32):
            contour = contour.astype(np.int32)
        # matplotlib version: 16 sec for random generating 93 images
        if not robust:
            p_list = get_point_in_polygon(contour, num, ww=self.slide_w, hh=self.slide_h, x=0, y=0)
        else: 
            p_list = get_point_in_polygon_robust(contour, num, ww=self.slide_w, hh=self.slide_h, x=0, y=0)
        bboxs  = get_bounding_box(p_list, (self.this_slide.get_size()), size=size)
        return (bboxs, [label for i in range(num)])
    
    def close(self):
        self.close_flag = True
        self.this_slide.close()
        self.cur_pos    = 0



class DataLoader(Sequence):
    def __init__(self, 
                 datasets,
                 num_slide,
                 batch_size=32, 
                 num_worker=10,
                 ):
        """"
        Arg: 
            datasets: a lists of SlideDataset object
            batch_size: default 32 per batch
            num_slide: randomly get num_slide datasets from directory
            Will keep a list of (dataset id, i-th item of this dataset), shuffle it and 
            take batches of images from this list.
        """
        self.datasets      = datasets
        self.batch_size    = batch_size
        self.num_slide     = num_slide
        
        # sequence will record i-th dataset j-th patch
        self._sequence = [(i, j) for i, dataset in enumerate(self.datasets) for j in range(len(dataset))]
        random.shuffle(self._sequence)

    def __getitem__(self, index):
        '''
        Function: return a batch of image, label 
        '''
        # return batches of images and labels
        batch_x = []
        batch_y = []

        end = min(self.batch_size*(index+1), len(self._sequence))
        selected_index = self._sequence[self.batch_size*index: end]
        for i in range(len(selected_index)):
            dataset_no, patch_no = selected_index[i]
            assert patch_no < len(self.datasets[dataset_no])
            img, label = self.datasets[dataset_no][patch_no]
            batch_x.append(img)
            batch_y.append(label)
        img_list = np.array(batch_x)
        lab_list = np.array(batch_y)
        return img_list, lab_list
        
    def on_epoch_end(self):
        """
        Method called at the end of every epoch. shuffle?
        """
        random.shuffle(self._sequence)
    
    def pack_data(self):
        '''
        Return:
            imgs: np array with shape (N, H, W, C), last channel RGB
            label: np array with shape(N, num_class), transformed to categorical(binary)
        '''
        imgs   = []
        labels = []
        for dataset in self.datasets:
            for img, label in dataset:
                imgs.append(img)
                labels.append(label)
        return np.array(imgs), np.array(labels)
    
    def get_labels(self):
        return [self.datasets[i].label_list[j] for i, j in self._sequence]

    def __len__(self):
        '''
        Return: total number of patches in datasets
        '''
        return math.ceil(sum([len(dataset) for dataset in self.datasets])*self.num_slide*1./(self.batch_size*len(self.datasets)))

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    from config import get_cfg_defaults

    cfg = get_cfg_defaults()

    #class_map      = { 80: 0, 56: 1, 43: 2, 53: 3, 45: 4, 42: 5, 54: 6, 41: 7, 55: 8 }
    #frequency_dict = { 80: 2, 56: 1, 43: 1, 53: 1, 45: 1, 42: 1, 54: 1, 41: 1, 55: 7 }
    #train_slides   = ["2019-10-30 02.05.46.ndpi",
    #                  "2019-10-30 02.04.50.ndpi",
    #                  "2019-10-30 02.09.05.ndpi"]
    
    #valid_slides   = ["2019-10-30 02.03.40.ndpi",
    #                  "2019-10-30 02.05.46.ndpi"]

    # prepare label dictionary and frequency dictionary
    train_histogram = {}
    train_slides=cfg.DATASET.TRAIN_SLIDE
    valid_histogram = {}
    valid_slides=cfg.DATASET.VALID_SLIDE
    class_map  = get_class_map(cfg.DATASET.CLASS_MAP)
    with open(cfg.DATASET.JSON_PATH) as f:
        data = json.load(f)
        for key, val in data.items():
            if type(data[key]) is list:
                targets = data[key][0]['targets']
            elif type(data[key]) is dict:
                targets = data[key]["targets"]
            else:
                print("Indeciphorable json file!")
                raise ValueError
            if key+".ndpi" in train_slides:
                collect_histogram(targets, train_histogram, interest=cfg.DATASET.INT_TO_CLASS)
            if key+".ndpi" in valid_slides:
                collect_histogram(targets, valid_histogram, interest=cfg.DATASET.INT_TO_CLASS)
    
    upsample = 1 if cfg.DATASET.INPUT_SHAPE[0] >=1024 else 4
    train_frequency = get_frequency_dict(train_histogram, upsample=upsample)
    valid_frequency = get_frequency_dict(valid_histogram, upsample=upsample)
    

    train_datasets =[SlideDataSet(slide_path=cfg.DATASET.SLIDE_DIR,
                                    slide_name=cfg.DATASET.TRAIN_SLIDE[i],
                                    label_path=cfg.DATASET.JSON_PATH,
                                    frequency_dict=train_frequency,
                                    class_map=class_map,
                                    patch_size=cfg.DATASET.INPUT_SHAPE[:2],
                                    interest = cfg.DATASET.INT_TO_CLASS,
                                    num_worker=10,
                                    preproc=preproc_resnet if cfg.DATASET.PREPROC else None,
                                    augment=None,
                                    save_bbox=True) for i in range(len(cfg.DATASET.TRAIN_SLIDE))]
    train_loader = DataLoader(datasets=train_datasets, 
                              batch_size=cfg.MODEL.BATCH_SIZE)
    
    
    def iterate_one_epoch(loader):
        start_time = time.time()
        for i in range(len(loader))[:5]:
            imgs, labels = loader[i]
            print(f"Image shape = {imgs.shape}, Label = {np.argmax(labels, axis=1)}")
        end_time = time.time()
        # print("\nTime elapse for iterating dataloader: {:.4f}\n".format(end_time-start_time))
    
    #for i in range(10):
    #    iterate_one_epoch(testloader)

    print(f"Train loader length = {len(train_loader)}")
    iterate_one_epoch(train_loader)
    