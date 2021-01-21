import numpy as np
import random # shuffling
import pandas as pd
import os
import json
from skimage.color import rgb2hsv
import skimage
import cv2
import matplotlib.path as mpltPath # check path
import math
from bresenham import bresenham
import typing
from typing import Tuple, Callable

from torch import nn # replace bn

def replace_bn(model:nn.Module, new_norm: Callable) -> nn.Module:
    '''replace bn of model to gn'''
    for name, module in model.named_children():
        if len(list(module.named_children())):
            model._modules[name] = replace_bn(module, new_norm)
        elif type(module) == nn.BatchNorm2d:# or type(module) == nn.BatchNorm1d:
            layer_new = new_norm(module.num_features)
            #del model._modules[name]
            model._modules[name]=layer_new
    return model

def concat_tiles(tiles, patch_size:int, tile_sz:int=6) -> np.ndarray:
    '''concat tiles into (3, tile_sz*patch_size, tile_size*patch_size) image'''
    ret = np.ones((patch_size*tile_sz, patch_size*tile_sz, 3), dtype=np.uint8)*255
    if len(tiles) == 0:
        return ret
    for i in range(tile_sz**2):
        tile = tiles[i%len(tiles)]
        h, w = i//tile_sz, i%tile_sz
        ret[h*patch_size:h*patch_size + tile.shape[0], w*patch_size: w*patch_size+tile.shape[1]] = tile
    return ret

def shuffle_two_arrays(imgs: list, labels: list)->Tuple[list, list]:
    zip_obj = list(zip(imgs, labels))
    random.shuffle(zip_obj)
    return zip(*zip_obj)

def binary_search(li, x, lo: int, hi:int):
    mid = (lo+hi)//2
    if li[mid][0] == x:
        return li[mid][1]
    if mid == hi:
        raise ValueError(f"{x} is not in list!")
    if li[mid][0] > x:
        return binary_search(li, x, lo, mid)
    else:
        return binary_search(li, x, mid+1, hi)

def pad_background(img: np.ndarray, saturation_threshold: float = 0.1, 
                   resize_ratio :int =64)-> np.ndarray:
    '''
    Padding background to 255
    
    :Usage
        img = (np.rand(32, 32, 3)*255).astype(np.uint8)
        out = pad_background(img)
    
    :Parameters:
        channel last np.ndarray, dtype uint8, range 0~255
        saturation_threshold: cut-off for background filtering, default 0.1
    '''
    assert img.shape[0] == img.shape[1], (f'image shape mismatch! ({img.shape}), dim0 != dim1')
    resized_image = skimage.measure.block_reduce(img, (resize_ratio,resize_ratio, 1), np.mean)
    hsv_image = rgb2hsv(resized_image)
    foreground_mask = (hsv_image[...,1] > saturation_threshold)
    foreground_mask = skimage.transform.resize(foreground_mask, 
                                               output_shape=(i*resize_ratio for i in foreground_mask.shape))
    foreground_mask = np.repeat(np.expand_dims(foreground_mask, axis=-1), img.shape[2], axis=-1).astype(np.uint8)
    out_img = 255-(255-img)*foreground_mask
    return out_img

class Metric:
    def __init__(self, metric_keys:dict):
        self.metric_dict = {}
        self.register_keys(metric_keys)
    
    def register_keys(self, keys: list)->None:
        for key in keys:
            self.metric_dict[key] = []
    
    def load_metrics(self, csv_path: str, resume: bool, model_selection_criterion: str
                    ) -> Tuple[float, float, int]:
        
        resume_from_epoch = -1
        best_kappa        = -10
        best_loss         = 1000
        if not os.path.isfile(csv_path) or not resume:
            return best_kappa, best_loss, resume_from_epoch
        # if csv file exist, then first find out the epoch with best kappa(named resume_from_epoch), 
        # get the losses, kappa values within range 0~ best_epoch +1
        df = pd.read_csv(csv_path)
        for key in self.metric_dict.keys():
            if key not in df.columns:
                print(f"Key {key} not found in {df.columns}, not loading csv")
                return best_kappa, best_loss, resume_from_epoch

        test_criterion = list(df[model_selection_criterion])
        
        best_idx   = np.argmax(np.array(test_criterion))
        best_criterion = test_criterion[best_idx]
        best_loss  = min(list(df['test_losses'])[:best_idx+1])
        resume_from_epoch = best_idx+1

        for key in self.metric_dict.keys():
            self.metric_dict[key]= list(df[key])[:resume_from_epoch]

        print("================Loading CSV==================")
        print(f"|Loading csv from {csv_path},")
        print(f"|best test loss = {best_loss:.4f},")
        print(f"|best {model_selection_criterion}     = {best_criterion:.4f},")
        print(f"|epoch          = {resume_from_epoch:.4f}")
        print("=============================================")
        return best_criterion, best_loss, resume_from_epoch
    
    def save_metrics(self, csv_path: str, debug=False) -> None:
        df = pd.DataFrame(self.metric_dict)
        print(df)
        if not debug:
            df.to_csv(csv_path, index=False)
    
    def push_loss_acc(self, loss, acc, train=True):
        if train:
            self.metric_dict['train_losses'].append(loss)
            self.metric_dict['train_acc'].append(acc)
        else: # test/valid
            self.metric_dict['test_losses'].append(loss)
            self.metric_dict['test_acc'].append(acc)
    
    def push_auc_precision_recall_AP_f1(self, auc: np.float32, precision: np.float32, 
                                        recall:np.float32, AP: np.float32, f1: np.float32) -> None:
        self.metric_dict['auc'].append(auc)
        self.metric_dict['precision'].append(precision)
        self.metric_dict['recall'].append(recall)
        self.metric_dict['AP'].append(AP)
        self.metric_dict['f1'].append(f1)
    
    def print_summary(self, epoch, total_epoch, lr):
        print(f"[{epoch+1}/{total_epoch}] lr = {lr:.7f}, ", end='')
        for key in self.metric_dict.keys():
            print(f"{key} =  {self.metric_dict[key][epoch]}, ", end='')
        print('\n')

def split_train_valid(train_val_list, valid_head, valid_len):
    
    total_len = len(train_val_list)
    train_len = total_len - valid_len
    assert train_len > 0
    
    valid = []
    train = []
    
    for idx in range(valid_head, valid_head + valid_len):
        valid.append(train_val_list[idx % total_len])
    
    for idx in range(valid_head+valid_len, valid_head+valid_len + train_len):
        train.append(train_val_list[idx % total_len])
    
    return train, valid
        
def cross_valid(train_val_list:list, json_path:str, num_cls:int, id_map:dict, split_ratio: int = 0.25):
    fold = get_k_fold(train_val_list, split_ratio)
    statistics, contour_len_stat = parse_labels(json_path, num_cls, id_map)
    while not check_train_not_zero(statistics, fold, num_cls):
        fold = get_k_fold(train_val_list, seed=int(random.randint(1, 100000)))
    return fold

def get_k_fold(train_val_list: list, split_ratio: int =0.25, seed: int = 65536):
    random.seed(seed)
    random.shuffle(train_val_list)
    total_len = len(train_val_list)
    valid_len = int(split_ratio*total_len)
    
    fold = []
    valid_head = 0
    
    while valid_head < total_len:
        train, valid = split_train_valid(train_val_list, valid_head, valid_len)
        fold.append([train, valid])
        valid_head += valid_len
    
    return fold

def parse_labels(json_path, num_cls, id_map, extension='.ndpi'):
    statistics = {}
    contour_len_stat = []

    with open(json_path) as f:
        json_dict = json.load(f)
        keys = list(json_dict.keys())

        # collect label
        for key in keys:
            targets = json_dict[key]['targets']
            
            # discard slides with few labels
            if (len(targets) < 35):
                continue
                
            # register keys
            statistics[key+extension] = np.zeros(num_cls, dtype=np.int32)
            
            for target in targets:
                # check there is item in labels
                if len(target['labels']) == 0:
                    continue

                # check label is in interset
                label = (target['labels'][0]['label'])
                if not label in id_map.keys():
                    continue

                contour_len = len(target['segments'])
                # check contour length rational
                contour_len_stat.append(contour_len)
                if contour_len <10:
                    continue
                statistics[key+extension][id_map[label][0]] += 1
    
    return statistics, contour_len_stat

def check_train_not_zero(statistics:dict, fold: list, num_cls: int):

    for idx, (train, val) in enumerate(fold):
        train_state = count_stat(statistics, train, num_cls)
        for i in train_state:
            if i == 0:
                # one class not present in train data --> invalid splitting, try another seed!
                return False
    return True
    

def count_stat(statistics, keys, num_cls):

    ret = np.zeros(num_cls, dtype=np.int32)
    for key in keys:
        ret += statistics[key]
    
    return ret

def open_json(label_path):
    with open(label_path) as f:
        js = json.load(f)
    return js, list(js.keys())

def create_statistic_table(slides: list, js:dict, class_map:dict, extension:str='.ndpi') -> np.ndarray:
    statistic_table = np.zeros((len(slides), len(class_map)), dtype=np.int32)
    for idx, name in enumerate(slides):
        targets = js[name.split(extension)[0]]['targets']
        for target in targets:
            if len(target['labels'])  == 0:
                continue
            if len(target['segments']) < 10:
                continue
            label = (target['labels'][0]['label'])

            cls = -1
            for i in range(len(class_map)):
                if label == class_map[i][0]:
                    cls = i
                    break
            if cls >= 0:
                statistic_table[idx][cls] += 1
    print(statistic_table)
    return statistic_table

def get_contour_frequency(statistics:np.ndarray) -> np.ndarray:
    return np.sum(statistics, axis=0)

def balance_contour(statistics: np.ndarray, fold: int=20) -> np.ndarray:
    ret = np.ones(statistics.shape, dtype=np.float32)
    ret = np.reciprocal(statistics + 1e-7)
    ret *= (np.mean(statistics)*fold)
    ret = ret.astype(np.int32)
    return ret

def print_sampling_method(contour_freq: np.ndarray, sample_freq: np.ndarray, class_map: dict)-> None:
    for i in range(contour_freq.shape[0]):
        print(f"[{i}]: {contour_freq[i]:>4} | {sample_freq[i]:>4} | {contour_freq[i]*sample_freq[i]} | {class_map[i][3]}")


def dump_contours_labels(json_path:str, slide_name:str, interest_cls: list)-> list:
    with open(json_path) as f:
        js = json.load(f)
    ret = []
    targets = js[slide_name]['targets']
    for target in targets:
        if len(target['labels']) != 1:
            continue
        elif len(target['segments']) < 10:
            continue
        
        label = target['labels'][0]['label']
        if label not in interest_cls:
            continue
        segment = np.array(target['segments'], dtype=np.int32)
        ret.append([segment, label])
    del js
    return ret

def raytrace(A: np.ndarray, B: np.ndarray, mask_matrix: np.ndarray) -> None:
    arr1 = [int(i) for i in np.floor(A)]
    arr2 = [int(i) for i in np.floor(B)]
    mask_matrix[arr1[1], arr1[0]] = mask_matrix[arr2[1], arr2[0]] = True
    if arr1 == arr2:
        return 
    cells = list(bresenham(*arr1, *arr2))
    for x, y in cells:
        mask_matrix[y, x] = True
    

def contour_bound(contour: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xmin, xmax = np.min(contour[:, 0]), np.max(contour[:, 0])
    ymin, ymax = np.min(contour[:, 1]), np.max(contour[:, 1])
    return (xmin, ymin, xmax-xmin, ymax-ymin)


def normalize_contour(contour: np.ndarray, x:np.ndarray, y:np.ndarray, w:np.ndarray, h:np.ndarray)-> np.ndarray:
    contour_norm = np.copy(contour).astype(np.float32)
    contour_norm[:, 0] = (contour_norm[:, 0] - x)/w
    contour_norm[:, 1] = (contour_norm[:, 1] - y)/h
    return contour_norm

def watershed(mask:np.ndarray)->np.ndarray:
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    visited_map = np.pad(np.copy(mask), ((1, 1),(1, 1)), mode='constant', constant_values=False)
    background_mask = np.zeros(visited_map.shape).astype(np.bool)
    
    visited_map[0, 0] = background_mask[0, 0] =  1
    queue = [(0, 0)]
    index = 0
    
    while(index < len(queue)):
        x, y = queue[index]
        for dx, dy in dirs:
            if visited_map[y+dy, x+dx] == False:
                visited_map[y+dy, x+dx] = background_mask[y+dy, x+dx] = True
                queue.append((x+dx, y+dy))
        index += 1
    
    return np.logical_not(background_mask[1:-1, 1:-1]).astype(np.uint8)*2-mask
    

def find_inside_tiles(contour_norm: np.ndarray, tile_sz = 10, epsilon=1e-4, debug=False
    ) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((tile_sz, tile_sz), dtype=np.bool)
    mesh = contour_norm * (tile_sz-epsilon)
    
    for i in range(mesh.shape[0]):
        prev = mesh[i-1]
        nex = mesh[i]
        raytrace(prev, nex, mask)
    
    foreground = watershed(mask)
    if debug:
        print(mask)
        print('--------------------------')
        print(foreground)
        print('--------------------------')
        
    return np.where(foreground > 0), foreground
    
def generate_from_foreground(tx, ty):
    tile_no = random.randint(0, len(tx)-1)
    dx, dy = random.uniform(0, 1), random.uniform(0, 1)
    return tx[tile_no]+dx, ty[tile_no]+dy, tile_no

def denorm(nx: np.ndarray, ny: np.ndarray, x: np.ndarray, y: np.ndarray, w: np.ndarray, h: np.ndarray):
    return int((nx*w)+x), int((ny*h)+y)

def find_n_point_inside(contour: np.ndarray, n: int, tile_sz: int=4):
    bbox_shape = contour_bound(contour)
    contour_norm = normalize_contour(contour, *bbox_shape)
    (ty, tx), foreground = find_inside_tiles(contour_norm, tile_sz=tile_sz)
    
    ret = []
    path_ = mpltPath.Path(contour, closed=True)
    i = 0
    while i< n:
        nx, ny, tile_no = generate_from_foreground(tx, ty)
        nx /= tile_sz
        ny /= tile_sz
        p = denorm(ny, nx, *bbox_shape)
        if foreground[int(tx[tile_no]), int(ty[tile_no])] == 2:
            ret.append(p)
            i += 1
        elif path_.contains_point(p):
            ret.append(p)
            i += 1
    return ret

def find_n_point_inside_robust(contour: np.ndarray, n: int):
    x, y, w, h = contour_bound(contour)
    
    ret = []
    path_ = mpltPath.Path(contour, closed=True)
    i = 0
    while i < n:
        p = (random.randint(x, x+w), random.randint(y, y+h))
        if path_.contains_point(p):
            ret.append(p)
            i+=1
    return ret
    
def get_bbox_one_slide(slide_dir: str, sample_slide_name:str, sample_freq: dict, label_path:str, 
                       id2cls: dict, patch_size:int, extension: str='.ndpi', robust: bool=False):
    sample_label_contour_pairs = dump_contours_labels(json_path=label_path, 
                                                      slide_name=sample_slide_name.split(extension)[0],
                                                      interest_cls=id2cls.keys(),
                                                     )
    ret = []
    for sample_contour, sample_label in sample_label_contour_pairs:
        sample_cls = id2cls[sample_label]
        if robust:
            xy_pairs = find_n_point_inside_robust(sample_contour, n=sample_freq[sample_cls])
        else:
            xy_pairs = find_n_point_inside(sample_contour, n=sample_freq[sample_cls])
        ret.extend([[x-patch_size//2, y-patch_size//2, sample_cls] for x, y in xy_pairs])
    return ret


if __name__ == '__main__':
    m = Metric()
    m.load_metrics('../checkpoint/tmp.csv')

    for i in range(10):
        train_ = [np.random.rand() for i in range(3)]
        test_  = [np.random.rand() for i in range(3)]
        m.push_loss_acc_kappa(*train_, train=True)
        m.push_loss_acc_kappa(*test_,  train=False)
    
    m.save_metrics('../checkpoint/tmp.csv')


