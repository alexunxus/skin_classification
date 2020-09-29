import numpy as np
from random import randint
import random
import matplotlib.path as mpltPath
import os
import tensorflow as tf

def exist_file(path, prefix, suffix):
    files = os.listdir(path)
    for f in files:
        if prefix in f and suffix in f:
            return True
    return False

def collect_histogram(_targets, label_dict, interest=None):
    '''
    Arg:
        _targets: the i-th slide in the folder, it has several contour within it
        label_dict: the dict of these contour, recording the histogram for each class
                    will be modified if there are new types of label
        
        NOTE:
            some of the target in _targets list is mis-annotated, they have no labels!
            they should not be used in training!
            so do some target with segment < 5 points
        
        the class "blood vessel" is only labeled in the last slide 
        "350013D01170 - 2019-10-30 02.21.40", only 8 images are collected
    Return:
        the number of class type
    '''

    class_type=len(label_dict)
    for target in _targets:
        if len(target['labels']) == 0 or len(target['labels']) > 1:
            continue
        if len(target['segments']) < 10:
            #print('Some segments are probably mis-annotated', target['segments'])
            continue
        label = target['labels'][0]['label']    
        if interest is not None and label not in interest:
            # do not include class "blood vessel" due to its small size
            continue

        if not label in label_dict:
            label_dict[label]=0
            class_type+=1
        else:
            label_dict[label]+=1
    return class_type

def get_frequency_dict(label_dict, upsample=4):
    mean = sum(label_dict.values())/len(label_dict)
    num_each_class = mean*10*upsample
    key_list = list(label_dict.keys())
    val_list = np.array(list(label_dict.values())).astype(np.float32)
    val_list = num_each_class/val_list
    val_list = np.ceil(val_list).astype(np.int32)
    #val_list *= upsample
    return dict(zip(key_list, val_list))


def get_class_map(config_class_list):
    class_map = {}
    for tup in config_class_list:
        class_map[tup[0]] = tup[1]
    return class_map

def get_bounding_box(center_point, boundary, size):
    '''
    Arg:
        boundary format: (w ,h)
        center_point: is list of Point or solitary one Point
    
    '''
    w , h  = boundary
    sx, sy = size[0], size[1]
    if isinstance(center_point, list):
        returnlist = True
    else:
        center_point = [center_point]
        returnlist = False
    ret = []
    for point in center_point:
        x, y = int(point[0]-sx/2), int(point[1]-sy/2)
        x = 0 if x<0 else x
        x = x-sx if x+sx>w else x
        y = 0 if y<0 else y
        y = y-sy if y+sy>h else y
        ret.append((x, y))
    if returnlist:
        return ret
    else:
        return ret[0]

def get_point_in_polygon(contour, num, ww, hh, x=0, y=0):
    '''
    Arg:
        contour: np array with size (n, 2)
        the polygon lies within (x,x+ww), (y, y+hh)
        num: the number of points one need to get in the polygon
    Special feature: if fail count > 10, then try to get middle point from the contour
    '''
    ret = []
    i = 0
    fail_count = 0
    path_ = mpltPath.Path(contour, closed=True)
    while i< num:
        if fail_count < 10 or random.randint(0, 10) < 5:
            p = (randint(x, x+ww-1), randint(y, y+hh-1))
        else:
            point_a = contour[randint(0, contour.shape[0]-1)]
            point_b = contour[randint(0, contour.shape[0]-1)]
            weight  = randint(0, 100)/100.
            p = point_a*weight + point_b*(1-weight)
            p = (int(p[0]), int(p[1]))
        if path_.contains_point(p):
            ret.append(p)
            i += 1
        else:
            fail_count += 1
    return ret

def get_point_in_polygon_robust(contour, num, ww, hh, x=0, y=0):
    '''
    Arg:
        contour: np array with size (n, 2)
        the polygon lies within (x,x+ww), (y, y+hh)
        num: the number of points one need to get in the polygon
    '''
    ret = []
    i = 0
    path_ = mpltPath.Path(contour, closed=True)
    while i< num:
        p = (randint(x, x+ww-1), randint(y, y+hh-1))
        if path_.contains_point(p):
            ret.append(p)
            i += 1
    return ret
