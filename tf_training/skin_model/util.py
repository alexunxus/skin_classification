import numpy as np
from random import randint
import random
import matplotlib.path as mpltPath
import os
import tensorflow as tf

def exist_file(path, prefix, suffix):
    # return if the file in path have both prefix and suffix string
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
    # will adjust the number of each class to their mean * 10 * upsample ratio, defaut 40*mean
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
    Return: the LEFT UPPER BORDER of the bboxs
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
        x = w-sx if x+sx>w else x
        y = 0 if y<0 else y
        y = h-sy if y+sy>h else y
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

###############################################################
#          Block for decision tree util functions             #
###############################################################
from scipy.ndimage import label


def df2thumbnail(df):
    w = np.max(df["w"]) + 1
    h = np.max(df["h"]) + 1
    return df.to_numpy().reshape((h, w, -1))[..., 2:].argmax(axis=-1)
    
def segment_region(mask):
    labels, numL = label(mask)
    label_indices = [(labels == i).nonzero() for i in range(1, numL+1)]
    return label_indices

def compute_areas(mask):
    labels, numL = label(mask)
    areas = [np.sum(labels == i) for i in range(0, numL+1)]
    return labels, areas

def get_tissue_map(mask):
    labels, numL = label(mask)
    return labels, numL


def find_nearest_dist_and_point(regions, centers):
    '''
    Arg: regions: [[xs, ys],...] regions of epidermis
         centers: [[x_c, y_c]...] centers of inflammation
    return: [(D1, p1), (D2, p2)...]
    '''
    xs = np.array([x for region in regions for x in region[0]])
    ys = np.array([y for region in regions for y in region[1]])
    xc = centers[:, 0]
    yc = centers[:, 1]
    
    dist_matrix =  np.square(np.tile(np.expand_dims(xs, 0), (len(xc), 1))-np.tile(np.expand_dims(xc, -1), (1, len(xs))))
    dist_matrix += np.square(np.tile(np.expand_dims(ys, 0), (len(yc), 1))-np.tile(np.expand_dims(yc, -1), (1, len(ys))))
    
    min_dist = np.min(dist_matrix, axis= -1)
    min_dist = np.sqrt(min_dist)
    min_arg  = np.argmin(dist_matrix, axis= -1)
    return min_dist, min_arg

def find_mean_and_var_dist(min_dist, center_size):
    mean_dist = np.dot(min_dist, center_size)*1./np.sum(center_size)
    var_dist  = np.dot(np.square(min_dist-mean_dist), center_size)/np.sum(center_size)
    return mean_dist,var_dist

def compute_cluster_distribution(test_img, cluster_id, target_id):
    epi_regions = segment_region(test_img == target_id)
    inf_regions = segment_region(test_img == cluster_id)
    tissue_map, numL = get_tissue_map(test_img > 0)

    inf_centers = [[] for i in range(0, numL+1)]
    epi_regions_by_tissue = [[] for i in range(0, numL+1)]
    inf_size    = [[] for i in range(0, numL+1)]

    for inf_region in inf_regions:
        tissue_id = tissue_map[inf_region[0][0], inf_region[1][0]]
        assert tissue_id <= numL
        inf_centers[tissue_id].append([inf_region[0].mean(), inf_region[1].mean()])
        inf_size[tissue_id].append(len(inf_region[0]))

    for epi_region in epi_regions:
        tissue_id = tissue_map[epi_region[0][0], epi_region[1][0]]
        assert tissue_id <= numL
        if len(epi_region[0]) >= 4:
            epi_regions_by_tissue[tissue_id].append(epi_region)

    min_dist = []
    min_arg  = []
    valid_center = ([], [])
    valid_size   = []
    for i in range(1, numL+1):
        if len(epi_regions_by_tissue[i]) == 0 or len(inf_centers[i]) == 0:
            continue
        valid_center[0].extend([x for x, y in inf_centers[i]])
        valid_center[1].extend([y for x, y in inf_centers[i]])
        valid_size.extend([sz for sz in inf_size[i]])
        dist_array, arg_array = find_nearest_dist_and_point(epi_regions_by_tissue[i], np.array(inf_centers[i]))
        min_dist.extend(dist_array)
        min_arg.append((i, arg_array))
    mean_dist = np.dot(min_dist, valid_size)*1./np.sum(valid_size)
    var_dist  = np.dot(np.square(min_dist-mean_dist), valid_size)/np.sum(valid_size)
    return mean_dist, var_dist

class DFS:
    def __init__(self, image, interest_id, inf_id):
        self.image = image
        self.interest_id = interest_id
        self.inf_id = inf_id
        self.visit_mask = np.zeros(image.shape[:2])
        self.contact_mask = np.zeros(image.shape[:2])
        self.interest_mask = image == interest_id
        self.inf_mask = image == inf_id
        self.contact_count = 0
        self.dirs = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        
    def solve(self):
        xs, ys = np.nonzero(self.interest_mask)
        for i in range(len(xs)):
            self.idfs(xs[i], ys[i])
        return self.contact_count, np.sum(self.interest_mask)
    
    def idfs(self, x, y):
        if self.visit_mask[x, y] == 1:
            return
        self.visit_mask[x, y] = 1
        
        stack = [(x, y)]
        
        while(stack):
            x, y = stack.pop()
            for dx, dy in self.dirs:
                if x+dx < 0 or x+dx >= self.image.shape[0] or y+dy < 0 or y+dy >= self.image.shape[1]:
                    continue
                if self.image[x+dx, y+dy] == self.inf_id:
                    self.contact_mask[x+dx, y+dy] = 1
                    self.contact_count += 1
                if self.image[x+dx, y+dy] == self.interest_id and self.visit_mask[x+dx, y+dy] == 0:
                    self.visit_mask[x+dx, y+dy] = 1
                    stack.append((x+dx, y+dy))