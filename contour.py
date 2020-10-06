import cv2
import os
import itertools
import json
import torch
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm
from mmcv import Config
from mmcv.ops.nms import batched_nms

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from Module.customized_show_result import show_result

from Module.contour_util import MMD_ContourParser

TRAIN_INPUT_SIZE = 1024
INFERENCE_STRIDES = 1024


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    threshold = 0.5

    result_dir = "/workspace/skin/roi_result/"
    infer_model_ckpt = "/workspace/mmdetection/results/Monu-pn/latest.pth"
    cfg_path = "/workspace/mmdetection/results/Monu-pn/cfg_model-Monu.py"
    img_path = "/workspace/skin/roi/"
    label_path = "/workspace/skin/label/label.json"
    cfg = Config.fromfile(cfg_path)
    cfg["test_cfg"]["rcnn"]["nms"]["iou_threshold"] = threshold
    #classes = ("N", "P")
    classes= ('epithelial', 'lymphocyte', 'neutrophil', 'macrophage')

    # read label json file --> want to know the bbox original (x, y)
    with open(label_path) as f:
        js = json.load(f)

    model = init_detector(config=cfg, checkpoint=infer_model_ckpt, device="cuda:0")
    model.CLASSES = classes

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    
    for image in os.listdir("/workspace/skin/roi/"):
        # image shape
        img_h, img_w, *_ = cv2.imread(img_path+image).shape
        print(f"Image h={img_h}, w={img_w}")

        # result = inference_detector(model, img_path)
        result = inference_detector(model, "/workspace/skin/roi/"+image)
        
        # result: ( [1st class bbox, 2nd class bbox...], [1st class segm, 2nd class segm...])
        # bboxes, segms = result
        bboxes, segms = zip(result)
        bboxes = list(zip(*bboxes))
        segms = list(zip(*segms))
        
        segm_masks = []
        bboxes_list = []
        for ic in range(len(classes)):
            b = torch.stack([torch.from_numpy(j) for i in bboxes[ic] for j in i])
            bboxes_list.append(b)
            m = np.concatenate(segms[ic])
            #m = torch.cat(segms[ic])
            segm_masks.append(m)
        result = (bboxes_list, segm_masks)
            
        contour_parser = MMD_ContourParser(
            label_dict={0:'epithelial', 1:'lymphocyte', 2:'neutrophil', 3:'macrophage'},
            nms_cfg=cfg["test_cfg"]["rcnn"]["nms"],
            score_threshold=threshold)
        contour_parser.convert_result(result)
        poly_contours, poly_class = contour_parser.get_contours()

        slide_name = ".".join(os.path.basename(image).split(".")[:-1])
        print(slide_name)
        formatted_result = {
            "slide_name": slide_name,
            "project_name": "SKIN_MONU",
            "targets": []
        }

        for p, c in zip(poly_contours, poly_class):
            if len(p.shape) == 1:
                continue
            x_min, y_min = p.min(axis=0)
            x_max, y_max = p.max(axis=0)
            item = {
                "label_name": c,
                "segments": p.tolist(),
                "viewport_x": int(x_min),
                "viewport_y": int(y_min),
                "viewport_width": int(x_max-x_min),
                "viewport_height": int(y_max-y_min),
                "scores": 0.999
            }
            formatted_result["targets"].append(item)

        os.makedirs(result_dir, exist_ok=True)
        save_filename = "{}/{}.json".format(result_dir, slide_name)
        with open(save_filename, "w") as f:
            json.dump(formatted_result, f)

    """
    raw_image = cv2.imread(img_path)

    rh, rw, _ = raw_image.shape
    coord_list = list(itertools.product(np.arange(0, rw, INFERENCE_STRIDES), np.arange(0, rh, INFERENCE_STRIDES)))

    result_list = []
    for cw, ch in tqdm(coord_list):
        # NOT SUPPORT BATCH FOR NOW
        img = raw_image[ch:(ch+TRAIN_INPUT_SIZE), cw:(cw+TRAIN_INPUT_SIZE), ...]
        img_h, img_w, *_ = img.shape
        img = np.pad(img, 
            [(0, TRAIN_INPUT_SIZE-img_h), (0, TRAIN_INPUT_SIZE-img_w), (0, 0)], 
            mode="constant", constant_values=255)
        result = inference_detector(model, img)
        bboxes, segms = result
        # process box coord
        xy_ = np.array([cw, ch, cw, ch])
        for cls_item in bboxes:
            cls_item[:, :4] += xy_
        # process masks
        segms_lst = []
        for cls_item in segms:
            cmlst_ = []
            for m in cls_item:
                right_pad = rw-cw-TRAIN_INPUT_SIZE
                right_pad = right_pad if right_pad > 0 else 0
                bottom_pad = rh-ch-TRAIN_INPUT_SIZE
                bottom_pad = bottom_pad if bottom_pad > 0 else 0
                m = np.pad(m, [(ch, bottom_pad), (cw, right_pad)])
                m = m[:rh, :rw] # remove out out image
                cmlst_.append(m)
            segms_lst.append(cmlst_)
        result_list.append((bboxes, segms_lst))
    
    bboxes, segms = zip(*result_list)
    bboxes = list(zip(*bboxes))
    segms = list(zip(*segms))

    segm_masks = []
    bboxes_list = []
    for ic in range(len(classes)):
        b = torch.stack([j for i in bboxes[ic] for j in i])
        bboxes_list.append(b)
        m = np.concatenate(segms[ic])
        segm_masks.append(m)
    result = (bboxes_list, segm_masks)

    contour_parser = MMD_ContourParser(
        label_dict={0:'epithelial', 1:'lymphocyte', 2:'neutrophil', 3:'macrophage'},
        nms_cfg=cfg["test_cfg"]["rcnn"]["nms"],
        score_threshold=threshold)
    contour_parser.convert_result(result)
    poly_contours, poly_class = contour_parser.get_contours()

    slide_name = ".".join(os.path.basename(img_path).split(".")[:-1])
    formatted_result = {
        "slide_name": slide_name,
        "project_name": "SKIN",
        "targets": []
    }

    for p, c in zip(poly_contours, poly_class):
        if len(p.shape) == 1:
            continue
        x_min, y_min = p.min(axis=0)
        x_max, y_max = p.max(axis=0)
        item = {
            "label_name": c,
            "segments": p.tolist(),
            "viewport_x": int(x_min),
            "viewport_y": int(y_min),
            "viewport_width": int(x_max-x_min),
            "viewport_height": int(y_max-y_min),
            "scores": 0.999
        }
        formatted_result["targets"].append(item)

    os.makedirs(result_dir, exist_ok=True)
    save_filename = "{}/{}.json".format(result_dir, slide_name)
    with open(save_filename, "w") as f:
        json.dump(formatted_result, f)
    
    """