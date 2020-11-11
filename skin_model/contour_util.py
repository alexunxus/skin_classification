import cv2
import os
import torch
import numpy as np
from mmcv.ops.nms import batched_nms

class MMDContourParser:
    def __init__(self, 
                 label_dict,
                 nms_cfg,
                 score_threshold=0.2,
            ):
        """
        label_dict: {
        0: "xxx",
        1: "yyy"
        }
        """
        self.label_dict = label_dict
        self.nms_cfg = nms_cfg
        self.score_threshold = score_threshold
    
    def convert_result(self, result):
        boxes, segm = result
        cls_ = np.concatenate([np.ones(len(b))*i for i, b in enumerate(boxes)])
        
        boxes_mix_cls = torch.cat(boxes)
        bbox_, scores_ = boxes_mix_cls[:, :4], boxes_mix_cls[:, -1]
        scores_ = scores_.contiguous() # the original memmap is not on the continous block, force them continous
        idxs = torch.zeros(len(bbox_))
        
        boxes_nms, keep_idx = batched_nms(
            boxes=bbox_, 
            scores=scores_, 
            idxs=idxs, 
            nms_cfg=self.nms_cfg
        )
        masks = np.concatenate(segm)
        masks_nms = np.take(masks, keep_idx, axis=0)
        cls_nms = cls_[keep_idx]
        scores_nms = scores_.cpu().detach().numpy()[keep_idx]
        
        # filter by scroes and mask_size
        keep_idx = np.where((scores_nms > self.score_threshold) & (masks_nms.sum(axis=(1,2)) >= 10))[0]
        
        box_out = boxes_nms[keep_idx]
        scores_out = scores_nms[keep_idx]
        mask_out = masks_nms[keep_idx]
        cls_out = cls_nms[keep_idx]
        
        output_poly = []
        output_clas = []
        for label_id, mask in zip(cls_out, mask_out):
            contour, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour = contour[0].squeeze()
            output_poly.append(contour)
            output_clas.append(self.label_dict[label_id])
        self.poly_contours = output_poly
        self.poly_class_name = output_clas
        self.poly_bbox = box_out
    
    def get_contours(self):
        return self.poly_contours, self.poly_class_name
    
    def get_formatted_contours(self):
        """Get 'target' slot for post-ready format
        """
        pass