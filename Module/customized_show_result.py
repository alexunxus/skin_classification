import mmcv
from mmcv.image import imread, imwrite
import numpy as np
import cv2

def show_result(img,
                result,
                score_thr=0.3,
                bbox_color=None,
                text_color=None,
                class_names=None,
                thickness=1,
                font_scale=0.5,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None):
    img = mmcv.imread(img)
    if class_names is None:
        class_names = ('epithelial', 'lymphocyte', 'neutrophil', 'macrophage')
   
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # draw segmentation masks
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        # class_names = ('epithelial', 'lymphocyte', 'neutrophil', 'macrophage')
        color_masks = [(0, 255, 255), (255, 255, 0), (204, 0, 102), (255, 102, 0)]
        for i in inds:
            i = int(i)
            color_mask = color_masks[labels[i]]
            mask = segms[i].astype(np.uint8)
            contours, hierarchy = cv2.findContours(mask.copy(), 
                                                   cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
                                                   offset=(0, 0))
            cv2.drawContours(img, contours, -1, color_mask, 1) 
            # img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False

    # draw bounding boxes
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = imread(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    img = np.ascontiguousarray(img)
    for bbox, label in zip(bboxes, labels):
        text_color = color_masks[label]
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        if bbox_color is not None:
            cv2.rectangle( img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = class_names[
            label] if class_names is not None else f'cls {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, color=text_color)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)
    if not (show or out_file):
        return img

