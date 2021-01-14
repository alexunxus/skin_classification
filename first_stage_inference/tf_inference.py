# inference.py
import glob
import os
import json

import pandas as pd
import numpy as np
import time
from tqdm import tqdm

# tensorflow
import tensorflow as tf

# import package from other folder
import sys
sys.path.insert(0,'../tf_training/')

# customized libraries
from skin_model.config import get_cfg_defaults
from skin_model.model import build_resnet, preproc_resnet
from skin_model.util  import get_class_map
from skin_model.eval import InferenceRunner, InfDataset
from tamsui_river.overlay_compressor import HeatmapCompressor

if __name__ == '__main__':

    cfg = get_cfg_defaults()
    
    # set GPU config
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = ','.join(str(i) for i in cfg.SYSTEM.DEVICES)
    os.environ['CUDA_VISIBLE_DEVICES'] = device

    # reserve GPU memory!
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # create model and build class_map
    columns = [item[3] for item in cfg.DATASET.CLASS_MAP]
    model = build_resnet(cfg.MODEL.BACKBONE, classNum=len(columns), in_shape=cfg.DATASET.INPUT_SHAPE)
    model.load_weights(cfg.INFERENCE.WEIGHT)

    # inference
    datalist = cfg.INFERENCE.INFERENCE_FILES
    for i_item, slide_file in enumerate(datalist):
        result_dir = cfg.INFERENCE.RESULT_DIR
        target_folder = os.path.join(result_dir,
                                     os.path.basename(slide_file).replace(cfg.DATASET.EXTENSION, ""))
        if os.path.exists(target_folder) and len(os.listdir(target_folder)):
            print("File {} has been worked before, skipping...".format(slide_file))
            continue
        else:
            os.makedirs(target_folder, exist_ok=True)
            print(f"Working on {i_item + 1}/{len(datalist)}, {slide_file}, Loader Ready, start inferencing")
            
            inference_dataset = InfDataset(slide_dir=cfg.INFERENCE.INFERENCE_DIR,
                                           slide_name=slide_file,
                                           patch_size=cfg.DATASET.INPUT_SHAPE[0:2],
                                           preproc_fn=preproc_resnet if cfg.DATASET.PREPROC else None,
                                           hsv_threshold=0.05)
            inference_runner = InferenceRunner(inference_dataset, model, center_weight=1, batch_size=32)
            pred_np = inference_runner.get_heatmap()
            raw_h = inference_dataset.get_shape()[1]
            raw_w = inference_dataset.get_shape()[0]
            heatmap_h, heatmap_w = inference_runner.heatmap.shape[:2]
        
        slide_name = os.path.basename(slide_file).replace(cfg.DATASET.EXTENSION, "")
        encoder = HeatmapCompressor()
        digits = ['{:.3f}'.format(i*0.1) for i in range(1, 10)]
        layer_dict = {digit: {'path':os.path.join(target_folder, f'alpha_thres-{digit}.png'), "score":0} for digit in digits}

        d = {
            "slide_name": slide_name,
            "default_threshold": "0.100",
            "tile_size": cfg.DATASET.INPUT_SHAPE[0],
            "meta_path": os.path.join(result_dir, slide_name, "df.csv"),
            "layer_map": layer_dict,
            "targets": [],
            "ai_project": 10,
            "store_meta_to_db": False,
            "heatmap": {
                "width": heatmap_w,
                "height": heatmap_h,
                "data": encoder.encode_maps(pred_np, columns)
            }
        }
        #print(d)
        with open(os.path.join(target_folder, "mapping.json"), "w") as f:
            json.dump(d, f)
        
    print('Finish Inference')
