# inference.py
import glob
import os
import json

import pandas as pd
import numpy as np
import time
from tqdm import tqdm

# import package from other folder
import sys
sys.path.append('../torch_training/')

# customized libraries
from pytorch_model.dataloader import skin_augment_fn, imagenet_preproc
from pytorch_model.config     import get_cfg_defaults
from pytorch_model.model_zoo  import CustomModel
from pytorch_model.runner     import InferenceRunner, InfDataset

from tamsui_river.overlay_compressor import HeatmapCompressor


def prepare_model(cfg):
    model = CustomModel(backbone=cfg.MODEL.BACKBONE, 
                        num_cls=len(cfg.DATASET.INT_TO_CLASS), 
                        resume_from=cfg.INFERENCE.WEIGHT,
                        norm=cfg.MODEL.NORM_USE)
    return model



if __name__ == '__main__':

    cfg = get_cfg_defaults()
    
    # Set GPU settings
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = ','.join(str(i) for i in cfg.SYSTEM.DEVICES)
    os.environ['CUDA_VISIBLE_DEVICES'] = device

    model = prepare_model(cfg)
    model = model.cuda()
    model.eval()
    
    # column names for output json file:
    columns = [item[3] for item in cfg.DATASET.CLASS_MAP]

    # inference and store the result into csv file
    datalist = cfg.INFERENCE.INFERENCE_FILES
    for i_item, slide_file in enumerate(datalist):
        result_dir = cfg.INFERENCE.RESULT_DIR
        target_folder = os.path.join(result_dir,
                                     os.path.basename(slide_file).replace(cfg.DATASET.EXTENSION, ""))
        if os.path.exists(target_folder) and len(os.listdir(target_folder)):
            # pass if exist
            print("File {} has been worked before, skipping...".format(slide_file))
            continue
        else:
            os.makedirs(target_folder, exist_ok=True)
            # inference started     
            print(f"Working on {i_item + 1}/{len(datalist)}, {slide_file}, Loader Ready, start inferencing")
            
            inference_dataset = InfDataset(slide_dir=cfg.INFERENCE.INFERENCE_DIR,
                                           slide_name=slide_file,
                                           patch_size=(cfg.DATASET.PATCH_SIZE, cfg.DATASET.PATCH_SIZE),
                                           preproc_fn=imagenet_preproc,
                                           hsv_threshold=0.05)
            inference_runner = InferenceRunner(inference_dataset, 
                                               model, center_weight=1, 
                                               batch_size=cfg.INFERENCE.BATCH_SIZE)
            pred_np = inference_runner.get_heatmap()
            raw_h   = inference_dataset.get_shape()[1]
            raw_w   = inference_dataset.get_shape()[0]
            heatmap_h, heatmap_w = inference_runner.heatmap.shape[:2]
        
        slide_name = os.path.basename(slide_file).replace(cfg.DATASET.EXTENSION, "")
        encoder = HeatmapCompressor()
        digits = ['{:.3f}'.format(i*0.1) for i in range(1, 10)]
        layer_dict = {digit: {'path':os.path.join(target_folder, f'alpha_thres-{digit}.png'), "score":0} for digit in digits}

        d = {
            "slide_name": slide_name,
            "default_threshold": "0.100",
            "tile_size": cfg.DATASET.PATCH_SIZE,
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
        with open(os.path.join(target_folder, "mapping.json"), "w") as f:
            json.dump(d, f)
        
    print('Finish Inference')

