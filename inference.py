# inference.py
import glob
import os
import json
import yaml

import pandas as pd
import numpy as np
import time
import tensorflow as tf

from tqdm import tqdm

from skin_model.config import get_cfg_defaults
from skin_model.model import build_resnet
from skin_model.util  import get_class_map
from skin_model.eval import InferenceRunner, InfDataset, getwh
from hephaestus.model_tools.yaml2model import Yaml2Model
from tamsui_river.overlay_compressor import HeatmapCompressor

### Modified if needed ###
PRE_BUILD_MODEL = True # In some case, model cannot directly load from yaml.

if PRE_BUILD_MODEL:
    cfg = get_cfg_defaults()
    with open(os.path.join(cfg.MODEL.MODEL_DIR, "config.yaml")) as f:
        inference_cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = ','.join(str(i) for i in inference_cfg["SYSTEM"]["DEVICES"])
    os.environ['CUDA_VISIBLE_DEVICES'] = device

    class_map = get_class_map(cfg.DATASET.CLASS_MAP)
    model = build_resnet(cfg.MODEL.BACKBONE, classNum=len(class_map), in_shape=cfg.DATASET.INPUT_SHAPE)
    model.load_weights(inference_cfg["SOURCE"]["MODEL_DIR"])
else:
    raise ValueError("Model doesn't exit!")

# inference and store the result into csv file
datalist = inference_cfg["SOURCE"]["INFERENCE_FILES"]
for i_item, slide_file in enumerate(datalist):
    result_dir = os.path.join(inference_cfg["SOURCE"]["RESULT_DIR"],
                              "10_class_inference_CSV_91_crop")
    target_folder = os.path.join(result_dir,
                                 os.path.basename(slide_file).replace(cfg.DATASET.EXTENSION, ""))
    if os.path.exists(target_folder):
        # pass if exist
        print("File {} has been worked before, skipping...".format(slide_file))
        #continue
        df = pd.read_csv(os.path.join(target_folder, "df.csv"))
        columns = df.columns[2:]
        raw_w, raw_h = getwh(os.path.join(cfg.DATASET.SLIDE_DIR, slide_file))
        heatmap_h, heatmap_w = max(df['h'])+1, max(df['w'])+1
        pred_np = df.to_numpy()[:, 2:].reshape((heatmap_h, heatmap_w, -1))
    else:
        os.makedirs(target_folder, exist_ok=True)
        # inference started     
        print(f"Working on {i_item + 1}/{len(datalist)}, {slide_file}, Loader Ready, start inferencing")
        
        inference_dataset = InfDataset(slide_dir=cfg.DATASET.SLIDE_DIR,
                                    slide_name=slide_file,
                                    patch_size=cfg.DATASET.INPUT_SHAPE[0:2],
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
    #break
    with open(os.path.join(target_folder, "mapping.json"), "w") as f:
        json.dump(d, f)
    
print('Finish Inference')

