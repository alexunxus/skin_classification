# inference.py
import argparse
import glob
import os
import json
import yaml

import pandas as pd
import numpy as np
import tensorflow as tf

from tqdm import tqdm

from Module.config import get_cfg_defaults
from hephaestus.model_tools.yaml2model import Yaml2Model
from hephaestus.compose.wsi_patch.api import PatchResultPostprocess
from Module.eval import SlidePredictor, InferenceRunner, InfDataset
import time

parser = argparse.ArgumentParser()
parser.add_argument('--config-file', default=None, type=str, help="config file if modify anything")
parser.add_argument("opts",
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()
cfg = get_cfg_defaults()

if args.config_file is not None:
    cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)

### Modified if needed ###
PRE_BUILD_MODEL = True # In some case, model cannot directly load from yaml.

if PRE_BUILD_MODEL:
    from Module.model import return_resnet
    from Module.util  import get_class_map
    with open(os.path.join(cfg.MODEL.MODEL_DIR, "config.yaml")) as f:
        train_cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = ','.join(str(i) for i in train_cfg["SYSTEM"]["DEVICES"])
    os.environ['CUDA_VISIBLE_DEVICES'] = device

    class_map = get_class_map(cfg.DATASET.CLASS_MAP)
    model = return_resnet(cfg.MODEL.BACKBONE, classNum=len(class_map), in_shape=cfg.DATASET.INPUT_SHAPE)
    model.load_weights(train_cfg["SOURCE"]["MODEL_DIR"])
else:
    model = None

# inference and store the result into csv file
datalist = train_cfg["SOURCE"]["INFERENCE_FILES"]
PRP_fn = PatchResultPostprocess
for i_item, slide_file in enumerate(datalist):
    result_dir = os.path.join(train_cfg["SOURCE"]["RESULT_DIR"],
                              "inference_CSV")
    target_folder = os.path.join(result_dir,
                                 os.path.basename(slide_file).replace(cfg.DATASET.EXTENSION, ""))
    if os.path.exists(target_folder):
        # pass if exist
        print("File {} has been worked before, skipping...".format(slide_file))
        continue
    else:
        os.makedirs(target_folder, exist_ok=True)

    # inference started     
    print("Working on {}/{}, {}, Loader Ready, start inferencing".format(i_item + 1, len(datalist), slide_file))
    
    inference_dataset = InfDataset(slide_dir=cfg.DATASET.SLIDE_DIR,
                                   slide_name=slide_file,
                                   patch_size=cfg.DATASET.INPUT_SHAPE[0:2],
                                   hsv_threshold=0.05)
    inference_runner = InferenceRunner(inference_dataset, model, center_weight=1, batch_size=32)
    pred_np = inference_runner.get_heatmap()
      
    # Patch Result Post-processing
    PRP = PRP_fn(patch_size=cfg.DATASET.INPUT_SHAPE[0],
                 raw_h=inference_dataset.get_shape()[1],
                 raw_w=inference_dataset.get_shape()[0],
                 blur=0,
                 # show_probability=True,
                 )#cfg.DATASET.BLUR)

    d = PRP.run(prediction=pred_np, result_dir=result_dir, target_folder=target_folder)
    with open(os.path.join(target_folder, "mapping.json"), "w") as f:
        json.dump(d, f)
   
print('Finish Inference')

