import argparse
import cv2
import glob
import itertools
import os
import re
import shutil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from hephaestus.data.ndpwrapper_v2 import Slide_ndpread as SlideReader

parser = argparse.ArgumentParser(description="Re-color script")
parser.add_argument("--mask_dir", type=str, help="location of alpha-map folder, should point to '.../whole_slide'")
parser.add_argument("--slide_name", type=str, help="slide name in the mask_dir")
parser.add_argument("--data_dir", default="/mnt/cephrbd/data/A19008_TMU_NEXT/Image", type=str, help="location of raw ndpi files")
parser.add_argument("--alpha_level", default=0.6, type=float, help="alpha of overlay")
args = parser.parse_args()
## Define Thres-level and Color Table
thres_level = np.linspace(0, 1, 11)[0:-1]
color_table = {
    0:{"class": "Muscle",
       "colorcode": (181, 136, 24)
      },
    1:{"class": "Inflammatory infiltration",
       "colorcode": (255, 87, 34)
      },
    2:{"class": "Adipose tissue",
       "colorcode": (233, 30, 99)
      },
    3:{"class": "Sweat gland",
       "colorcode": (205, 220, 57)
      },
    4:{"class": "Hair follicles",
       "colorcode": (3, 169, 244)
      },
    5:{"class": "Dermis",
       "colorcode": (76, 175, 80)
      },
    6:{"class": "Sebaceous gland",
       "colorcode": (156, 39, 176)
      },
    7:{"class": "Epidermis",
       "colorcode": (3, 169, 244)
      },
    8:{"class": "Background",
       "colorcode": (32, 32, 32)
      },
}


## main script
slide_folder = os.path.join(args.mask_dir, args.slide_name)
if not os.path.exists("{}/binary_map".format(slide_folder)):
    os.makedirs("{}/binary_map".format(slide_folder))
    binary_masks = glob.glob("{}/alpha*".format(slide_folder))
    for ff in binary_masks:
        shutil.copyfile(ff, "{}/binary_map/{}".format(slide_folder, os.path.basename(ff)))

fname = os.path.join(slide_folder, "df.csv")
df = pd.read_csv(fname)
slide_w = df["w"].max() + 1
slide_h = df["h"].max() + 1

# For save overlap slide & predict mask

col_name = [i for i in df.columns if "pred" in i]
new_col_name = [color_table[idx]['class'] for idx, name in enumerate(col_name)]

for thres in thres_level:
    # if the most confidnent value is less than thres, make it as normal
    value = df[col_name].values
    
    value = value.reshape((slide_h,slide_w,-1)) if len(value.shape) > 1 else value.reshape((slide_h,slide_w)) #[H,W,cls]
    
    mask = np.ones((value.shape[:2] + (4,)))*255
    dominant_value = value.argmax(axis=-1)
    hh, ww= dominant_value.shape
    for h,w in itertools.product(np.arange(hh), np.arange(ww)):
        cls = dominant_value[h,w]

        v = value[h,w,cls]
        if v < thres:
            cls = 8
        colorcode = color_table[cls]["colorcode"]
        mask[h,w,:3] = colorcode
        mask[h,w,3] = 255*args.alpha_level

    mask_array = Image.fromarray(mask.astype('uint8'))
    
    fname = "{}/alpha_thres-{:.3f}.png".format(slide_folder, thres)
    mask_array.save(fname)
    
# replace 'pred_as_N' to class name
fname = os.path.join(slide_folder, "df.csv")
df.columns = ['w', 'h'] + new_col_name
df.to_csv(fname, index=False)
