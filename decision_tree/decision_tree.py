import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import cv2
from hephaestus.data.ndpwrapper_v2 import Slide_ndpread
from tf_training.skin_model.config import get_cfg_defaults
from tf_training.skin_model.util import compute_cluster_distribution, DFS, compute_areas, \
  segment_region, df2thumbnail

COLOR_TABLE = {
    0:{"class": "Background",
       "colorcode": (255, 255, 255)
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
       "colorcode": (0, 0, 153)
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
    8:{"class": "Blood vessels", #??
       "colorcode": (255, 0, 0)
      },
    9:{"class": "Muscle", #??
       "colorcode": (181, 136, 24)
      },
}

class FeatureExtractor:
    def __init__(self, csv_path, slides, groundtruth_path):
        self.csv_path = csv_path
        self.slides = slides
        self.save_path = "decision_input.csv"
        self.groundtruth_path = groundtruth_path
        if os.path.isfile(self.save_path):
          print("========loading df==========")
          self.df = pd.read_csv(self.save_path)
          return
        dfs = [pd.read_csv(os.path.join(csv_path, slide.split(".ndpi")[0], "df.csv")) for slide in slides]
        self.bit_images = [df2thumbnail(df) for df in dfs]

        operations = [self.inf2epi, self.adipose2epi, self.follicle2epi, self.inf2tissue]

        self.result_csv = [[] for i in range(len(self.bit_images))]
        for i, bit_image in enumerate(self.bit_images):
            print(f"========{slides[i]}=========")
            for operation in operations:
              params = operation(bit_image)
              if isinstance(params, tuple):
                for param in params:
                  self.result_csv[i].append(param)
              else:
                self.result_csv[i].append(params)
        self.result_column = ['inf_mean_dist', 'inf_var_dist', 'fat_contact', 'fat_area',
                              'hair_contact', 'hair_area', 'max_inf_ratio']
        df = pd.DataFrame(np.array(self.result_csv), columns=self.result_column, index=self.slides)
        df.to_csv("decision_input.csv", index=True)
        self.df = df

    def inf2epi(self, test_img):
        return compute_cluster_distribution(test_img, cluster_id=1, target_id=7)

    def adipose2epi(self, test_img):
        adipose_solver = DFS(test_img, interest_id=2, inf_id=1)
        contact_count, adipose_area = adipose_solver.solve()
        print(f"Adipose contact={contact_count}, adipose area={adipose_area}")
        return contact_count, adipose_area
    
    def follicle2epi(self, test_img):
        follicle_solver = DFS(test_img, interest_id=4, inf_id=1)
        contact_count, follicle_area = follicle_solver.solve()
        print(f"Follicle contact={contact_count}, follicle area={follicle_area}")
        return contact_count, follicle_area
    
    def inf2tissue(self, test_img):
        tissue_labels, tissue_area = compute_areas(test_img>0)
        inf_regions = segment_region(test_img == 1)
        largest_ratio = 0
        for inf_region in inf_regions:
            x, y = inf_region[0][0], inf_region[1][0]
            tissue_id = tissue_labels[x, y]
            ratio = 1.*inf_region[0].shape[0]/tissue_area[tissue_id] if tissue_area[tissue_id] > 50 else 0
            largest_ratio = max(ratio, largest_ratio)
        print(f"Largest inflammatory ratio = {largest_ratio}")
        return largest_ratio
    
    def load_label(self):
      df = pd.read_csv(self.groundtruth_path)
      print(df.to_dict('split'))

if __name__ == "__main__":
  cfg = get_cfg_defaults()
  csv_path = "/workspace/skin/tf_training/result/inference_CSV/"
  groundtruth_path = "/workspace/skin/decision_tree/data.csv"
  slides = cfg.DATASET.VALID_SLIDE + cfg.DATASET.TRAIN_SLIDE +['2019-10-30 02.13.08.ndpi']
  featureExtractor = FeatureExtractor(csv_path, slides, groundtruth_path)

  featureExtractor.load_label()
