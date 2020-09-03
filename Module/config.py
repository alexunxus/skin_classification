from yacs.config import CfgNode as CN

_C = CN() # Node, lv0
_C.SYSTEM = CN() # None, lv1
_C.SYSTEM.DEVICES = [1]

_C.DATASET = CN()
_C.DATASET.SLIDE_DIR = "/mnt/cephrbd/data/A19001_NCKU_SKIN/Image/20191106/"
_C.DATASET.JSON_PATH = "/mnt/cephrbd/data/A19001_NCKU_SKIN/Meta/key-image-locations.json"
_C.DATASET.TRAIN_SLIDE = [
"2019-10-30 02.05.46.ndpi",
"2019-10-30 02.04.50.ndpi",
"2019-10-30 02.09.05.ndpi",
"2019-10-30 02.07.27.ndpi",
"2019-10-30 02.10.47.ndpi",
"2019-10-30 02.14.37.ndpi",
"2019-10-30 02.18.03.ndpi",
"2019-10-30 02.19.24.ndpi",
"2019-10-30 02.15.32.ndpi",
"2019-10-30 02.23.07.ndpi",
"2019-10-30 01.59.42.ndpi",
"350013D01170 - 2019-10-30 02.21.40.ndpi"]

_C.DATASET.VALID_SLIDE=[
"2019-10-30 02.01.19.ndpi",
"2019-10-30 02.02.21.ndpi",
"2019-10-30 02.03.40.ndpi",
"2019-10-30 02.05.46.ndpi"]

_C.DATASET.EXTENSION=".ndpi"
_C.DATASET.BLUR=0.1

_C.DATASET.INPUT_SHAPE   = (512,512,3) #(256,256,3)
_C.DATASET.NUM_SLIDE_HOLD = 5
_C.DATASET.NUM_WORKER = 10
_C.DATASET.AUGMENT = True
_C.DATASET.PREPROC = False
_C.DATASET.CLASS_MAP = [
    (80, 0), 
    (56, 1), 
    (43, 2), 
    (53, 3), 
    (45, 4), 
    (42, 5),
    (54, 6), 
    (41, 7)
    #(55, 8)
]
_C.DATASET.INT_TO_CLASS = [80, 56, 42, 53, 45, 42, 54, 41]

_C.MODEL = CN()
_C.MODEL.BACKBONE = "R-50-v1"
_C.MODEL.BATCH_SIZE = 32
_C.MODEL.EPOCHS = 50
_C.MODEL.LEARNING_RATE = 1e-4
_C.MODEL.USE_PRETRAIN = True
_C.MODEL.NORM_USE = "bn" # bn, gn
_C.MODEL.OPTIMIZER = "Adam" # SGD, Adam
_C.MODEL.ALPHA = [
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
#[1], # if include vessel classes
]
_C.MODEL.CHECKPOINT_DIR = "/workspace/skin/checkpoint/"
_C.MODEL.RESULT_DIR="/workspace/skin/result/"
_C.MODEL.DEBUG = False
_C.MODEL.MODEL_DIR = "/workspace/skin/inference_configs/"

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()