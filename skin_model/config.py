from yacs.config import CfgNode as CN

_C = CN() # Node, lv0
_C.SYSTEM = CN() # None, lv1
_C.SYSTEM.DEVICES = [2]
_C.SYSTEM.USE_HOROVOD=False

_C.DATASET = CN()
_C.DATASET.SLIDE_DIR = "/mnt/cephrbd/data/A19001_NCKU_SKIN/Image/20191106/"
_C.DATASET.JSON_PATH = "/workspace/skin/label/label.json" #"/mnt/cephrbd/data/A19001_NCKU_SKIN/Meta/key-image-locations.json"
_C.DATASET.TRAIN_SLIDE = [
"2019-10-30 02.03.40.ndpi",
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
"2019-10-30 02.05.46.ndpi",]

_C.DATASET.EXTENSION=".ndpi"
_C.DATASET.BLUR=0.1

_C.DATASET.INPUT_SHAPE   = (512,512,3) #(256,256,3)
_C.DATASET.NUM_WORKER = 10
_C.DATASET.NUM_SLIDE = 4
_C.DATASET.AUGMENT = True
_C.DATASET.PREPROC = True

_C.DATASET.CLASS_MAP = [
    [225,0, 1, "background"               ], # "background"
    [56, 1, 2, "inflammatory infiltration"], # "inflammatory infiltration"
    [43, 2, 1, "adipose tissue"           ], # "adipose tissue"
    [53, 3, 1, "sweat gland"              ], # "sweat gland"
    [45, 4, 2, "hair follicles"           ], # "hair follicles" 
    [42, 5, 1, "dermis"                   ], # "dermis"
    [54, 6, 1, "sebaceous gland"          ], # "sebaceous gland"
    [41, 7, 2, "epidermis"                ], # "epidermis"
    [202,8, 1, "skeletal muscle"          ], # "skeletal muscle"
    [55, 9, 1, "blood vessels"            ], # "blood vessels"
    #[226,10, 1, "nerve fiber"],# "nerve fiber"
    #[80, 11, 1, "smooth muscle"],# "smooth muscle"
    #[44, 12, 1, "unspecified"],# "unspecified"
    #[220,13, 1, "ROI"],# "ROI"   
]
_C.DATASET.HIST_NAME = [
    "background", 
    "inflammatory infiltration",
    "adipose tissue",
    "sweat gland",
    "hair follicles" ,
    "dermis",
    "sebaceous gland",
    "epidermis",
    "skeletal muscle",
    "blood vessels",
    #"nerve fiber"
    #"smooth muscle"
    #"unspecified"
    #"ROI"   
]
# complete one: int_to_class = [225, 56, 43, 53, 45, 42, 54, 41, 55, 202, 226, 80, 44, 220]
#_C.DATASET.INT_TO_CLASS = [80, 56, 42, 53, 45, 42, 54, 41]
_C.DATASET.INT_TO_CLASS = [225, 56, 43, 53, 45, 42, 54, 41, 202, 55]

_C.MODEL = CN()
_C.MODEL.LIBRARY = "tensorflow" # "pytorch"
_C.MODEL.BACKBONE = "R-50-xt" #"R-101-v1"# "R-50-v1"
_C.MODEL.BATCH_SIZE = 8
_C.MODEL.EPOCHS = 50
_C.MODEL.LEARNING_RATE = 3e-5 # 1e-4 for SGD
_C.MODEL.USE_PRETRAIN = True
_C.MODEL.NORM_USE = "bn" # bn, gn
_C.MODEL.OPTIMIZER = "SGD" #"Adam" # SGD, Adam
_C.MODEL.LOAD_WEIGHT = False

# new function: multiscale learning
_C.MODEL.MULTISCALE = 0#4096

_C.MODEL.CHECKPOINT_DIR = "/workspace/skin/checkpoint/"
_C.MODEL.RESULT_DIR     = "/workspace/skin/checkpoint/"
_C.MODEL.DEBUG          = False
_C.MODEL.MODEL_DIR      = "/workspace/skin/inference_configs/"

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()