from yacs.config import CfgNode as CN

_C = CN() # Node, lv0
_C.SYSTEM = CN() # None, lv1
_C.SYSTEM.DEVICES = [1]
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
    [225,0, 1, "Background"               ], # "background"
    [56, 1, 2, "Inflammatory_infiltration"], # "inflammatory infiltration"
    [43, 2, 1, "Adipose_tissue"           ], # "adipose tissue"
    [53, 3, 1, "Sweat_gland"              ], # "sweat gland"
    [45, 4, 2, "Hair_follicles"           ], # "hair follicles" 
    [42, 5, 1, "Dermis"                   ], # "dermis"
    [54, 6, 1, "Sebaceous_gland"          ], # "sebaceous gland"
    [41, 7, 2, "Epidermis"                ], # "epidermis"
    [202,8, 1, "Skeletal_muscle"          ], # "skeletal muscle"
    [55, 9, 1, "Blood_vessel"             ], # "blood vessels"
    #[226,10, 1, "Nerve"],# "nerve fiber"
    #[80, 11, 1, "Smooth_muscle"],# "smooth muscle"
    #[44, 12, 1, "Unspecified"],# "unspecified"
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
_C.MODEL.BACKBONE = "R-101-v1" #"R-50-xt" "R-101-v1"# "R-50-v1"
_C.MODEL.BATCH_SIZE = 8
_C.MODEL.EPOCHS = 50
_C.MODEL.LEARNING_RATE = 3e-5 # 1e-4 for SGD
_C.MODEL.USE_PRETRAIN = True
_C.MODEL.NORM_USE = "bn" # bn, gn
_C.MODEL.OPTIMIZER = "SGD" #"Adam" # SGD, Adam
_C.MODEL.LOAD_WEIGHT = False

# new function: multiscale learning
_C.MODEL.MULTISCALE = 0#4096

_C.MODEL.CHECKPOINT_DIR = "/workspace/skin/tf_training/checkpoint/"
_C.MODEL.RESULT_DIR     = "/workspace/skin/tf_training/checkpoint/"
_C.MODEL.DEBUG          = False

_C.INFERENCE = CN()
_C.INFERENCE.WEIGHT = "/workspace/skin/tf_training/R-101-v1_512_E50_cls10_AUG_PREPROC.h5"
_C.INFERENCE.RESULT_DIR =  '/workspace/skin/first_stage_inference/inference_result/tf/'
_C.INFERENCE.BATCH_SIZE =  32
_C.INFERENCE.FIVE_CROP =  True
_C.INFERENCE.INFERENCE_DIR = '/mnt/cephrbd/data/A19001_NCKU_SKIN/Image/20210107/'
_C.INFERENCE.INFERENCE_FILES= [
    '17-D01434.ndpi',
    '18-D03693.ndpi',
    '19-011040.ndpi',
    '19-D01869.ndpi',
    '19-D02531.ndpi',
    '20-000077.ndpi',
    '18-D01424.ndpi',
    '19-008222.ndpi',
    '19-D00935.ndpi',
    '19-D01869_DX1.ndpi',
    '19-D03012.ndpi',
    '20-017360.ndpi',
    '18-D02833.ndpi',
    '19-009291.ndpi',
    '19-D01068.ndpi',
    '19-D02212.ndpi',
    '19-D03071.ndpi',
    '20-020112.ndpi',
    '18-D03211.ndpi',
    '19-010228.ndpi',
    '19-D01315.ndpi',
    '19-D02345.ndpi',
    '19-D03132.ndpi',
    '20-021357.ndpi',
]


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()