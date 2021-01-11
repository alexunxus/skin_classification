from yacs.config import CfgNode as CN

_C = CN() # Node, lv0
_C.SYSTEM = CN() # None, lv1
_C.SYSTEM.DEVICES = [0] # 0, 2, 3, 4

_C.SOURCE = CN()
_C.SOURCE.DEBUG= False

_C.DATASET = CN()
_C.DATASET.USE_CROSS_VALID = False
_C.DATASET.SLIDE_DIR = "/mnt/cephrbd/data/A19001_NCKU_SKIN/Image/20191106/"
_C.DATASET.LABEL_PATH = "/workspace/skin/label/label.json" #"/mnt/cephrbd/data/A19001_NCKU_SKIN/Meta/key-image-locations.json"
_C.DATASET.BBOX_PATH  = "/workspace/skin/bbox/"
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
    "350013D01170 - 2019-10-30 02.21.40.ndpi" ]

_C.DATASET.VALID_SLIDE=[
    "2019-10-30 02.01.19.ndpi",
    "2019-10-30 02.02.21.ndpi",
    "2019-10-30 02.05.46.ndpi", ]

_C.DATASET.TEST_SLIDE = [
    "2019-10-30 02.01.19.ndpi",
    "2019-10-30 02.02.21.ndpi",
    "2019-10-30 02.05.46.ndpi", ]

_C.DATASET.EXTENSION=".ndpi"

_C.DATASET.PATCH_SIZE   = 512
_C.DATASET.NUM_SLIDE = 9
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
]

# complete one: int_to_class = [225, 56, 43, 53, 45, 42, 54, 41, 55, 202, 226, 80, 44, 220]
_C.DATASET.INT_TO_CLASS = [225, 56, 43, 53, 45, 42, 54, 41, 202, 55]

_C.MODEL = CN()
_C.MODEL.BACKBONE = "r101" #'r50', 'r101', 'e-b0', 'e-b1', 'se-r101', 'se-r50'
_C.MODEL.BATCH_SIZE = 16
_C.MODEL.EPOCHS = 50
_C.MODEL.LEARNING_RATE = 1e-4 # for pytorch, lr should be greater.
_C.MODEL.USE_PRETRAIN = True
_C.MODEL.NORM_USE          = "bn" # bn, gn
_C.MODEL.OPTIMIZER         = "SGD" #"Adam", "SGD"
_C.MODEL.SCHEDULER         = 'step' # 'cos_grad'
_C.MODEL.LOAD_CSV          = False
_C.MODEL.PATIENCE          = 5


_C.MODEL.CHECKPOINT_DIR = "/workspace/skin/torch_training/checkpoint/"
_C.MODEL.RESULT_DIR     = "/workspace/skin/torch_training/checkpoint/"
_C.MODEL.RESUME_FROM    = ''
_C.MODEL.DEBUG          = False

_C.METRIC = CN()
_C.METRIC.MODEL_SELECTION_CRITERION = 'auc'
_C.METRIC.KEYS = ['test_acc','train_acc', 'test_losses','train_losses', 
                  'auc', 'precision','recall', 'AP', 'f1']

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()