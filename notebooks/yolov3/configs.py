#================================================================
#   Adapted from: https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : yolov3 configuration file
#
#================================================================

# YOLO options
YOLO_DARKNET_WEIGHTS        = "C:/Users/jeane/Documents/new-hands-on-2021/models/yolov3.weights"
YOLO_V3_WEIGHTS             = "C:/Users/jeane/Documents/new-hands-on-2021/models/yolov3.weights"
YOLO_COCO_CLASSES           = "./data/TrainIJCNN2013/coco.names"
YOLO_FRAMEWORK              = "tf"
YOLO_CUSTOM_WEIGHTS         =  "checkpoints/yolov3_custom" #False # used in evaluate_mAP.py and custom model detection, if not using leave False
YOLO_STRIDES                = [8, 16, 32]
YOLO_IOU_LOSS_THRESH        = 0.5
YOLO_ANCHOR_PER_SCALE       = 3
YOLO_MAX_BBOX_PER_SCALE     = 100
YOLO_INPUT_SIZE             = 416
YOLO_ANCHORS                = [[[10,  13], [16,   30], [33,   23]],
                               [[30,  61], [62,   45], [59,  119]],
                               [[116, 90], [156, 198], [373, 326]]]

YOLO_TYPE = "yolov3"

# Train options
TRAIN_YOLO_TINY             = False
TRAIN_LOAD_IMAGES_TO_RAM    = True
TRAIN_SAVE_BEST_ONLY        = True # saves only best model according validation loss (True recommended)
TRAIN_SAVE_CHECKPOINT       = False # saves all best validated checkpoints in training process (may require a lot disk space) (False recommended)
TRAIN_CLASSES               = "./data/TrainIJCNN2013/Dataset_names.txt"
TRAIN_ANNOT_PATH            = "./data/TrainIJCNN2013/Dataset_train.txt"
TRAIN_LOGDIR                = "./log"
TRAIN_BATCH_SIZE            = 4
TRAIN_INPUT_SIZE            = 416
TRAIN_DATA_AUG              = True
TRAIN_CHECKPOINTS_FOLDER    = "checkpoints"
TRAIN_MODEL_NAME            = f"{YOLO_TYPE}_custom"
TRAIN_TRANSFER              = True
TRAIN_FROM_CHECKPOINT       = False #"checkpoints/yolov3_custom"
TRAIN_LR_INIT               = 1e-4
TRAIN_LR_END                = 1e-6
TRAIN_WARMUP_EPOCHS         = 2
#TRAIN_EPOCHS                = 30
TRAIN_EPOCHS                = 150

# TEST options
TEST_ANNOT_PATH             = "./data/TrainIJCNN2013/Dataset_val.txt"
TEST_BATCH_SIZE             = 4
TEST_INPUT_SIZE             = 416
TEST_DATA_AUG               = False
TEST_DECTECTED_IMAGE_PATH   = ""
TEST_SCORE_THRESHOLD        = 0.3
TEST_IOU_THRESHOLD          = 0.5
