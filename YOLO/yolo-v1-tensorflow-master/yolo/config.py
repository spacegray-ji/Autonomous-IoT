import os

DATA_PATH = 'data'  

PASCAL_PATH = os.path.join(DATA_PATH, 'pascal_voc')  

CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')  

OUTPUT_DIR = os.path.join(PASCAL_PATH, 'output')  
WEIGHTS_DIR = os.path.join(DATA_PATH, 'weights')  
WEIGHTS_META = 'YOLO.meta'
# WEIGHTS_FILE = None
WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt') 
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']  

FLIPPED = True  



IMAGE_SIZE = 448  

CELL_SIZE = 7  

BOXES_PER_CELL = 2  

ALPHA = 0.1  
DISP_CONSOLE = False  


OBJECT_SCALE = 1.0  
NOOBJECT_SCALE = 1.0  
CLASS_SCALE = 2.0  
COORD_SCALE = 5.0  

"""
solver parameter
训练参数设置
"""

GPU = ''

LEARNING_RATE = 0.0001  

DECAY_STEPS = 30000  

DECAY_RATE = 0.1 

STAIRCASE = True

BATCH_SIZE = 45  
MAX_ITER = 15000  

SUMMARY_ITER = 10  

SAVE_ITER = 1000  

"""
test parameter
测试相关参数
"""

THRESHOLD = 0.2  # cell 有目标的置信度阈值

IOU_THRESHOLD = 0.5  # 非极大抑制 IOU阈值
