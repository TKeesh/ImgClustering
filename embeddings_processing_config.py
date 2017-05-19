from easydict import EasyDict as edict
import os

project_dir = os.path.join(os.getcwd())

__C = edict()
cfg = __C

__C.DATASET_FOLDER = project_dir + '/data/'

__C.TENSORBOARD_PATH = 'tensorboard/test'

__C.EMB_IMAGE_WIDTH = 50
__C.EMB_IMAGE_HEIGHT = 50

