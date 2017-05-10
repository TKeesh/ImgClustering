from easydict import EasyDict as edict

import os

project_dir = os.path.join(os.getcwd())

__C = edict()
cfg = __C

#################
## data config ##
#################
__C.TRAIN_FOLDER = project_dir + '/dataset/data/'
__C.TEST_FOLDER = project_dir + '/dataset/data/'

#################
## extract config ##
#################
__C.TENSORBOARD_PATH = 'tensorboard/test'

__C.BATCH_SIZE = 1

__C.EMB_IMAGE_WIDTH = 50
__C.EMB_IMAGE_HEIGHT = 50

__C.INPUT_TENSOR_NAME = 'DecodeJpeg/contents:0'
#__C.OUTPUT_TENSOR_NAME = 'pool_3/_reshape:0'
__C.OUTPUT_TENSOR_NAME = 'softmax/logits:0'
