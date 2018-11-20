from yacs.config import CfgNode as CN 
import os.path as osp

_C = CN()

_C.MODEL = CN()
_C.MODEL.PRETRAINED_MODEL_PATH = ""
_C.MODEL.ARCHITECTURE = "NORM"  # GOOD, NORM
_C.MODEL.DIM_G = 128  # Generator dimensionality
_C.MODEL.DIM_D = 128  # Critic dimensionality
_C.MODEL.DIM = 64  # DIM for good Generator and Discriminator
_C.MODEL.HASH_DIM = 64

_C.DATA = CN()
_C.DATA.USE_DATASET = "cifar10"  # "cifar10", "nuswide81", "coco"
_C.DATA.LABEL_DIM = 10
_C.DATA.DB_SIZE = 54000
_C.DATA.TEST_SIZE = 1000
_C.DATA.WIDTH_HEIGHT = 32
_C.DATA.OUTPUT_DIM = 32 * 32 * 3  # Number of pixels (32*32*3)
_C.DATA.MAP_R = 54000

_C.DATA.OUTPUT_DIR = "./output/cifar10_step_1"
_C.DATA.IMAGE_DIR = osp.join(_C.DATA.OUTPUT_DIR, "images")
_C.DATA.MODEL_DIR = osp.join(_C.DATA.OUTPUT_DIR, "models")
_C.DATA.LOG_DIR = osp.join(_C.DATA.OUTPUT_DIR, "logs")

_C.TRAIN = CN()
_C.TRAIN.USE_PRETRAIN = False
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.ITERS = 10000
_C.TRAIN.CROSS_ENTROPY_ALPHA = 5
_C.TRAIN.LR = 1e-4  # Initial learning rate
_C.TRAIN.G_LR = 1e-4  # 1e-4
_C.TRAIN.DECAY = True  # Whether to decay LR over learning
_C.TRAIN.N_CRITIC = 5  # Critic steps per generator steps
_C.TRAIN.SAVE_FREQUENCY = 20000  # How frequently to save model
_C.TRAIN.ACGAN_SCALE = 1.0
_C.TRAIN.ACGAN_SCALE_G = 0.1
_C.TRAIN.WGAN_SCALE = 1.0  
_C.TRAIN.WGAN_SCALE_G = 1.0
_C.TRAIN.NORMED_CROSS_ENTROPY = True
_C.TRAIN.FAKE_RATIO = 1.0

config = _C
