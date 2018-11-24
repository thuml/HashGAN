from yacs.config import CfgNode
import os

config = CfgNode()

config.MODEL = CfgNode()
config.MODEL.DIM_G = 128  # generator dimensionality
config.MODEL.DIM_D = 128  # Critic dimensionality
config.MODEL.DIM = 64  # DIM for good generator and discriminator
config.MODEL.HASH_DIM = 64
config.MODEL.G_ARCHITECTURE = "NORM"  # GOOD, NORM
config.MODEL.D_ARCHITECTURE = "NORM"  # GOOD, NORM, ALEXNET
config.MODEL.G_PRETRAINED_MODEL_PATH = ""
config.MODEL.D_PRETRAINED_MODEL_PATH = ""
# TODO: merge ALEXNET_PRETRAINED_MODEL_PATH  and D_PRETRAINED_MODEL_PATH
config.MODEL.ALEXNET_PRETRAINED_MODEL_PATH = "./pretrained_models/reference_pretrain.npy"

config.DATA = CfgNode()
config.DATA.USE_DATASET = "cifar10"  # "cifar10", "nuswide81", "coco"
config.DATA.LIST_ROOT = "./data/cifar10"
config.DATA.DATA_ROOT = "./data_list/cifar10"
config.DATA.LABEL_DIM = 10
config.DATA.DB_SIZE = 54000
config.DATA.TEST_SIZE = 1000
config.DATA.WIDTH_HEIGHT = 32
config.DATA.OUTPUT_DIM = 3 * (config.DATA.WIDTH_HEIGHT ** 2)  # Number of pixels (32*32*3)
config.DATA.MAP_R = 54000
config.DATA.OUTPUT_DIR = "./output/cifar10_step_1"
config.DATA.IMAGE_DIR = os.path.join(config.DATA.OUTPUT_DIR, "images")
config.DATA.MODEL_DIR = os.path.join(config.DATA.OUTPUT_DIR, "models")
config.DATA.LOG_DIR = os.path.join(config.DATA.OUTPUT_DIR, "logs")

config.TRAIN = CfgNode()
config.TRAIN.EVALUATE_MODE = False
config.TRAIN.BATCH_SIZE = 64
config.TRAIN.ITERS = 100000
config.TRAIN.CROSS_ENTROPY_ALPHA = 5
config.TRAIN.LR = 1e-4  # Initial learning rate
config.TRAIN.G_LR = 1e-4  # 1e-4
config.TRAIN.DECAY = True  # Whether to decay LR over learning
config.TRAIN.N_CRITIC = 5  # Critic steps per generator steps
config.TRAIN.EVAL_FREQUENCY = 20000  # How frequently to evaluate and save model
config.TRAIN.CHECKPOINT_FREQUENCY = 2000  # How frequently to evaluate and save model
config.TRAIN.SAMPLE_FREQUENCY = 1000  # How frequently to evaluate and save model
config.TRAIN.ACGAN_SCALE = 1.0
config.TRAIN.ACGAN_SCALE_FAKE = 1.0
config.TRAIN.WGAN_SCALE = 1.0
config.TRAIN.WGAN_SCALE_GP = 10.0
config.TRAIN.ACGAN_SCALE_G = 0.1
config.TRAIN.WGAN_SCALE_G = 1.0
config.TRAIN.NORMED_CROSS_ENTROPY = True
config.TRAIN.FAKE_RATIO = 1.0


def update_and_inference_config(cfg_file):
    config.merge_from_file(cfg_file)

    config.DATA.IMAGE_DIR = os.path.join(config.DATA.OUTPUT_DIR, "images")
    config.DATA.MODEL_DIR = os.path.join(config.DATA.OUTPUT_DIR, "models")
    config.DATA.LOG_DIR = os.path.join(config.DATA.OUTPUT_DIR, "logs")
    config.DATA.OUTPUT_DIM = 3 * (config.DATA.WIDTH_HEIGHT ** 2)  # Number of pixels (32*32*3)

    os.makedirs(config.DATA.IMAGE_DIR, exist_ok=True)
    os.makedirs(config.DATA.MODEL_DIR, exist_ok=True)
    os.makedirs(config.DATA.LOG_DIR, exist_ok=True)

    config.freeze()
    return config
