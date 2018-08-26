# -----------------------------------------------------------------------
# HashGAN: Deep Learning to Hash with Pair Conditional Wasserstein GAN
# Licensed under The MIT License [see LICENSE for details]
# Modified by Bin Liu
# -----------------------------------------------------------------------
# Based on:
# Improved Training of Wasserstein GANs
# Licensed under The MIT License
# https://github.com/igul222/improved_wgan_training
# -----------------------------------------------------------------------

import os
import sys
import util
import argparse
import tflib as lib
import tflib.ops.linear
import tflib.ops.layernorm
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.save_images
import tflib.plot
from tflib.losses import cross_entropy
from datetime import datetime
import numpy as np
import tensorflow as tf
import time
from tensorflow.python.client import device_lib
import functools
import locale

sys.path.append(os.getcwd())
locale.setlocale(locale.LC_ALL, '')

GEN_BS_MULTIPLE = 2  # Generator batch size, as a multiple of BATCH_SIZE
DIM_G = 128  # Generator dimensionality
DIM_D = 128  # Critic dimensionality

USE_PRETRAIN = False
if USE_PRETRAIN:
    BATCH_SIZE = 128  # Critic batch size
    ITERS = 2000
    ARCHITECTURE = "ALEXNET"
else:
    BATCH_SIZE = 64
    ITERS = 100000  # How many iterations to train for
    ARCHITECTURE = "GOOD"  # GOOD, NROM

USE_DATASET = "nuswide81"  # "cifar10", "nuswide81", "coco"
if USE_DATASET == "cifar10":
    import dataset.cifar10 as dataset

    LABEL_DIM = 10
    DB_SIZE = 54000
    TEST_SIZE = 1000
    MAP_R = 54000
    PRETRAINED_MODEL_PATH = "models/PRETRAIN_ACGAN_SCALE_5.0_ALPHA_1.0/iteration_80000.ckpt"
    if not USE_PRETRAIN:
        ARCHITECTURE = "NORM"

elif USE_DATASET == "nuswide81":
    import dataset.nuswide81 as dataset

    LABEL_DIM = 81
    DB_SIZE = 168692
    TEST_SIZE = 5000
    MAP_R = 5000
    PRETRAINED_MODEL_PATH = "models/PRETRAIN_nuswide81_ALPHA_5/iteration_79999.ckpt"

elif USE_DATASET == "coco":
    import dataset.coco as dataset

    LABEL_DIM = 80
    DB_SIZE = 107218
    TEST_SIZE = 5000
    MAP_R = 5000
    PRETRAINED_MODEL_PATH = "models/PRETRAIN_COCO/iteration_79999.ckpt"

if ARCHITECTURE == "NORM":
    WIDTH_HEIGHT = 32
else:
    WIDTH_HEIGHT = 64

DIM = 64  # DIM for good Generator and Discriminator

NORMALIZATION_G = True  # Use batchnorm in generator?
NORMALIZATION_D = False  # Use batchnorm (or layernorm) in critic?
OUTPUT_DIM = WIDTH_HEIGHT * WIDTH_HEIGHT * 3  # Number of pixels in CIFAR10 (32*32*3)
HASH_DIM = 64
# LR = 2e-4 # previous learning rate
CROSS_ENTROPY_ALPHA = 5
LR = 1e-4  # Initial learning rate
G_LR = 1e-4  # 1e-4
DECAY = True  # Whether to decay LR over learning
N_CRITIC = 5  # Critic steps per generator steps
SAVE_FREQUENCY = 20000  # How frequently to save model

CONDITIONAL = True  # Whether to train a conditional or unconditional model
ACGAN = True  # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
ACGAN_SCALE = 1.0  # 1.0 # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 0.1  # 0.1*ACGAN_SCALE # How to scale generator's ACGAN loss relative to WGAN loss

WGAN_SCALE = 1.0  # 1.0 # How to scale the critic's ACGAN loss relative to WGAN loss
WGAN_SCALE_G = WGAN_SCALE  # How to scale generator's ACGAN loss relative to WGAN loss

PARTIAL_CROSS_ENTROPY = True
REAL_VS_FAKE = False
NORMED_CROSS_ENTROPY = True
FAKE_RATIO = 1.0

if CONDITIONAL and (not ACGAN) and (not NORMALIZATION_D):
    print("WARNING! Conditional model without normalization in D might be effectively unconditional!")

DEVICES = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
# if len(DEVICES) == 1: # Hack because the code assumes at least 2 GPUs
# DEVICES = [DEVICES[0], DEVICES[0]]

lib.print_model_settings(locals().copy())
now = str(datetime.now().strftime("%Y%m%d_%H%M%S"))

save_name = "{}_{}_D_LR_{}_G_LR_{}_ARCHITECTURE_{}_PRETRAIN_{}_PARTIAL_{}_REALFAKE_{}_NORMED{}_WGAN_SCALE_{}_ACGAN_SCALE_{}_ALPHA_{}_FAKE_RATIO_{}".format(
    USE_DATASET, now, LR, G_LR, ARCHITECTURE, USE_PRETRAIN, PARTIAL_CROSS_ENTROPY, REAL_VS_FAKE, NORMED_CROSS_ENTROPY,
    WGAN_SCALE, ACGAN_SCALE, CROSS_ENTROPY_ALPHA, FAKE_RATIO)

img_save_folder = os.path.join("images", save_name)
model_save_folder = os.path.join("models", save_name)
os.makedirs(img_save_folder)
os.makedirs(model_save_folder)


def nonlinearity(x):
    return tf.nn.relu(x)


def Normalize(name, inputs, labels=None):
    """This is messy, but basically it chooses between batchnorm, layernorm,
    their conditional variants, or nothing, depending on the value of `name` and
    the global hyperparam flags."""
    if not CONDITIONAL:
        labels = None
    if CONDITIONAL and ACGAN and ('Discriminator' in name):
        labels = None

    if ('Discriminator' in name) and NORMALIZATION_D:
        return lib.ops.layernorm.Layernorm(name, [1, 2, 3], inputs, labels=labels, n_labels=10)
    elif ('Generator' in name) and NORMALIZATION_G:
        return lib.ops.batchnorm.Batchnorm(name, [0, 2, 3], inputs, fused=True)
    else:
        return inputs


def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n(
        [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    return output


def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n(
        [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output


def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0, 3, 1, 2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output


def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, labels=None):
    """
    resample: None, 'down', or 'up'
    """
    if resample == 'down':
        conv_1 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2 = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = ConvMeanPool
    elif resample == 'up':
        conv_1 = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = UpsampleConv
        conv_2 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample is None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
        conv_2 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim == input_dim and resample is None:
        shortcut = inputs  # Identity skip-connection
    else:
        shortcut = conv_shortcut(name + '.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name + '.N1', output, labels=labels)
    output = nonlinearity(output)
    output = conv_1(name + '.Conv1', filter_size=filter_size, inputs=output)
    output = Normalize(name + '.N2', output, labels=labels)
    output = nonlinearity(output)
    output = conv_2(name + '.Conv2', filter_size=filter_size, inputs=output)

    return shortcut + output


def OptimizedResBlockDisc1(inputs):
    conv_1 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=3, output_dim=DIM_D)
    conv_2 = functools.partial(ConvMeanPool, input_dim=DIM_D, output_dim=DIM_D)
    conv_shortcut = MeanPoolConv
    shortcut = conv_shortcut('Discriminator.1.Shortcut', input_dim=3, output_dim=DIM_D, filter_size=1, he_init=False,
                             biases=True, inputs=inputs)

    output = inputs
    output = conv_1('Discriminator.1.Conv1', filter_size=3, inputs=output)
    output = nonlinearity(output)
    output = conv_2('Discriminator.1.Conv2', filter_size=3, inputs=output)
    return shortcut + output


def oldGenerator(n_samples, labels, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 256])

    # concat noise with label
    noise = tf.concat([tf.cast(labels, tf.float32), tf.slice(noise, [0, LABEL_DIM], [-1, -1])], 1)
    output = lib.ops.linear.Linear('Generator.Input', 256, 4 * 4 * DIM_G, noise)
    output = tf.reshape(output, [-1, DIM_G, 4, 4])
    output = ResidualBlock('Generator.1', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.2', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.3', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = Normalize('Generator.OutputN', output)
    output = nonlinearity(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', DIM_G, 3, 3, output, he_init=False)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, OUTPUT_DIM])


def oldDiscriminator(inputs, labels):
    output = tf.reshape(inputs, [-1, 3, 32, 32])
    output = OptimizedResBlockDisc1(output)
    output = ResidualBlock('Discriminator.2', DIM_D, DIM_D, 3, output, resample='down', labels=labels)
    output = ResidualBlock('Discriminator.3', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Discriminator.4', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    output = nonlinearity(output)
    output = tf.reduce_mean(output, axis=[2, 3])
    output_wgan = lib.ops.linear.Linear('Discriminator.Output', DIM_D, 1, output)
    output_wgan = tf.reshape(output_wgan, [-1])
    if CONDITIONAL and ACGAN:
        output_acgan = lib.ops.linear.Linear('Discriminator.ACGANOutput', DIM_D, HASH_DIM, output)
        output_acgan = tf.nn.tanh(output_acgan)
        return output_wgan, output_acgan
    else:
        return output_wgan, None


def GoodGenerator(n_samples, labels, noise=None, dim=DIM):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    noise = tf.concat([tf.cast(labels, tf.float32), tf.slice(noise, [0, LABEL_DIM], [-1, -1])], 1)

    output = lib.ops.linear.Linear('Generator.Input', 128, 4 * 4 * 8 * dim, noise)
    output = tf.reshape(output, [-1, 8 * dim, 4, 4])

    output = ResidualBlock('Generator.Res1', 8 * dim, 8 * dim, 3, output, resample='up')
    output = ResidualBlock('Generator.Res2', 8 * dim, 4 * dim, 3, output, resample='up')
    output = ResidualBlock('Generator.Res3', 4 * dim, 2 * dim, 3, output, resample='up')
    output = ResidualBlock('Generator.Res4', 2 * dim, 1 * dim, 3, output, resample='up')

    output = Normalize('Generator.OutputN', output)
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', 1 * dim, 3, 3, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])


def GoodDiscriminator(inputs, dim=DIM):
    output = tf.reshape(inputs, [-1, 3, 64, 64])
    output = lib.ops.conv2d.Conv2D('Discriminator.Input', 3, dim, 3, output, he_init=False)

    output = ResidualBlock('Discriminator.Res1', dim, 2 * dim, 3, output, resample='down')
    output = ResidualBlock('Discriminator.Res2', 2 * dim, 4 * dim, 3, output, resample='down')
    output = ResidualBlock('Discriminator.Res3', 4 * dim, 8 * dim, 3, output, resample='down')
    output = ResidualBlock('Discriminator.Res4', 8 * dim, 8 * dim, 3, output, resample='down')

    output = tf.reshape(output, [-1, 4 * 4 * 8 * dim])
    output_wgan = lib.ops.linear.Linear('Discriminator.Output', 4 * 4 * 8 * dim, 1, output)

    if CONDITIONAL and ACGAN:
        output_acgan = lib.ops.linear.Linear('Discriminator.ACGANOutput', 4 * 4 * 8 * dim, HASH_DIM, output)
        output_acgan = tf.nn.tanh(output_acgan)
        return output_wgan, output_acgan
    else:
        return output_wgan, None


def preprocess_resize_scale_img(inputs):
    img = (inputs + 1.) * 255.99 / 2
    reshaped_image = tf.cast(img, tf.float32)
    reshaped_image = tf.reshape(reshaped_image, [-1, 3, WIDTH_HEIGHT, WIDTH_HEIGHT])

    transpose_image = tf.transpose(reshaped_image, perm=[0, 2, 3, 1])
    resized_image = tf.image.resize_bilinear(transpose_image, [256, 256])

    return resized_image


def AlexnetDiscriminator(inputs, stage="train"):
    # with tf.name_scope('Discriminator.preprocess') as scope:
    net_data = dict(np.load("tflib/reference_pretrain.npy", encoding='latin1').item())

    if inputs.shape[1] != 256:
        inputs = preprocess_resize_scale_img(inputs)

    reshaped_image = inputs
    # reshaped_image = tf.reshape(reshaped_image,[BATCH_SIZE, 256 , 256, 3])

    IMAGE_SIZE = 227
    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Randomly crop a [height, width] section of each image
    if stage == "train":
        distorted_image = tf.stack(
            [tf.random_crop(tf.image.random_flip_left_right(each_image), [height, width, 3]) for each_image in
             tf.unstack(reshaped_image)])
    else:
        # Randomly crop a [height, width] section of each image
        distorted_image1 = tf.stack(
            [tf.image.crop_to_bounding_box(tf.image.flip_left_right(each_image), 0, 0, height, width) for each_image in
             tf.unstack(reshaped_image)])
        distorted_image2 = tf.stack(
            [tf.image.crop_to_bounding_box(tf.image.flip_left_right(each_image), 28, 28, height, width) for each_image
             in tf.unstack(reshaped_image)])
        distorted_image3 = tf.stack(
            [tf.image.crop_to_bounding_box(tf.image.flip_left_right(each_image), 28, 0, height, width) for each_image in
             tf.unstack(reshaped_image)])
        distorted_image4 = tf.stack(
            [tf.image.crop_to_bounding_box(tf.image.flip_left_right(each_image), 0, 28, height, width) for each_image in
             tf.unstack(reshaped_image)])
        distorted_image5 = tf.stack(
            [tf.image.crop_to_bounding_box(tf.image.flip_left_right(each_image), 14, 14, height, width) for each_image
             in tf.unstack(reshaped_image)])

        distorted_image6 = tf.stack([tf.image.crop_to_bounding_box(each_image, 0, 0, height, width) for each_image in
                                     tf.unstack(reshaped_image)])
        distorted_image7 = tf.stack([tf.image.crop_to_bounding_box(each_image, 28, 28, height, width) for each_image in
                                     tf.unstack(reshaped_image)])
        distorted_image8 = tf.stack([tf.image.crop_to_bounding_box(each_image, 28, 0, height, width) for each_image in
                                     tf.unstack(reshaped_image)])
        distorted_image9 = tf.stack([tf.image.crop_to_bounding_box(each_image, 0, 28, height, width) for each_image in
                                     tf.unstack(reshaped_image)])
        distorted_image0 = tf.stack([tf.image.crop_to_bounding_box(each_image, 14, 14, height, width) for each_image in
                                     tf.unstack(reshaped_image)])

        distorted_image = tf.concat(
            [distorted_image1, distorted_image2, distorted_image3, distorted_image4, distorted_image5, distorted_image6,
             distorted_image7, distorted_image8, distorted_image9, distorted_image0], 0)

    # Zero-mean input
    mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32, shape=[1, 1, 1, 3], name='img-mean')
    distorted_image = distorted_image - mean

    # Conv1
    # Output 96, kernel 11, stride 4
    scope = 'Discriminator.conv1.'
    kernel = lib.param(scope + 'weights', net_data['conv1'][0])
    biases = lib.param(scope + 'biases', net_data['conv1'][1])
    conv = tf.nn.conv2d(distorted_image, kernel, [1, 4, 4, 1], padding='VALID')
    out = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(out, name=scope)

    # Pool1
    pool1 = tf.nn.max_pool(conv1,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool1')

    # LRN1
    if WGAN_SCALE == 0:
        radius = 2
        alpha = 2e-05
        beta = 0.75
        bias = 1.0
        lrn1 = tf.nn.local_response_normalization(pool1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
    else:
        lrn1 = pool1

    # Conv2
    # Output 256, pad 2, kernel 5, group 2
    scope = 'Discriminator.conv2.'
    kernel = lib.param(scope + '.weights', net_data['conv2'][0])
    biases = lib.param(scope + '.biases', net_data['conv2'][1])
    group = 2
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
    input_groups = tf.split(lrn1, group, 3)
    kernel_groups = tf.split(kernel, group, 3)
    output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
    # Concatenate the groups
    conv = tf.concat(output_groups, 3)
    out = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(out, name=scope)

    # Pool2
    pool2 = tf.nn.max_pool(conv2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool2')

    # LRN2
    if WGAN_SCALE == 0:
        radius = 2
        alpha = 2e-05
        beta = 0.75
        bias = 1.0
        lrn2 = tf.nn.local_response_normalization(pool2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
    else:
        lrn2 = pool2

    # Conv3
    # Output 384, pad 1, kernel 3
    scope = 'Discriminator.conv3.'
    kernel = lib.param(scope + '.weights', net_data['conv3'][0])
    biases = lib.param(scope + '.biases', net_data['conv3'][1])
    conv = tf.nn.conv2d(lrn2, kernel, [1, 1, 1, 1], padding='SAME')
    out = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(out, name=scope)

    # Conv4
    # Output 384, pad 1, kernel 3, group 2
    scope = 'Discriminator.conv4.'
    kernel = lib.param(scope + '.weights', net_data['conv4'][0])
    biases = lib.param(scope + '.biases', net_data['conv4'][1])
    group = 2
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
    input_groups = tf.split(conv3, group, 3)
    kernel_groups = tf.split(kernel, group, 3)
    output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
    # Concatenate the groups
    conv = tf.concat(output_groups, 3)
    out = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(out, name=scope)

    # Conv5
    # Output 256, pad 1, kernel 3, group 2
    scope = 'Discriminator.conv5.'
    kernel = lib.param(scope + '.weights', net_data['conv5'][0])
    biases = lib.param(scope + '.biases', net_data['conv5'][1])
    group = 2
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
    input_groups = tf.split(conv4, group, 3)
    kernel_groups = tf.split(kernel, group, 3)
    output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
    # Concatenate the groups
    conv = tf.concat(output_groups, 3)
    out = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(out, name=scope)

    # Pool5
    pool5 = tf.nn.max_pool(conv5,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool5')

    # FC6
    # Output 4096
    # with tf.name_scope('Discriminator.fc6') as scope:
    shape = int(np.prod(pool5.get_shape()[1:]))
    scope = 'Discriminator.fc6.'
    fc6w = lib.param(scope + '.weights', net_data['fc6'][0])
    fc6b = lib.param(scope + '.biases', net_data['fc6'][1])
    pool5_flat = tf.reshape(pool5, [-1, shape])
    fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
    fc6 = tf.nn.dropout(tf.nn.relu(fc6l), 0.5)

    # FC7
    # Output 4096
    scope = 'Discriminator.fc7.'
    fc7w = lib.param(scope + '.weights', net_data['fc7'][0])
    fc7b = lib.param(scope + '.biases', net_data['fc7'][1])
    fc7l = tf.nn.bias_add(tf.matmul(fc6, fc7w), fc7b)
    fc7 = tf.nn.dropout(tf.nn.relu(fc7l), 0.5)

    # FC8
    # Output output_dim
    fc8 = lib.ops.linear.Linear('Discriminator.ACGANOutput', 4096, HASH_DIM, fc7)
    if stage == "train":
        output = tf.nn.tanh(fc8)
    else:
        fc8_t = tf.nn.tanh(fc8)
        fc8_t = tf.concat([tf.expand_dims(i, 0) for i in tf.split(fc8_t, 10, 0)], 0)
        output = tf.reduce_mean(fc8_t, 0)
    output_wgan = lib.ops.linear.Linear('Discriminator.Output', 4096, 1, fc7)

    return output_wgan, output


def Generator(n_samples, labels, noise=None):
    if ARCHITECTURE == "NORM":
        return oldGenerator(n_samples, labels, noise)
    else:
        return GoodGenerator(n_samples, labels, noise)


def Discriminator(inputs, labels, stage="train"):
    if ARCHITECTURE == "GOOD":
        return GoodDiscriminator(inputs)
    elif ARCHITECTURE == "ALEXNET":
        return AlexnetDiscriminator(inputs, stage)
    else:
        return oldDiscriminator(inputs, labels)


def average_gradients(tower_grads, alpha=1.0):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        g_is_none = False
        for g, v in grad_and_vars:
            if g is not None:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
                var = v
            else:
                g_is_none = True
                break

        if g_is_none:
            continue

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        grad = grad * alpha

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        # v = grad_and_vars[0][1]
        grad_and_var = (grad, var)
        average_grads.append(grad_and_var)
    return average_grads


def main():
    configProto = tf.ConfigProto()
    configProto.gpu_options.allow_growth = True
    configProto.allow_soft_placement = True
    with tf.Session(config=configProto) as session:

        _iteration = tf.placeholder(tf.int32, shape=None)

        # unlabeled data initialization
        all_unlabel_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
        all_unlabel_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE, LABEL_DIM])
        unlabel_labels_splits = tf.split(all_unlabel_labels, len(DEVICES), axis=0)

        unlabel_fake_data_splits = []
        for i, device in enumerate(DEVICES):
            with tf.device(device):
                unlabel_fake_data_splits.append(Generator(BATCH_SIZE // len(DEVICES), unlabel_labels_splits[i]))

        all_unlabel_data = tf.reshape(2 * ((tf.cast(all_unlabel_data_int, tf.float32) / 256.) - .5),
                                      [BATCH_SIZE, OUTPUT_DIM])
        all_unlabel_data += tf.random_uniform(shape=[BATCH_SIZE, OUTPUT_DIM], minval=0., maxval=1. / 128)  # dequantize
        all_unlabel_data_splits = tf.split(all_unlabel_data, len(DEVICES), axis=0)

        # labeled data init
        all_real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
        all_real_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE, LABEL_DIM])
        labels_splits = tf.split(all_real_labels, len(DEVICES), axis=0)

        fake_data_splits = []
        for i, device in enumerate(DEVICES):
            with tf.device(device):
                fake_data_splits.append(Generator(BATCH_SIZE // len(DEVICES), labels_splits[i]))

        all_real_data = tf.reshape(2 * ((tf.cast(all_real_data_int, tf.float32) / 256.) - .5), [BATCH_SIZE, OUTPUT_DIM])
        all_real_data += tf.random_uniform(shape=[BATCH_SIZE, OUTPUT_DIM], minval=0., maxval=1. / 128)  # dequantize
        all_real_data_splits = tf.split(all_real_data, len(DEVICES), axis=0)

        # init optimizer
        if DECAY:
            # decay = tf.train.exponential_decay(1.0, _iteration, 500, 0.5, staircase=True)
            decay = tf.maximum(0., 1. - (tf.cast(_iteration, tf.float32) / ITERS))
        else:
            decay = 1.0
        # TODO
        # if ARCHITECTURE == "ALEXNET":
        #   disc_opt = tf.train.MomentumOptimizer(learning_rate=LR*decay, momentum=0.9)
        # else:
        disc_opt = tf.train.AdamOptimizer(learning_rate=LR * decay, beta1=0., beta2=0.9)
        gen_opt = tf.train.AdamOptimizer(learning_rate=G_LR * decay, beta1=0., beta2=0.9)

        disc_costs = []
        disc_acgan_costs = []
        disc_acgan_costs_real_real = []

        disc_costs_gs = []
        disc_acgan_costs_gs = []

        disc_costs_wgan = []
        disc_costs_gradient_penalty = []
        for i, device in enumerate(DEVICES):
            with tf.device(device):
                real_and_fake_data = tf.concat([
                    all_unlabel_data_splits[i],
                    all_real_data_splits[i],
                    fake_data_splits[i],
                    unlabel_fake_data_splits[i],
                ], axis=0)
                real_and_fake_labels = tf.concat([
                    unlabel_labels_splits[i],
                    labels_splits[i],
                    labels_splits[i],
                    unlabel_labels_splits[i],
                ], axis=0)
                disc_all, disc_all_acgan = Discriminator(real_and_fake_data, real_and_fake_labels)
                # size * 2 for unlabeled data
                disc_real = disc_all[:BATCH_SIZE // len(DEVICES) * 2]
                disc_fake = disc_all[BATCH_SIZE // len(DEVICES) * 2:]
                disc_costs.append(tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real))
                disc_costs_wgan.append(tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real))

                # gradients computation
                disc_costs_gs.append(disc_opt.compute_gradients((tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)),
                                                                var_list=lib.params_with_name('Discriminator.')))

                if CONDITIONAL and ACGAN:
                    pos_start = BATCH_SIZE // len(DEVICES)
                    pos_middle = 2 * BATCH_SIZE // len(DEVICES)
                    pos_end = 3 * BATCH_SIZE // len(DEVICES)

                    var_list = lib.params_with_name('Discriminator.')
                    if not PARTIAL_CROSS_ENTROPY:
                        cost_all = cross_entropy(
                            disc_all_acgan[pos_start:pos_end],
                            real_and_fake_labels[pos_start:pos_end],
                            alpha=CROSS_ENTROPY_ALPHA,
                            normed=NORMED_CROSS_ENTROPY)
                        # real & fake together
                        disc_acgan_costs.append(cost_all)
                        # gradients computation
                        disc_acgan_costs_gs.append(disc_opt.compute_gradients(cost_all, var_list=var_list))

                    elif REAL_VS_FAKE:
                        # real vs fake, fake cannot influence real
                        cost_fr = cross_entropy(
                            disc_all_acgan[pos_start:pos_middle],
                            real_and_fake_labels[pos_start:pos_middle],
                            disc_all_acgan[pos_middle:pos_end],
                            real_and_fake_labels[pos_middle:pos_end], alpha=CROSS_ENTROPY_ALPHA, partial=False,
                            normed=NORMED_CROSS_ENTROPY)
                        disc_acgan_costs.append(cost_fr)
                        # gradients computation
                        disc_acgan_costs_gs.append(disc_opt.compute_gradients(cost_fr, var_list=var_list))
                    else:
                        # real vs real
                        cost_rr = cross_entropy(
                            disc_all_acgan[pos_start:pos_middle],
                            real_and_fake_labels[pos_start:pos_middle],
                            alpha=CROSS_ENTROPY_ALPHA,
                            normed=NORMED_CROSS_ENTROPY)
                        disc_acgan_costs.append(cost_rr)
                        # gradients computation
                        disc_acgan_costs_gs.append(disc_opt.compute_gradients(cost_rr, var_list=var_list))
                        # real vs fake, fake cannot influence real
                        if FAKE_RATIO != 0.0:
                            cost_fr = FAKE_RATIO * cross_entropy(
                                disc_all_acgan[pos_start:pos_middle],
                                real_and_fake_labels[pos_start:pos_middle],
                                disc_all_acgan[pos_middle:pos_end],
                                real_and_fake_labels[pos_middle:pos_end], alpha=CROSS_ENTROPY_ALPHA, partial=True,
                                normed=NORMED_CROSS_ENTROPY)
                            disc_acgan_costs.append(cost_fr)
                            disc_acgan_costs_gs.append(disc_opt.compute_gradients(cost_fr, var_list=var_list))

                    disc_acgan_costs_real_real.append(cross_entropy(
                        disc_all_acgan[pos_start:pos_middle],
                        real_and_fake_labels[pos_start:pos_middle],
                        alpha=CROSS_ENTROPY_ALPHA,
                        normed=NORMED_CROSS_ENTROPY))

        if WGAN_SCALE != 0:
            for i, device in enumerate(DEVICES):
                with tf.device(device):
                    real_data = tf.concat([all_unlabel_data_splits[i], all_real_data_splits[i]], axis=0)
                    fake_data = tf.concat([fake_data_splits[i], unlabel_fake_data_splits[i]], axis=0)
                    labels = tf.concat([
                        unlabel_labels_splits[i],
                        labels_splits[i],
                    ], axis=0)
                    alpha = tf.random_uniform(
                        shape=[2 * BATCH_SIZE // len(DEVICES), 1],
                        minval=0.,
                        maxval=1.
                    )
                    if ARCHITECTURE == "ALEXNET":
                        real_data = preprocess_resize_scale_img(real_data)
                        fake_data = preprocess_resize_scale_img(fake_data)
                        alpha = tf.random_uniform(
                            shape=[2 * BATCH_SIZE // len(DEVICES), 1, 1, 1],
                            minval=0.,
                            maxval=1.
                        )

                    differences = fake_data - real_data
                    interpolates = real_data + (alpha * differences)
                    gradients = tf.gradients(Discriminator(interpolates, labels)[0], [interpolates])[0]
                    if ARCHITECTURE == "ALEXNET":
                        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
                    else:
                        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                    gradient_penalty = 10 * tf.reduce_mean((slopes - 1.) ** 2)
                    disc_costs.append(gradient_penalty)
                    disc_costs_gradient_penalty.append(gradient_penalty)

                    # gradients computation
                    disc_costs_gs.append(
                        disc_opt.compute_gradients(gradient_penalty, var_list=lib.params_with_name('Discriminator.')))

            disc_wgan_gradient = tf.add_n(disc_costs_gradient_penalty) / len(DEVICES)

        disc_wgan = tf.add_n(disc_costs) / len(DEVICES)
        disc_wgan_l = tf.add_n(disc_costs_wgan) / len(DEVICES)

        if CONDITIONAL and ACGAN:
            disc_acgan = tf.add_n(disc_acgan_costs) / len(DEVICES)
            disc_acgan_real_real = tf.add_n(disc_acgan_costs_real_real) / len(DEVICES)
            disc_acgan_acc = tf.constant(0.)  # disc_acgan_accs) / len(DEVICES_A)
            disc_acgan_fake_acc = tf.constant(0.)  # disc_acgan_fake_accs) / len(DEVICES_A)

            disc_cost = ACGAN_SCALE * disc_acgan

            disc_gv = average_gradients(disc_acgan_costs_gs, ACGAN_SCALE)
            if WGAN_SCALE != 0:
                disc_cost = disc_cost + WGAN_SCALE * disc_wgan
                disc_gv = disc_gv + average_gradients(disc_costs_gs, WGAN_SCALE)

        else:
            disc_acgan = tf.constant(0.)
            disc_acgan_acc = tf.constant(0.)
            disc_acgan_fake_acc = tf.constant(0.)
            disc_cost = disc_wgan

        gen_costs = []
        gen_acgan_costs = []

        gen_costs_gs = []
        gen_acgan_costs_gs = []

        def to_one_hot(sparse_labels):
            return tf.one_hot(sparse_labels, LABEL_DIM, dtype=tf.int32)

        # for device in DEVICES:
        for i, device in enumerate(DEVICES):
            with tf.device(device):
                if CONDITIONAL and ACGAN:
                    n_samples = BATCH_SIZE // len(DEVICES)
                    fake_data = Generator(n_samples, labels_splits[i])
                    real_and_fake_data = tf.concat([
                        all_real_data_splits[i],
                        fake_data
                    ], axis=0)
                    real_and_fake_labels = tf.concat([
                        labels_splits[i],
                        labels_splits[i],
                    ], axis=0)
                    disc_all, disc_all_acgan = Discriminator(real_and_fake_data, real_and_fake_labels)
                    disc_fake = disc_all[n_samples:]
                    gen_costs.append(-tf.reduce_mean(disc_fake))
                    gen_acgan_costs.append(cross_entropy(
                        disc_all_acgan[:n_samples],
                        real_and_fake_labels[:n_samples],
                        disc_all_acgan[n_samples:],
                        real_and_fake_labels[n_samples:],
                        alpha=CROSS_ENTROPY_ALPHA, partial=True,
                        normed=NORMED_CROSS_ENTROPY))
                    gen_costs_gs.append(gen_opt.compute_gradients(-tf.reduce_mean(disc_fake),
                                                                  var_list=lib.params_with_name('Generator')))
                    gen_acgan_costs_gs.append(gen_opt.compute_gradients(cross_entropy(
                        disc_all_acgan[:n_samples],
                        real_and_fake_labels[:n_samples],
                        disc_all_acgan[n_samples:],
                        real_and_fake_labels[n_samples:],
                        alpha=CROSS_ENTROPY_ALPHA, partial=True,
                        normed=NORMED_CROSS_ENTROPY), var_list=lib.params_with_name('Generator')))
                else:
                    n_samples = GEN_BS_MULTIPLE * BATCH_SIZE // len(DEVICES)
                    fake_labels = to_one_hot(tf.cast(tf.random_uniform([n_samples]) * 10, tf.int32))
                    gen_costs.append(-tf.reduce_mean(Discriminator(Generator(n_samples, fake_labels), fake_labels)[0]))

        # set acgan_output
        disc_real_acgan = []
        disc_real_acgan_cost_t = []
        for i, device in enumerate(DEVICES):
            with tf.device(device):
                if CONDITIONAL and ACGAN:
                    real_data = all_real_data_splits[i]
                    real_labels = labels_splits[i]

                    _, _disc_real_acgan = Discriminator(real_data, real_labels, "val")
                    disc_real_acgan.append(_disc_real_acgan)

                    disc_real_acgan_cost_t.append(cross_entropy(
                        _disc_real_acgan,
                        real_labels,
                        alpha=CROSS_ENTROPY_ALPHA,
                        normed=NORMED_CROSS_ENTROPY))
        disc_real_acgan_cost = tf.add_n(disc_real_acgan_cost_t) / len(DEVICES)

        gen_cost = WGAN_SCALE_G * (tf.add_n(gen_costs) / len(DEVICES))
        gen_gv = average_gradients(gen_costs_gs, WGAN_SCALE_G)
        if CONDITIONAL and ACGAN:
            gen_cost += (ACGAN_SCALE_G * (tf.add_n(gen_acgan_costs) / len(DEVICES)))
            gen_gv = gen_gv + average_gradients(gen_acgan_costs_gs, ACGAN_SCALE_G)

        gen_train_op = gen_opt.apply_gradients(gen_gv)
        disc_train_op = disc_opt.apply_gradients(disc_gv)

        # Function for generating samples
        noise_dim = 256 if USE_DATASET == "cifar10" else 128  # TODO: refactor
        fixed_noise = tf.constant(np.random.normal(size=(100, noise_dim)).astype('float32'))
        fixed_labels = to_one_hot(tf.constant(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10, dtype='int32')))
        fixed_noise_samples = Generator(100, fixed_labels, noise=fixed_noise)

        def generate_image(frame):
            samples = session.run(fixed_noise_samples)
            samples = ((samples + 1.) * (255. / 2)).astype('int32')
            lib.save_images.save_images(samples.reshape((100, 3, WIDTH_HEIGHT, WIDTH_HEIGHT)),
                                        '{}/samples_{}.png'.format(img_save_folder, frame))

        fake_labels_100 = to_one_hot(tf.cast(tf.random_uniform([100]) * 10, tf.int32))
        train_gen, unlabel_train_gen, dev_gen = dataset.load(BATCH_SIZE, WIDTH_HEIGHT)

        def inf_train_gen():
            while True:
                for images, _labels in train_gen():
                    yield images, _labels

        def inf_unlabel_train_gen():
            while True:
                for images, _labels in unlabel_train_gen():
                    yield images, _labels

        # compute param size
        print("computing param size")
        for name, grads_and_vars in [('G', gen_gv), ('D', disc_gv)]:
            print("{} Params:".format(name))
            total_param_count = 0
            for g, v in grads_and_vars:
                shape = v.get_shape()
                shape_str = ",".join([str(x) for x in v.get_shape()])

                param_count = 1
                for dim in shape:
                    param_count *= int(dim)
                total_param_count += param_count

                if g is None:
                    print("\t{} ({}) [no grad!]".format(v.name, shape_str))
                else:
                    print("\t{} ({})".format(v.name, shape_str))
            print("Total param count: {}".format(
                locale.format("%d", total_param_count, grouping=True)
            ))

        print("initializing global variables")
        session.run(tf.global_variables_initializer())

        gen = inf_train_gen()
        unlabel_gen = inf_unlabel_train_gen()

        if USE_PRETRAIN:
            saver = tf.train.Saver(lib.params_with_name('Generator'))
            saver.restore(session, PRETRAINED_MODEL_PATH)
            print("model restored")

        print("training")
        for iteration in range(ITERS):
            start_time = time.time()

            if iteration > 0:
                if G_LR != 0:
                    _data, _labels = next(gen)
                    # _ = session.run([gen_train_op], feed_dict={_iteration:iteration})
                    _ = session.run([gen_train_op], feed_dict={
                        all_real_data_int: _data,
                        all_real_labels: _labels,
                        _iteration: iteration,
                    })

            for i in range(N_CRITIC):
                _data, _labels = next(gen)
                _unlabel_data, _unlabel_labels = next(unlabel_gen)
                if CONDITIONAL and ACGAN:
                    if WGAN_SCALE == 0:
                        _disc_cost, _disc_acgan, _disc_acgan_real_real, _disc_acgan_acc, _disc_acgan_fake_acc, _ = session.run(
                            [disc_cost, disc_acgan, disc_acgan_real_real, disc_acgan_acc, disc_acgan_fake_acc,
                             disc_train_op],
                            feed_dict={
                                all_real_data_int: _data,
                                all_real_labels: _labels,
                                all_unlabel_data_int: _unlabel_data,
                                all_unlabel_labels: _unlabel_labels,
                                _iteration: iteration,
                            })
                    else:
                        _disc_cost, _disc_wgan, _disc_wgan_l, _disc_wgan_gradient, _disc_acgan, _disc_acgan_real_real, _disc_acgan_acc, _disc_acgan_fake_acc, _ = session.run(
                            [disc_cost, disc_wgan, disc_wgan_l, disc_wgan_gradient, disc_acgan, disc_acgan_real_real,
                             disc_acgan_acc, disc_acgan_fake_acc, disc_train_op],
                            feed_dict={
                                all_real_data_int: _data,
                                all_real_labels: _labels,
                                all_unlabel_data_int: _unlabel_data,
                                all_unlabel_labels: _unlabel_labels,
                                _iteration: iteration,
                            })
                else:
                    _disc_cost, _ = session.run([disc_cost, disc_train_op],
                                                feed_dict={all_real_data_int: _data, all_real_labels: _labels,
                                                           _iteration: iteration})

            lib.plot.plot('cost', _disc_cost)
            if CONDITIONAL and ACGAN:
                if not PARTIAL_CROSS_ENTROPY:
                    lib.plot.plot('acgan', _disc_acgan)
                else:
                    lib.plot.plot('acgan_f', _disc_acgan - _disc_acgan_real_real)
                lib.plot.plot('acgan_r', _disc_acgan_real_real)

                if WGAN_SCALE != 0:
                    lib.plot.plot('wgan', _disc_wgan)
                    lib.plot.plot('wgan_l', _disc_wgan_l)
                    lib.plot.plot('wgan_g', _disc_wgan_gradient)
            lib.plot.plot('time', time.time() - start_time)

            if (iteration + 1) % 1000 == 0:
                generate_image(iteration)

            # calculate mAP score w.r.t all db data every 10000 iters
            if (iteration + 1) % 10000 == 0:
                _db_gen, _test_gen = dataset.load_val(BATCH_SIZE, WIDTH_HEIGHT)
                db_output = []
                db_labels = []
                test_output = []
                test_labels = []
                for images, _labels in _test_gen():
                    _disc_acgan_output, __cost = session.run([disc_real_acgan, disc_real_acgan_cost],
                                                             feed_dict={all_real_data_int: images,
                                                                        all_real_labels: _labels})
                    test_output.append(_disc_acgan_output)
                    test_labels.append(_labels)

                for images, _labels in _db_gen():
                    _disc_acgan_output, _ = session.run([disc_real_acgan, disc_real_acgan_cost],
                                                        feed_dict={all_real_data_int: images, all_real_labels: _labels})
                    db_output.append(_disc_acgan_output)
                    db_labels.append(_labels)

                db = argparse.Namespace()
                db.output = np.reshape(np.array(db_output), [-1, HASH_DIM])[:DB_SIZE, :]
                db.label = np.reshape(np.array(db_labels), [-1, LABEL_DIM])[:DB_SIZE, :]
                test = argparse.Namespace()
                test.output = np.reshape(np.array(test_output), [-1, HASH_DIM])[:TEST_SIZE, :]
                test.label = np.reshape(np.array(test_labels), [-1, LABEL_DIM])[:TEST_SIZE, :]

                mAP_ = util.MAPs(MAP_R)
                mAP_val = mAP_.get_mAPs_by_feature(db, test)
                lib.plot.plot("mAP_feature", mAP_val)

            if (iteration < 500) or (iteration % 1000 == 999):
                lib.plot.flush(img_save_folder)

            if (iteration + 1) % SAVE_FREQUENCY == 0 or iteration + 1 == ITERS:
                save_path = os.path.join(model_save_folder, "iteration_{}.ckpt".format(iteration))
                saver = tf.train.Saver()
                saver.save(session, save_path)
                print(("Model saved in file: %s" % save_path))

            lib.plot.tick()


if __name__ == "__main__":
    main()
