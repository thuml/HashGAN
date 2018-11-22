# -----------------------------------------------------------------------
# HashGAN: Deep Learning to Hash with Pair Conditional Wasserstein GAN
# Licensed under The MIT License [see LICENSE for details]
# Modified by Bin Liu
# -----------------------------------------------------------------------

import numpy as np
from scipy.misc import imsave
import tensorflow as tf


def preprocess_resize_scale_img(inputs, width_height):
    img = (inputs + 1.) * 255.99 / 2
    reshaped_image = tf.cast(img, tf.float32)
    reshaped_image = tf.reshape(
        reshaped_image, [-1, 3, width_height, width_height])

    transpose_image = tf.transpose(reshaped_image, perm=[0, 2, 3, 1])
    resized_image = tf.image.resize_bilinear(transpose_image, [256, 256])

    return resized_image


# noinspection PyUnboundLocalVariable
def save_images(x, save_path):
    # [0, 1] -> [0,255]
    if isinstance(x.flatten()[0], np.floating):
        x = (255.99 * x).astype('uint8')

    n_samples = x.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, n_samples // rows

    if x.ndim == 2:
        x = np.reshape(
            x, (x.shape[0], int(np.sqrt(x.shape[1])), int(np.sqrt(x.shape[1]))))

    if x.ndim == 4:
        # BCHW -> BHWC
        x = x.transpose((0, 2, 3, 1))
        h, w = x[0].shape[:2]
        img = np.zeros((h * nh, w * nw, 3))
    elif x.ndim == 3:
        h, w = x[0].shape[:2]
        img = np.zeros((h * nh, w * nw))
    else:
        print('x.ndim must be 3 or 4')

    for n, x in enumerate(x):
        j = n // nw
        i = n % nw
        img[j * h:j * h + h, i * w:i * w + w] = x

    imsave(save_path, img)


def scalar_summary(tag, value):
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
