import functools

import numpy as np
import tensorflow as tf

from lib.ops import conv2D, linear, batch_norm
from lib.util import preprocess_resize_scale_img
from lib.params import param


def normalize(name, inputs):
    """This is messy, but basically it chooses between batchnorm, layernorm,
    their conditional variants, or nothing, depending on the value of `name` and
    hyperparam flags."""

    if 'generator' in name:
        return batch_norm(name, [0, 2, 3], inputs, fused=True)
    else:
        return inputs


def conv_mean_pool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = conv2D(
        name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n(
        [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    return output


def mean_pool_conv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n(
        [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    output = conv2D(
        name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output


def upsample_conv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0, 3, 1, 2])
    output = conv2D(
        name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output


def residual_block(name, input_dim, output_dim, filter_size, inputs, resample=None):
    """
    resample: None, 'down', or 'up'
    """
    if resample == 'down':
        conv_1 = functools.partial(
            conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2 = functools.partial(
            conv_mean_pool, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = conv_mean_pool
    elif resample == 'up':
        conv_1 = functools.partial(
            upsample_conv, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = upsample_conv
        conv_2 = functools.partial(
            conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample is None:
        conv_shortcut = conv2D
        conv_1 = functools.partial(
            conv2D, input_dim=input_dim, output_dim=output_dim)
        conv_2 = functools.partial(
            conv2D, input_dim=output_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim == input_dim and resample is None:
        shortcut = inputs  # Identity skip-connection
    else:
        shortcut = conv_shortcut(name + '.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = normalize(name + '.N1', output)
    output = tf.nn.relu(output)
    output = conv_1(name + '.Conv1', filter_size=filter_size, inputs=output)
    output = normalize(name + '.N2', output)
    output = tf.nn.relu(output)
    output = conv_2(name + '.Conv2', filter_size=filter_size, inputs=output)

    return shortcut + output


def optimized_res_block_disc1(inputs, cfg):
    conv_1 = functools.partial(
        conv2D, input_dim=3, output_dim=cfg.MODEL.DIM_D)
    conv_2 = functools.partial(conv_mean_pool, input_dim=cfg.MODEL.DIM_D, output_dim=cfg.MODEL.DIM_D)
    conv_shortcut = mean_pool_conv
    shortcut = conv_shortcut('discriminator.1.Shortcut', input_dim=3, output_dim=cfg.MODEL.DIM_D,
                             filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = conv_1('discriminator.1.Conv1', filter_size=3, inputs=output)
    output = tf.nn.relu(output)
    output = conv_2('discriminator.1.Conv2', filter_size=3, inputs=output)
    return shortcut + output


def old_generator(n_samples, labels, cfg, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 256])

    # concat noise with label
    noise = tf.concat([tf.cast(labels, tf.float32), tf.slice(
        noise, [0, cfg.DATA.LABEL_DIM], [-1, -1])], 1)
    output = linear(
        'generator.Input', 256, 4 * 4 * cfg.MODEL.DIM_G, noise)
    output = tf.reshape(output, [-1, cfg.MODEL.DIM_G, 4, 4])
    output = residual_block('generator.1', cfg.MODEL.DIM_G, cfg.MODEL.DIM_G,
                            3, output, resample='up')
    output = residual_block('generator.2', cfg.MODEL.DIM_G, cfg.MODEL.DIM_G,
                            3, output, resample='up')
    output = residual_block('generator.3', cfg.MODEL.DIM_G, cfg.MODEL.DIM_G,
                            3, output, resample='up')
    output = normalize('generator.OutputN', output)
    output = tf.nn.relu(output)
    output = conv2D(
        'generator.Output', cfg.MODEL.DIM_G, 3, 3, output, he_init=False)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, cfg.DATA.OUTPUT_DIM])


def old_discriminator(inputs, cfg):
    output = tf.reshape(inputs, [-1, 3, 32, 32])
    output = optimized_res_block_disc1(output, cfg=cfg)
    output = residual_block('discriminator.2', cfg.MODEL.DIM_D, cfg.MODEL.DIM_D,
                            3, output, resample='down')
    output = residual_block('discriminator.3', cfg.MODEL.DIM_D, cfg.MODEL.DIM_D,
                            3, output, resample=None)
    output = residual_block('discriminator.4', cfg.MODEL.DIM_D, cfg.MODEL.DIM_D,
                            3, output, resample=None)
    output = tf.nn.relu(output)
    output = tf.reduce_mean(output, axis=[2, 3])
    output_wgan = linear(
        'discriminator.Output', cfg.MODEL.DIM_D, 1, output)
    output_wgan = tf.reshape(output_wgan, [-1])
    output_acgan = linear(
        'discriminator.ACGANOutput', cfg.MODEL.DIM_D, cfg.MODEL.HASH_DIM, output)
    output_acgan = tf.nn.tanh(output_acgan)
    return output_wgan, output_acgan


def good_generator(n_samples, labels, cfg, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    noise = tf.concat([tf.cast(labels, tf.float32), tf.slice(noise, [0, cfg.DATA.LABEL_DIM], [-1, -1])], 1)
    output = linear('generator.Input', 128, 4 * 4 * 8 * cfg.MODEL.DIM, noise)
    output = tf.reshape(output, [-1, 8 * cfg.MODEL.DIM, 4, 4])

    output = residual_block('generator.Res1', 8 * cfg.MODEL.DIM, 8 * cfg.MODEL.DIM, 3, output, resample='up')
    output = residual_block('generator.Res2', 8 * cfg.MODEL.DIM, 4 * cfg.MODEL.DIM, 3, output, resample='up')
    output = residual_block('generator.Res3', 4 * cfg.MODEL.DIM, 2 * cfg.MODEL.DIM, 3, output, resample='up')
    output = residual_block('generator.Res4', 2 * cfg.MODEL.DIM, 1 * cfg.MODEL.DIM, 3, output, resample='up')

    output = normalize('generator.OutputN', output)
    output = tf.nn.relu(output)
    output = conv2D('generator.Output', 1 * cfg.MODEL.DIM, 3, 3, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, cfg.DATA.OUTPUT_DIM])


def good_discriminator(inputs, cfg):
    output = tf.reshape(inputs, [-1, 3, 64, 64])
    output = conv2D(
        'discriminator.Input', 3, cfg.MODEL.DIM, 3, output, he_init=False)

    output = residual_block('discriminator.Res1', cfg.MODEL.DIM,
                            2 * cfg.MODEL.DIM, 3, output, resample='down')
    output = residual_block('discriminator.Res2', 2 * cfg.MODEL.DIM,
                            4 * cfg.MODEL.DIM, 3, output, resample='down')
    output = residual_block('discriminator.Res3', 4 * cfg.MODEL.DIM,
                            8 * cfg.MODEL.DIM, 3, output, resample='down')
    output = residual_block('discriminator.Res4', 8 * cfg.MODEL.DIM,
                            8 * cfg.MODEL.DIM, 3, output, resample='down')

    output = tf.reshape(output, [-1, 4 * 4 * 8 * cfg.MODEL.DIM])
    output_wgan = linear(
        'discriminator.Output', 4 * 4 * 8 * cfg.MODEL.DIM, 1, output)

    output_acgan = linear(
        'discriminator.ACGANOutput', 4 * 4 * 8 * cfg.MODEL.DIM, cfg.MODEL.HASH_DIM, output)
    output_acgan = tf.nn.tanh(output_acgan)
    return output_wgan, output_acgan


def alexnet_discriminator(inputs, cfg, stage="train"):
    # noinspection PyTypeChecker
    # TODO: don't load imagenet pretrained model when D_PRETRAINED_MODEL_PATH is given
    net_data = dict(np.load(cfg.MODEL.ALEXNET_PRETRAINED_MODEL_PATH, encoding='latin1').item())

    if inputs.shape[1] != 256:
        reshaped_image = preprocess_resize_scale_img(inputs, cfg.DATA.WIDTH_HEIGHT)
    else:
        reshaped_image = inputs

    height = width = 227

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
    mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32, shape=[
                       1, 1, 1, 3], name='img-mean')
    distorted_image = distorted_image - mean

    # Conv1
    # Output 96, kernel 11, stride 4
    scope = 'discriminator.conv1.'
    kernel = param(scope + 'weights', net_data['conv1'][0])
    biases = param(scope + 'biases', net_data['conv1'][1])
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
    if cfg.TRAIN.WGAN_SCALE == 0:
        lrn1 = tf.nn.local_response_normalization(pool1, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)
    else:
        lrn1 = pool1

    # Conv2
    # Output 256, pad 2, kernel 5, group 2
    scope = 'discriminator.conv2.'
    kernel = param(scope + 'weights', net_data['conv2'][0])
    biases = param(scope + 'biases', net_data['conv2'][1])
    group = 2

    def convolve(i, k): return tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
    input_groups = tf.split(lrn1, group, 3)
    kernel_groups = tf.split(kernel, group, 3)
    output_groups = [convolve(i, k)
                     for i, k in zip(input_groups, kernel_groups)]
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
    if cfg.TRAIN.WGAN_SCALE == 0:
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
    scope = 'discriminator.conv3.'
    kernel = param(scope + 'weights', net_data['conv3'][0])
    biases = param(scope + 'biases', net_data['conv3'][1])
    conv = tf.nn.conv2d(lrn2, kernel, [1, 1, 1, 1], padding='SAME')
    out = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(out, name=scope)

    # Conv4
    # Output 384, pad 1, kernel 3, group 2
    scope = 'discriminator.conv4.'
    kernel = param(scope + 'weights', net_data['conv4'][0])
    biases = param(scope + 'biases', net_data['conv4'][1])
    group = 2

    def convolve(i, k): return tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
    input_groups = tf.split(conv3, group, 3)
    kernel_groups = tf.split(kernel, group, 3)
    output_groups = [convolve(i, k)
                     for i, k in zip(input_groups, kernel_groups)]
    # Concatenate the groups
    conv = tf.concat(output_groups, 3)
    out = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(out, name=scope)

    # Conv5
    # Output 256, pad 1, kernel 3, group 2
    scope = 'discriminator.conv5.'
    kernel = param(scope + 'weights', net_data['conv5'][0])
    biases = param(scope + 'biases', net_data['conv5'][1])
    group = 2

    def convolve(i, k): return tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
    input_groups = tf.split(conv4, group, 3)
    kernel_groups = tf.split(kernel, group, 3)
    output_groups = [convolve(i, k)
                     for i, k in zip(input_groups, kernel_groups)]
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
    shape = int(np.prod(pool5.get_shape()[1:]))
    scope = 'discriminator.fc6.'
    fc6w = param(scope + 'weights', net_data['fc6'][0])
    fc6b = param(scope + 'biases', net_data['fc6'][1])
    pool5_flat = tf.reshape(pool5, [-1, shape])
    fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
    fc6 = tf.nn.dropout(tf.nn.relu(fc6l), 0.5)

    # FC7
    # Output 4096
    scope = 'discriminator.fc7.'
    fc7w = param(scope + 'weights', net_data['fc7'][0])
    fc7b = param(scope + 'biases', net_data['fc7'][1])
    fc7l = tf.nn.bias_add(tf.matmul(fc6, fc7w), fc7b)
    fc7 = tf.nn.dropout(tf.nn.relu(fc7l), 0.5)

    # FC8
    # Output output_dim
    fc8 = linear(
        'discriminator.ACGANOutput', 4096, cfg.MODEL.HASH_DIM, fc7)
    if stage == "train":
        output = tf.nn.tanh(fc8)
    else:
        fc8_t = tf.nn.tanh(fc8)
        fc8_t = tf.concat([tf.expand_dims(i, 0)
                           for i in tf.split(fc8_t, 10, 0)], 0)
        output = tf.reduce_mean(fc8_t, 0)
    output_wgan = linear('discriminator.Output', 4096, 1, fc7)

    return output_wgan, output


def generator(n_samples, labels, cfg, noise=None):
    if cfg.MODEL.G_ARCHITECTURE == "GOOD":
        return good_generator(n_samples, labels, noise=noise, cfg=cfg)
    else:
        return old_generator(n_samples, labels, noise=noise, cfg=cfg)


def discriminator(inputs, cfg, stage="train"):
    if cfg.MODEL.D_ARCHITECTURE == "GOOD":
        return good_discriminator(inputs, cfg=cfg)
    elif cfg.MODEL.D_ARCHITECTURE == "ALEXNET":
        return alexnet_discriminator(inputs, stage=stage, cfg=cfg)
    else:
        return old_discriminator(inputs, cfg=cfg)
