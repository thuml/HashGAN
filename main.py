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

import argparse
import locale
import os
import sys
import time
from pprint import pprint

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

import dataloader
import tflib as lib
import tflib.plot
import tflib.save_images
import util
from tflib import preprocess_resize_scale_img
from tflib import average_gradients
from tflib.losses import cross_entropy
from tflib.architecture import Generator, Discriminator
from config import config


def main(cfg):
    dataset = dataloader.__dict__[cfg.DATA.USE_DATASET]
    DEVICES = [x.name for x in device_lib.list_local_devices()
               if x.device_type == 'GPU']

    configProto = tf.ConfigProto()
    configProto.gpu_options.allow_growth = True
    configProto.allow_soft_placement = True
    with tf.Session(config=configProto) as session:

        _iteration = tf.placeholder(tf.int32, shape=None)

        # unlabeled data initialization
        all_unlabel_data_int = tf.placeholder(
            tf.int32, shape=[cfg.TRAIN.BATCH_SIZE, cfg.DATA.OUTPUT_DIM])
        all_unlabel_labels = tf.placeholder(
            tf.int32, shape=[cfg.TRAIN.BATCH_SIZE, cfg.DATA.LABEL_DIM])
        unlabel_labels_splits = tf.split(
            all_unlabel_labels, len(DEVICES), axis=0)

        unlabel_fake_data_splits = []
        for i, device in enumerate(DEVICES):
            with tf.device(device):
                unlabel_fake_data_splits.append(
                    Generator(cfg.TRAIN.BATCH_SIZE // len(DEVICES), unlabel_labels_splits[i], cfg=cfg))

        all_unlabel_data = tf.reshape(2 * ((tf.cast(all_unlabel_data_int, tf.float32) / 256.) - .5),
                                      [cfg.TRAIN.BATCH_SIZE, cfg.DATA.OUTPUT_DIM])
        all_unlabel_data += tf.random_uniform(
            shape=[cfg.TRAIN.BATCH_SIZE, cfg.DATA.OUTPUT_DIM], minval=0., maxval=1. / 128)  # dequantize
        all_unlabel_data_splits = tf.split(
            all_unlabel_data, len(DEVICES), axis=0)

        # labeled data init
        all_real_data_int = tf.placeholder(
            tf.int32, shape=[cfg.TRAIN.BATCH_SIZE, cfg.DATA.OUTPUT_DIM])
        all_real_labels = tf.placeholder(
            tf.int32, shape=[cfg.TRAIN.BATCH_SIZE, cfg.DATA.LABEL_DIM])
        labels_splits = tf.split(all_real_labels, len(DEVICES), axis=0)

        fake_data_splits = []
        for i, device in enumerate(DEVICES):
            with tf.device(device):
                fake_data_splits.append(
                    Generator(cfg.TRAIN.BATCH_SIZE // len(DEVICES), labels_splits[i], cfg=cfg))

        all_real_data = tf.reshape(
            2 * ((tf.cast(all_real_data_int, tf.float32) / 256.) - .5), [cfg.TRAIN.BATCH_SIZE, cfg.DATA.OUTPUT_DIM])
        # dequantize
        all_real_data += tf.random_uniform(
            shape=[cfg.TRAIN.BATCH_SIZE, cfg.DATA.OUTPUT_DIM], minval=0., maxval=1. / 128)
        all_real_data_splits = tf.split(all_real_data, len(DEVICES), axis=0)

        # init optimizer
        if cfg.TRAIN.DECAY:
            decay = tf.maximum(
                0., 1. - (tf.cast(_iteration, tf.float32) / cfg.TRAIN.ITERS))
        else:
            decay = 1.0
        # TODO
        # if config.MODEL.ARCHITECTURE == "ALEXNET":
        #   disc_opt = tf.train.MomentumOptimizer(learning_rate=LR*decay, momentum=0.9)
        # else:
        disc_opt = tf.train.AdamOptimizer(
            learning_rate=cfg.TRAIN.LR * decay, beta1=0., beta2=0.9)
        gen_opt = tf.train.AdamOptimizer(
            learning_rate=cfg.TRAIN.G_LR * decay, beta1=0., beta2=0.9)

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
                disc_all, disc_all_acgan = Discriminator(
                    real_and_fake_data, cfg=cfg)
                # size * 2 for unlabeled data
                disc_real = disc_all[:cfg.TRAIN.BATCH_SIZE // len(DEVICES) * 2]
                disc_fake = disc_all[cfg.TRAIN.BATCH_SIZE // len(DEVICES) * 2:]
                disc_costs.append(tf.reduce_mean(
                    disc_fake) - tf.reduce_mean(disc_real))
                disc_costs_wgan.append(tf.reduce_mean(
                    disc_fake) - tf.reduce_mean(disc_real))

                # gradients computation
                disc_costs_gs.append(disc_opt.compute_gradients((tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)),
                                                                var_list=lib.params_with_name('Discriminator.')))

                pos_start = cfg.TRAIN.BATCH_SIZE // len(DEVICES)
                pos_middle = 2 * cfg.TRAIN.BATCH_SIZE // len(DEVICES)
                pos_end = 3 * cfg.TRAIN.BATCH_SIZE // len(DEVICES)

                var_list = lib.params_with_name('Discriminator.')

                # real vs real
                cost_rr = cross_entropy(
                    disc_all_acgan[pos_start:pos_middle],
                    real_and_fake_labels[pos_start:pos_middle],
                    alpha=cfg.TRAIN.CROSS_ENTROPY_ALPHA,
                    normed=cfg.TRAIN.NORMED_CROSS_ENTROPY)
                disc_acgan_costs.append(cost_rr)
                # gradients computation
                disc_acgan_costs_gs.append(
                    disc_opt.compute_gradients(cost_rr, var_list=var_list))
                # real vs fake, fake cannot influence real
                if cfg.TRAIN.FAKE_RATIO != 0.0:
                    cost_fr = cfg.TRAIN.FAKE_RATIO * cross_entropy(
                        disc_all_acgan[pos_start:pos_middle],
                        real_and_fake_labels[pos_start:pos_middle],
                        disc_all_acgan[pos_middle:pos_end],
                        real_and_fake_labels[pos_middle:pos_end], alpha=cfg.TRAIN.CROSS_ENTROPY_ALPHA, partial=True,
                        normed=cfg.TRAIN.NORMED_CROSS_ENTROPY)
                    disc_acgan_costs.append(cost_fr)
                    disc_acgan_costs_gs.append(
                        disc_opt.compute_gradients(cost_fr, var_list=var_list))

                disc_acgan_costs_real_real.append(cross_entropy(
                    disc_all_acgan[pos_start:pos_middle],
                    real_and_fake_labels[pos_start:pos_middle],
                    alpha=cfg.TRAIN.CROSS_ENTROPY_ALPHA,
                    normed=cfg.TRAIN.NORMED_CROSS_ENTROPY))

        if cfg.TRAIN.WGAN_SCALE != 0:
            for i, device in enumerate(DEVICES):
                with tf.device(device):
                    real_data = tf.concat(
                        [all_unlabel_data_splits[i], all_real_data_splits[i]], axis=0)
                    fake_data = tf.concat(
                        [fake_data_splits[i], unlabel_fake_data_splits[i]], axis=0)
                    alpha = tf.random_uniform(
                        shape=[2 * cfg.TRAIN.BATCH_SIZE // len(DEVICES), 1],
                        minval=0.,
                        maxval=1.
                    )
                    if cfg.MODEL.ARCHITECTURE == "ALEXNET":
                        real_data = preprocess_resize_scale_img(real_data, WIDTH_HEIGHT=cfg.DATA.WIDTH_HEIGHT)
                        fake_data = preprocess_resize_scale_img(fake_data, WIDTH_HEIGHT=cfg.DATA.WIDTH_HEIGHT)
                        alpha = tf.random_uniform(
                            shape=[2 * cfg.TRAIN.BATCH_SIZE // len(DEVICES), 1, 1, 1],
                            minval=0.,
                            maxval=1.
                        )

                    differences = fake_data - real_data
                    interpolates = real_data + (alpha * differences)
                    gradients = tf.gradients(Discriminator(interpolates, cfg=cfg)[0], [interpolates])[0]
                    if cfg.MODEL.ARCHITECTURE == "ALEXNET":
                        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
                    else:
                        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                    gradient_penalty = 10 * tf.reduce_mean((slopes - 1.) ** 2)
                    disc_costs.append(gradient_penalty)
                    disc_costs_gradient_penalty.append(gradient_penalty)

                    # gradients computation
                    disc_costs_gs.append(
                        disc_opt.compute_gradients(gradient_penalty, var_list=lib.params_with_name('Discriminator.')))

            disc_wgan_gradient = tf.add_n(
                disc_costs_gradient_penalty) / len(DEVICES)

        disc_wgan = tf.add_n(disc_costs) / len(DEVICES)
        disc_wgan_l = tf.add_n(disc_costs_wgan) / len(DEVICES)

        disc_acgan = tf.add_n(disc_acgan_costs) / len(DEVICES)
        disc_acgan_real_real = tf.add_n(
            disc_acgan_costs_real_real) / len(DEVICES)

        disc_cost = cfg.TRAIN.ACGAN_SCALE * disc_acgan

        disc_gv = average_gradients(disc_acgan_costs_gs, cfg.TRAIN.ACGAN_SCALE)
        if cfg.TRAIN.WGAN_SCALE != 0:
            disc_cost = disc_cost + cfg.TRAIN.WGAN_SCALE * disc_wgan
            disc_gv = disc_gv + average_gradients(disc_costs_gs, cfg.TRAIN.WGAN_SCALE)

        gen_costs = []
        gen_acgan_costs = []

        gen_costs_gs = []
        gen_acgan_costs_gs = []

        def to_one_hot(sparse_labels):
            return tf.one_hot(sparse_labels, cfg.DATA.LABEL_DIM, dtype=tf.int32)

        # for device in DEVICES:
        for i, device in enumerate(DEVICES):
            with tf.device(device):
                n_samples = cfg.TRAIN.BATCH_SIZE // len(DEVICES)
                fake_data = Generator(n_samples, labels_splits[i], cfg=cfg)
                real_and_fake_data = tf.concat([
                    all_real_data_splits[i],
                    fake_data
                ], axis=0)
                real_and_fake_labels = tf.concat([
                    labels_splits[i],
                    labels_splits[i],
                ], axis=0)
                disc_all, disc_all_acgan = Discriminator(
                    real_and_fake_data, cfg=cfg)
                disc_fake = disc_all[n_samples:]
                gen_costs.append(-tf.reduce_mean(disc_fake))
                gen_acgan_costs.append(cross_entropy(
                    disc_all_acgan[:n_samples],
                    real_and_fake_labels[:n_samples],
                    disc_all_acgan[n_samples:],
                    real_and_fake_labels[n_samples:],
                    alpha=cfg.TRAIN.CROSS_ENTROPY_ALPHA, partial=True,
                    normed=cfg.TRAIN.NORMED_CROSS_ENTROPY))
                gen_costs_gs.append(gen_opt.compute_gradients(-tf.reduce_mean(disc_fake),
                                                              var_list=lib.params_with_name('Generator')))
                gen_acgan_costs_gs.append(gen_opt.compute_gradients(cross_entropy(
                    disc_all_acgan[:n_samples],
                    real_and_fake_labels[:n_samples],
                    disc_all_acgan[n_samples:],
                    real_and_fake_labels[n_samples:],
                    alpha=cfg.TRAIN.CROSS_ENTROPY_ALPHA, partial=True,
                    normed=cfg.TRAIN.NORMED_CROSS_ENTROPY), var_list=lib.params_with_name('Generator')))

        # set acgan_output
        disc_real_acgan = []
        disc_real_acgan_cost_t = []
        for i, device in enumerate(DEVICES):
            with tf.device(device):
                real_data = all_real_data_splits[i]
                real_labels = labels_splits[i]

                _, _disc_real_acgan = Discriminator(
                    real_data, stage="val", cfg=cfg)
                disc_real_acgan.append(_disc_real_acgan)

                disc_real_acgan_cost_t.append(cross_entropy(
                    _disc_real_acgan,
                    real_labels,
                    alpha=cfg.TRAIN.CROSS_ENTROPY_ALPHA,
                    normed=cfg.TRAIN.NORMED_CROSS_ENTROPY))
        disc_real_acgan_cost = tf.add_n(disc_real_acgan_cost_t) / len(DEVICES)

        gen_cost = cfg.TRAIN.WGAN_SCALE_G * (tf.add_n(gen_costs) / len(DEVICES))
        gen_gv = average_gradients(gen_costs_gs, cfg.TRAIN.WGAN_SCALE_G)
        gen_cost += (cfg.TRAIN.ACGAN_SCALE_G *
                     (tf.add_n(gen_acgan_costs) / len(DEVICES)))
        gen_gv = gen_gv + average_gradients(gen_acgan_costs_gs, cfg.TRAIN.ACGAN_SCALE_G)

        gen_train_op = gen_opt.apply_gradients(gen_gv)
        disc_train_op = disc_opt.apply_gradients(disc_gv)

        # Function for generating samples
        noise_dim = 256 if cfg.DATA.USE_DATASET == "cifar10" else 128  # TODO: refactor
        fixed_noise = tf.constant(np.random.normal(
            size=(100, noise_dim)).astype('float32'))
        fixed_labels = to_one_hot(tf.constant(
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10, dtype='int32')))
        fixed_noise_samples = Generator(100, fixed_labels, noise=fixed_noise, cfg=cfg)

        def generate_image(frame):
            samples = session.run(fixed_noise_samples)
            samples = ((samples + 1.) * (255. / 2)).astype('int32')
            lib.save_images.save_images(samples.reshape((100, 3, cfg.DATA.WIDTH_HEIGHT, cfg.DATA.WIDTH_HEIGHT)),
                                        '{}/samples_{}.png'.format(cfg.DATA.IMAGE_DIR, frame))

        train_gen, unlabel_train_gen, dev_gen = dataset.load(cfg.TRAIN.BATCH_SIZE, cfg.DATA.WIDTH_HEIGHT)
        gen = util.inf_gen(train_gen)
        unlabel_gen = util.inf_gen(unlabel_train_gen)

        util.print_param_size(gen_gv, disc_gv)

        print("initializing global variables")
        session.run(tf.global_variables_initializer())


        if cfg.TRAIN.USE_PRETRAIN:
            saver = tf.train.Saver(lib.params_with_name('Generator'))
            saver.restore(session, cfg.MODEL.PRETRAINED_MODEL_PATH)
            print("model restored")

        print("training")
        for iteration in range(cfg.TRAIN.ITERS):
            start_time = time.time()

            if iteration > 0:
                if cfg.TRAIN.G_LR != 0:
                    _data, _labels = gen()
                    _ = session.run([gen_train_op], feed_dict={
                        all_real_data_int: _data,
                        all_real_labels: _labels,
                        _iteration: iteration,
                    })

            for i in range(cfg.TRAIN.N_CRITIC):
                _data, _labels = gen()
                _unlabel_data, _unlabel_labels = unlabel_gen()
                if cfg.TRAIN.WGAN_SCALE == 0:
                    _disc_cost, _disc_acgan, _disc_acgan_r_r, _ = session.run(
                        [disc_cost, disc_acgan, disc_acgan_real_real,
                            disc_train_op],
                        feed_dict={
                            all_real_data_int: _data,
                            all_real_labels: _labels,
                            all_unlabel_data_int: _unlabel_data,
                            all_unlabel_labels: _unlabel_labels,
                            _iteration: iteration,
                        })
                else:
                    _disc_cost, _disc_wgan, _disc_wgan_l, _disc_wgan_g, _disc_acgan, _disc_acgan_r_r, _ = session.run(
                        [disc_cost, disc_wgan, disc_wgan_l, disc_wgan_gradient, disc_acgan, disc_acgan_real_real,
                        disc_train_op],
                        feed_dict={
                            all_real_data_int: _data,
                            all_real_labels: _labels,
                            all_unlabel_data_int: _unlabel_data,
                            all_unlabel_labels: _unlabel_labels,
                            _iteration: iteration,
                        })

            lib.plot.plot('cost', _disc_cost)
            lib.plot.plot('acgan_f', _disc_acgan - _disc_acgan_r_r)
            lib.plot.plot('acgan_r', _disc_acgan_r_r)

            if cfg.TRAIN.WGAN_SCALE != 0:
                lib.plot.plot('wgan', _disc_wgan)
                lib.plot.plot('wgan_l', _disc_wgan_l)
                lib.plot.plot('wgan_g', _disc_wgan_g)
            lib.plot.plot('time', time.time() - start_time)

            if (iteration + 1) % 1000 == 0:
                generate_image(iteration)

            # calculate mAP score w.r.t all db data every 10000 config.TRAIN.ITERS
            if (iteration + 1) % 10000 == 0:
                _db_gen, _test_gen = dataset.load_val(cfg.TRAIN.BATCH_SIZE, cfg.DATA.WIDTH_HEIGHT)
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
                db.output = np.reshape(
                    np.array(db_output), [-1, cfg.MODEL.HASH_DIM])[:cfg.DATA.DB_SIZE, :]
                db.label = np.reshape(
                    np.array(db_labels), [-1, cfg.DATA.LABEL_DIM])[:cfg.DATA.DB_SIZE, :]
                test = argparse.Namespace()
                test.output = np.reshape(
                    np.array(test_output), [-1, cfg.MODEL.HASH_DIM])[:cfg.DATA.TEST_SIZE, :]
                test.label = np.reshape(
                    np.array(test_labels), [-1, cfg.DATA.LABEL_DIM])[:cfg.DATA.TEST_SIZE, :]

                mAP_ = util.MAPs(cfg.DATA.MAP_R)
                mAP_val = mAP_.get_mAPs_by_feature(db, test)
                lib.plot.plot("mAP_feature", mAP_val)

            if (iteration < 500) or (iteration % 1000 == 999):
                lib.plot.flush(cfg.DATA.IMAGE_DIR)

            if (iteration + 1) % cfg.TRAIN.SAVE_FREQUENCY == 0 or iteration + 1 == cfg.TRAIN.ITERS:
                save_path = os.path.join(
                    cfg.DATA.MODEL_DIR, "iteration_{}.ckpt".format(iteration))
                saver = tf.train.Saver()
                saver.save(session, save_path)
                print(("Model saved in file: %s" % save_path))

            lib.plot.tick()


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    locale.setlocale(locale.LC_ALL, '')

    parser = argparse.ArgumentParser(description='HashGAN')
    parser.add_argument('--cfg', '--config', required=True,
                        type=str, metavar="FILE", help="path to yaml config")
    parser.add_argument('--gpus', default='0', type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    config.merge_from_file(args.cfg)
    config.freeze()
    pprint(config)

    os.makedirs(config.DATA.IMAGE_DIR, exist_ok=True)
    os.makedirs(config.DATA.MODEL_DIR, exist_ok=True)
    os.makedirs(config.DATA.LOG_DIR, exist_ok=True)

    main(config)
