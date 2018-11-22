#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf


def cross_entropy(u, label_u, v=None, label_v=None, alpha=1, partial=False, normed=False):
    if v is None:
        v = u
    else:
        # v is the fake data_list, which cannot influence real data_list
        if partial is True:
            u = tf.stop_gradient(u)

    if label_v is None:
        label_v = label_u
    label_ip = tf.cast(tf.matmul(label_u, tf.transpose(label_v)), tf.float32)
    s = tf.clip_by_value(label_ip, 0.0, 1.0)

    # compute balance param
    # s_t \in {-1, 1}
    s_t = tf.multiply(tf.add(s, tf.constant(-0.5)), tf.constant(2.0))
    sum_1 = tf.reduce_sum(s)
    sum_all = tf.reduce_sum(tf.abs(s_t))
    balance_param = tf.add(tf.abs(tf.add(s, tf.constant(-1.0))),
                           tf.multiply(tf.div(sum_all, sum_1), s))

    if normed:
        # ip = tf.clip_by_value(tf.matmul(u, tf.transpose(u)), -1.5e1, 1.5e1)
        ip_1 = tf.matmul(u, tf.transpose(v))

        def reduce_shaper(t):
            return tf.reshape(tf.reduce_sum(t, 1), [tf.shape(t)[0], 1])

        mod_1 = tf.sqrt(tf.matmul(reduce_shaper(tf.square(u)),
                                  reduce_shaper(tf.square(v)), transpose_b=True))
        ip = tf.div(ip_1, mod_1)
    else:
        ip = tf.clip_by_value(tf.matmul(u, tf.transpose(v)), -1.5e1, 1.5e1)
    ones = tf.ones([tf.shape(u)[0], tf.shape(v)[0]])
    return tf.reduce_mean(tf.multiply(tf.log(ones + tf.exp(alpha * ip)) - s * alpha * ip, balance_param))


if __name__ == "__main__":
    import numpy as np

    sess = tf.InteractiveSession()
    u_ = np.ones([2, 3], dtype=np.float32)
    label = np.ones([2, 2])
    print(("cross entropy loss 1 = %d" % cross_entropy(u_, label).eval()))

    label[1, :] = 0
    print(("cross entropy loss 2 = %d" % cross_entropy(u_, label).eval()))
