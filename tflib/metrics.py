import tensorflow as tf


def acc_multilabel(logits, labels):
    predict = tf.nn.sigmoid(logits) >= 0.5
    equal = tf.equal(tf.cast(predict, tf.int32), labels)
    return tf.reduce_mean(tf.cast(equal, tf.float32))
