import numpy as np
import tensorflow as tf
from lib.params import param


# noinspection PyUnboundLocalVariable,PyPep8Naming
def conv2D(
        name,
        input_dim,
        output_dim,
        filter_size,
        inputs,
        he_init=True,
        mask_type=None,
        stride=1,
        weightnorm=None,
        biases=True,
        gain=1.):
    """
    inputs: tensor of shape (batch size, num channels, height, width)
    mask_type: one of None, 'a', 'b'

    returns: tensor of shape (batch size, num channels, height, width)
    """
    if mask_type is not None:
        mask_type, mask_n_channels = mask_type

        mask = np.ones(
            (filter_size, filter_size, input_dim, output_dim),
            dtype='float32'
        )
        center = filter_size // 2

        # Mask out future locations
        # filter shape is (height, width, input channels, output channels)
        mask[center + 1:, :, :, :] = 0.
        mask[center, center + 1:, :, :] = 0.

        # Mask out future channels
        for i in range(mask_n_channels):
            for j in range(mask_n_channels):
                if (mask_type == 'a' and i >= j) or (mask_type == 'b' and i > j):
                    mask[center, center, i::mask_n_channels, j::mask_n_channels] = 0.

    def uniform(stdev, size):
        return np.random.uniform(
            low=-stdev * np.sqrt(3),
            high=stdev * np.sqrt(3),
            size=size
        ).astype('float32')

    fan_in = input_dim * filter_size ** 2
    fan_out = output_dim * filter_size ** 2 / (stride ** 2)

    if mask_type is not None:  # only approximately correct
        fan_in /= 2.
        fan_out /= 2.

    if he_init:
        filters_stdev = np.sqrt(4. / (fan_in + fan_out))
    else:  # Normalized init (Glorot & Bengio)
        filters_stdev = np.sqrt(2. / (fan_in + fan_out))

    filter_values = uniform(
        filters_stdev,
        (filter_size, filter_size, input_dim, output_dim)
    )

    # print "WARNING IGNORING GAIN"
    filter_values *= gain

    filters = param(name + '.Filters', filter_values)

    if weightnorm:
        norm_values = np.sqrt(
            np.sum(np.square(filter_values), axis=(0, 1, 2)))
        target_norms = param(
            name + '.g',
            norm_values
        )
        with tf.name_scope('weightnorm'):
            norms = tf.sqrt(tf.reduce_sum(
                tf.square(filters), reduction_indices=[0, 1, 2]))
            filters = filters * (target_norms / norms)

    if mask_type is not None:
        with tf.name_scope('filter_mask'):
            filters = filters * mask

    result = tf.nn.conv2d(
        input=inputs,
        filter=filters,
        strides=[1, 1, stride, stride],
        padding='SAME',
        data_format='NCHW'
    )

    if biases:
        _biases = param(
            name + '.Biases',
            np.zeros(output_dim, dtype='float32')
        )

        result = tf.nn.bias_add(result, _biases, data_format='NCHW')

    return result


def batch_norm(name, axes, inputs, is_training=None, stats_iter=None, update_moving_stats=True, fused=True):
    if ((axes == [0, 2, 3]) or (axes == [0, 2])) and fused is True:
        if axes == [0, 2]:
            inputs = tf.expand_dims(inputs, 3)

        offset = param(
            name + '.offset', np.zeros(inputs.get_shape()[1], dtype='float32'))
        scale = param(
            name + '.scale', np.ones(inputs.get_shape()[1], dtype='float32'))

        moving_mean = param(
            name + '.moving_mean', np.zeros(inputs.get_shape()[1], dtype='float32'), trainable=False)
        moving_variance = param(
            name + '.moving_variance', np.ones(inputs.get_shape()[1], dtype='float32'), trainable=False)

        def _fused_batch_norm_training():
            return tf.nn.fused_batch_norm(inputs, scale, offset, epsilon=1e-5, data_format='NCHW')

        def _fused_batch_norm_inference():
            # Version which blends in the current item's statistics
            batch_size = tf.cast(tf.shape(inputs)[0], 'float32')
            mean_, var_ = tf.nn.moments(inputs, [2, 3], keep_dims=True)
            mean_ = ((1. / batch_size) * mean_) + (((batch_size - 1.) /
                                                    batch_size) * moving_mean)[None, :, None, None]
            var_ = ((1. / batch_size) * var_) + (((batch_size - 1.) / batch_size)
                                                 * moving_variance)[None, :, None, None]
            bn = tf.nn.batch_normalization(inputs, mean_, var_,
                                           offset[None, :, None, None],
                                           scale[None, :, None, None],
                                           1e-5)
            return bn, mean_, var_

        if is_training is None:
            outputs, batch_mean, batch_var = _fused_batch_norm_training()
        else:
            outputs, batch_mean, batch_var = tf.cond(is_training,
                                                     _fused_batch_norm_training,
                                                     _fused_batch_norm_inference)
            if update_moving_stats:
                def no_updates(): return outputs

                def _force_updates():
                    """Internal function forces updates moving_vars if is_training."""
                    float_stats_iter = tf.cast(stats_iter, tf.float32)

                    alpha = float_stats_iter / (float_stats_iter + 1)
                    update_moving_mean = tf.assign(moving_mean, (alpha * moving_mean) + (1 - alpha) * batch_mean)
                    update_moving_variance = tf.assign(moving_variance,
                                                       (alpha * moving_variance) + (1 - alpha) * batch_var)

                    with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                        return tf.identity(outputs)

                outputs = tf.cond(is_training, _force_updates, no_updates)

        if axes == [0, 2]:
            return outputs[:, :, :, 0]  # collapse last dim
        else:
            return outputs
    else:
        mean, var = tf.nn.moments(inputs, axes, keep_dims=True)
        shape = mean.get_shape().as_list()
        if 0 not in axes:
            print("WARNING ({}): didn't find 0 in axes, but not using separate BN params for each item in batch"
                  .format(name))
            shape[0] = 1
        offset = param(name + '.offset', np.zeros(shape, dtype='float32'))
        scale = param(name + '.scale', np.ones(shape, dtype='float32'))
        result = tf.nn.batch_normalization(
            inputs, mean, var, offset, scale, 1e-5)

        return result


def linear(
        name,
        input_dim,
        output_dim,
        inputs,
        biases=True,
        initialization=None,
        weightnorm=None,
        gain=1.
):
    """
    initialization: None, `lecun`, 'glorot', `he`, 'glorot_he', `orthogonal`, `("uniform", range)`
    """

    def uniform(stdev, size):
        return np.random.uniform(
            low=-stdev * np.sqrt(3),
            high=stdev * np.sqrt(3),
            size=size
        ).astype('float32')

    if initialization == 'lecun':  # and input_dim != output_dim):
        # disabling orth. init for now because it's too slow
        weight_values = uniform(
            np.sqrt(1. / input_dim),
            (input_dim, output_dim)
        )

    elif initialization == 'glorot' or (initialization is None):

        weight_values = uniform(
            np.sqrt(2. / (input_dim + output_dim)),
            (input_dim, output_dim)
        )

    elif initialization == 'he':

        weight_values = uniform(
            np.sqrt(2. / input_dim),
            (input_dim, output_dim)
        )

    elif initialization == 'glorot_he':

        weight_values = uniform(
            np.sqrt(4. / (input_dim + output_dim)),
            (input_dim, output_dim)
        )

    elif initialization == 'orthogonal' or \
            (initialization is None and input_dim == output_dim):

        # From lasagne
        def sample(shape):
            if len(shape) < 2:
                raise RuntimeError("Only shapes of length 2 or more are "
                                   "supported.")
            flat_shape = (shape[0], np.prod(shape[1:]))
            a = np.random.normal(0.0, 1.0, flat_shape)
            u, _, v = np.linalg.svd(a, full_matrices=False)
            # pick the one with the correct shape
            q = u if u.shape == flat_shape else v
            q = q.reshape(shape)
            return q.astype('float32')

        weight_values = sample((input_dim, output_dim))

    elif initialization[0] == 'uniform':

        weight_values = np.random.uniform(
            low=-initialization[1],
            high=initialization[1],
            size=(input_dim, output_dim)
        ).astype('float32')

    else:

        raise Exception('Invalid initialization!')

    weight_values *= gain

    weight = param(
        name + '.W',
        weight_values
    )

    if weightnorm:
        norm_values = np.sqrt(np.sum(np.square(weight_values), axis=0))
        # norm_values = np.linalg.norm(weight_values, axis=0)

        target_norms = param(
            name + '.g',
            norm_values
        )

        with tf.name_scope('weightnorm'):
            norms = tf.sqrt(tf.reduce_sum(
                tf.square(weight), reduction_indices=[0]))
            weight = weight * (target_norms / norms)

    # if 'discriminator' in name:
    #     print "WARNING weight constraint on {}".format(name)
    #     weight = tf.nn.softsign(10.*weight)*.1

    if inputs.get_shape().ndims == 2:
        result = tf.matmul(inputs, weight)
    else:
        reshaped_inputs = tf.reshape(inputs, [-1, input_dim])
        result = tf.matmul(reshaped_inputs, weight)
        result = tf.reshape(result, tf.stack(
            tf.unstack(tf.shape(inputs))[:-1] + [output_dim]))

    if biases:
        result = tf.nn.bias_add(
            result,
            param(
                name + '.b',
                np.zeros((output_dim,), dtype='float32')
            )
        )

    return result
