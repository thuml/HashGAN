from .tower_grad import average_gradients
import locale

import tensorflow as tf

locale.setlocale(locale.LC_ALL, '')

_params = {}
_param_aliases = {}


def preprocess_resize_scale_img(inputs, width_height):
    img = (inputs + 1.) * 255.99 / 2
    reshaped_image = tf.cast(img, tf.float32)
    reshaped_image = tf.reshape(
        reshaped_image, [-1, 3, width_height, width_height])

    transpose_image = tf.transpose(reshaped_image, perm=[0, 2, 3, 1])
    resized_image = tf.image.resize_bilinear(transpose_image, [256, 256])

    return resized_image


def param(name, *args, **kwargs):
    """
    A wrapper for `tf.Variable` which enables parameter sharing in models.

    Creates and returns theano shared variables similarly to `tf.Variable`, 
    except if you try to create a param with the same name as a 
    previously-created one, `param(...)` will just return the old one instead of 
    making a new one.

    This constructor also adds a `param` attribute to the shared variables it 
    creates, so that you can easily search a graph for all params.
    """

    if name not in _params:
        kwargs['name'] = name
        param = tf.Variable(*args, **kwargs)
        param.param = True
        _params[name] = param
    result = _params[name]
    i = 0
    while result in _param_aliases:
        # print 'following alias {}: {} to {}'.format(i, result, _param_aliases[result])
        i += 1
        result = _param_aliases[result]
    return result


def params_with_name(name):
    return [p for n, p in list(_params.items()) if name in n]
