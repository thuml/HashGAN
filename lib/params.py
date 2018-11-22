import locale

import tensorflow as tf

locale.setlocale(locale.LC_ALL, '')

_params = {}
_param_aliases = {}


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
        var = tf.Variable(*args, **kwargs)
        var.param = True
        _params[name] = var
    result = _params[name]
    i = 0
    while result in _param_aliases:
        # print 'following alias {}: {} to {}'.format(i, result, _param_aliases[result])
        i += 1
        result = _param_aliases[result]
    return result


def params_with_name(name):
    return [p for n, p in list(_params.items()) if name in n]


# compute param size
def print_param_size(gen_gv, disc_gv):
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
