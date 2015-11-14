#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Model utilities'''

import tensorflow as tf


def reduce_gmean(input_tensor, reduction_indices=None, keep_dims=False,
                 name=None):
    '''Geometric mean reduction.

    Parameters
    ----------
    input_tensor : tf.Tensor
        The tensor to reduce

    reduction_indices : int or list of int
        The axes along which to reduce

    keep_dims : bool
        Whether the reduced dimensions should be kept or not

    name : str (optional)
        Name for this operator

    Returns
    -------
    gmean_op : tf.Operator
        The geometric mean reduction operator

    '''
    with tf.name_scope(name):
        x_log = tf.log(input_tensor)

        output = tf.exp(tf.reduce_mean(x_log,
                                       reduction_indices=reduction_indices,
                                       keep_dims=keep_dims))

    return output
