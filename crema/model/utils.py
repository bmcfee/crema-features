#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Model utilities'''

import tensorflow as tf


def reduce_gmean(input_tensor, reduction_indices=None, keep_dims=False):
    '''Geometric mean reduction.

    Parameters
    ----------
    input_tensor : tf.Tensor
        The tensor to reduce

    reduction_indices : int or list of int
        The axes along which to reduce

    keep_dims : bool
        Whether the reduced dimensions should be kept or not

    Returns
    -------
    gmean_op : tf.Operator
        The geometric mean reduction operator

    '''
    with tf.name_scope('gmean'):
        x_log = tf.log(input_tensor)

        output = tf.exp(tf.reduce_mean(x_log,
                                       reduction_indices=reduction_indices,
                                       keep_dims=keep_dims))

    return output


def ndsoftmax(input_tensor, reduction_indices):
    '''n-dimensional log soft-max

    Parameters
    ----------
    input_tensor : tf.Tensor
        The tensor to re-scale

    reduction_indices : int or list of int
        The axes along which to normalize

    Returns
    -------
    softmax_op : tf.Operator
        The softmax operator

    logit_op : tf.Operator
        The corresponding logits
    '''
    with tf.name_scope('ndsoftmax'):
        x_max = tf.reduce_max(input_tensor,
                              reduction_indices=reduction_indices,
                              keep_dims=True)

        x_rebase = input_tensor - x_max

        ex = tf.exp(x_rebase)

        logits = x_rebase - tf.log(tf.reduce_sum(ex,
                                                 reduction_indices=reduction_indices,
                                                 keep_dims=True))
        softmax = tf.exp(logits)

    return softmax, logits

