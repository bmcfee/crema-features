#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Model utilities'''

import tensorflow as tf
from . import init


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
                                       keep_dims=keep_dims),
                        name='activation')

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
    logit_op : tf.Operator
        The corresponding logits (log of softmax)
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

    return logits


def whiten(input_tensor, s_min=1e-5, name=None):
    '''Per-sample whitening:
        input_tensor[i] -> zscore(input_tensor[i])

    Parameters
    ----------
    input_tensor : tf.Tensor
        The tensor to whiten

    s_min : float > 0
        Clip standard deviation

    name : str [optional]
        Name scope for the whitening operator

    Returns
    -------
    whitened_tensor : tf.Operator
        Operator which computes the locally whitened input tensor
    '''
    ndim = len(input_tensor.get_shape())
    reduction_idx = list(range(1, ndim))

    with tf.name_scope(name):
        mean = tf.reduce_mean(input_tensor,
                              reduction_indices=reduction_idx,
                              keep_dims=True)

        centered = input_tensor - mean

        istd = tf.rsqrt(tf.reduce_mean(tf.square(centered),
                                       reduction_indices=reduction_idx,
                                       keep_dims=True))
        zscored = tf.mul(centered, tf.maximum(s_min, istd), name='activation')

    return zscored


def gain(input_tensor, default=10.0, name=None):
    '''Band-parametric mu-law scaling.

    https://en.wikipedia.org/wiki/%CE%9C-law_algorithm

        output[f] = log(1 + W[f] * input[f]) / log(1 + W[f])

    where ``f`` denotes a frequency band, and ``W`` denotes a (non-negative)
    scaling coefficient.

    Parameters
    ----------
    input_tensor : tf.Operator
        The tensor to scale

    default : float > 0
        The initial value for scaling

    name : str
        The name of this operator node


    Returns
    -------
    scaler : tf.Operator
        The scaling operator
    '''
    shape = input_tensor.get_shape()

    with tf.name_scope(name):
        weight = init.constant([1, 1, int(shape[2]), int(shape[3])],
                               name='weight',
                               default=default)

        x_pow = tf.abs(input_tensor)

        w_mag = tf.nn.softplus(weight)

        output = tf.div(tf.log(1.0 + tf.mul(x_pow, w_mag)),
                        tf.log(1.0 + w_mag),
                        name='activation')

    return output


def logsigmoid(input_tensor, name='logsigmoid'):
    '''Compute the log-sigmoid of an input tensor:

        ``f(x) = - log(1 + exp(-x))``

    Parameters
    ----------
    input_tensor : tf.Tensor
        The input tensor to scale

    Returns
    -------
    logsigmoid : tf.Operator
        The log-sigmoid operator
    '''
    with tf.name_scope(name):
        output = tf.neg(tf.nn.softplus(-input_tensor))
    return output


def ndxent(logits, labels, reduction_indices, name='ndxent'):
    '''N-dimensional cross-entropy

    Parameters
    ----------
    logits : tf.Tensor
        The input logits (log probabilities)

    labels : tf.Tesnor
        The target labels

    reduction_indices : list of int
        Axes along which to reduce the predictions

    name : str
        Name for this node

    Returns
    -------
    conv_softmax : tf.Operator
        The convolutional softmax operator
    '''
    with tf.name_scope(name):
        output = -tf.reduce_sum(tf.mul(logits, labels), reduction_indices)

    return output


def expand_mask(mask, name='expand_mask'):
    '''Expand the binary mask for an observation.

    Parameters
    ----------
    mask : tf.Tensor [shape=(n,), dtype=bool]
        Input binary mask

    Returns
    -------
    expand : tf.Operator
        Output binary mask.  Will have shape (n, 1, 1) and dtype=float.
    '''
    with tf.name_scope(name):
        f_mask = tf.to_float(mask)
        z_mask = tf.reshape(f_mask, (-1, 1, 1))

    return z_mask
