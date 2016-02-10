#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Variable initializers'''

import tensorflow as tf
import numpy as np


def he_std(shape, sym):
    '''Helper function to compute the standard deviation for
    He-style initialized variables.

    Parameters
    ----------
    shape : tuple of ints
        The shape of the variable

    sym : bool
        Whether the nonlinearity is symmetric or not

    Returns
    -------
    sigma : float > 0
        The standard deviation for the parameter
    '''
    if sym:
        gain = 1.0
    else:
        gain = np.sqrt(2)

    sigma = np.prod(shape[:-1])**(-0.5)

    return gain * sigma


def he_normal(shape, name='weight', sym=False, dtype=tf.float32):
    '''He-initialized (normal) variable

    Parameters
    ----------
    shape : tuple of int
        Shape of the variable

    name : str
        Name of this variable

    sym : bool
        Whether the nonlinearity applied to this variable is symmetric
        (e.g., sigmoid) or asymmetric (e.g., relu)

    dtype : type
        Data type for this variable

    Returns
    -------
    var : tf.Variable
        The variable with initialization
    '''

    sigma = he_std(shape, sym)

    # This constant = sqrt(1 - 4 * pdf(2) / (cdf(2) - cdf(-2)))
    # computes the effective standard deviation of tensorflow's truncated normal
    # sampling
    trunc_scale = 0.87962566103423978

    initial = tf.truncated_normal(shape,
                                  mean=0.0,
                                  stddev=sigma / trunc_scale,
                                  dtype=dtype)

    return tf.Variable(initial, name=name)


def he_uniform(shape, name='weight', sym=False, dtype=tf.float32):
    '''He-initialized (uniform) variable

    Parameters
    ----------
    shape : tuple of int
        Shape of the variable

    name : str
        Name of this variable

    sym : bool
        Whether the nonlinearity applied to this variable is symmetric
        (e.g., sigmoid) or asymmetric (e.g., relu)

    dtype : type
        Data type for this variable

    Returns
    -------
    var : tf.Variable
        The variable with initialization
    '''

    sigma = np.sqrt(3) * he_std(shape, sym)

    initial = tf.random_uniform(shape, minval=-sigma, maxval=sigma, dtype=dtype)

    return tf.Variable(initial, name=name)


def he_uniform_tied(shape, name='weight', sym=False, dtype=tf.float32):
    '''He-initialized (uniform) variable

    Parameters
    ----------
    shape : tuple of int
        Shape of the variable

    name : str
        Name of this variable

    sym : bool
        Whether the nonlinearity applied to this variable is symmetric
        (e.g., sigmoid) or asymmetric (e.g., relu)

    dtype : type
        Data type for this variable

    Returns
    -------
    var : tf.Variable
        The variable with initialization
    '''

    sigma = np.sqrt(3) * he_std(shape, sym)

    initial_raw = tf.random_uniform([shape[0], shape[1], shape[3]], minval=-sigma, maxval=sigma, dtype=dtype)
    initial_full = tf.expand_dims(initial_raw, 2)
    initial = tf.tile(initial_full, [1, 1, shape[2], 1])
    return tf.Variable(initial, name=name)

def constant(shape, name='bias', default=0.0, dtype=tf.float32):
    '''Initialize with a constant

    Parameters
    ----------
    shape : tuple of int
        The shape of the variable

    name : str
        The name of the variable

    default : float or np.ndarray
        The default value to initialize with

    dtype : type
        The data type of the variable

    Returns
    -------
    var : tf.Variable
        The variable with initialization
    '''

    initial = tf.constant(default, shape=shape, dtype=dtype)
    return tf.Variable(initial, name=name)

