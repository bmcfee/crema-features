#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tests for model components'''

import numpy as np
import scipy
import tensorflow as tf
from nose.tools import eq_

import crema


def test_gmean():

    x = np.abs(np.random.randn(5, 5, 5), dtype=np.float32)

    def __test(axis, keep_dims):
        y_true = scipy.stats.gmean(x, axis=axis)

        x_in = tf.placeholder(tf.float32, shape=(5, 5, 5), name='x')

        outvar = crema.model.ops.reduce_gmean(x_in, reduction_indices=[axis],
                                                keep_dims=keep_dims)

        with tf.Session() as sess:
            y_pred = sess.run(outvar, feed_dict={x_in: x})

        if keep_dims:
            eq_(y_pred.ndim, x.ndim)
            y_pred = y_pred.squeeze()

        assert np.allclose(y_true, y_pred)

    for axis in [0, 1, 2]:
        for keep_dims in [False, True]:
            yield __test, axis, keep_dims


def __softmax(x, axes):

    x_max = x.max(axis=tuple(axes), keepdims=True)

    x_rebase = x - x_max
    ex = np.exp(x_rebase)

    return ex / ex.sum(axis=tuple(axes), keepdims=True)


def test_ndsoftmax():

    x = 100 * np.abs(np.random.randn(5, 5, 5), dtype=np.float32)

    def __test(axis):

        y_true = __softmax(x, axis)

        x_in = tf.placeholder(tf.float32, shape=(5, 5, 5), name='x')

        outvars = crema.model.ops.ndsoftmax(x_in, reduction_indices=axis)

        with tf.Session() as sess:
            y_logits = sess.run(outvars, feed_dict={x_in: x})
            y_pred = np.exp(y_logits)

        eq_(y_pred.shape, x.shape)

        assert np.allclose(y_true, y_pred)
        assert np.all(y_pred >= 0)
        assert np.all(np.isfinite(y_pred))
        assert np.allclose(y_pred.sum(tuple(axis)), 1.0)

    for axis in [[0], [1], [2], [1, 2], [0, 2]]:
        yield __test, axis


def __whiten(x):

    z = np.zeros_like(x)

    for i in range(x.shape[0]):
        z[i] = scipy.stats.zscore(x[i], axis=None)

    return z


def test_whiten():
    x = 100 * np.abs(np.random.randn(5, 5, 5), dtype=np.float32) + 30

    x_in = tf.placeholder(tf.float32, shape=x.shape, name='x')

    outvars = crema.model.ops.whiten(x_in, s_min=0.0)

    with tf.Session() as sess:
        y_pred = sess.run(outvars, feed_dict={x_in: x})

    y_true = __whiten(x)

    eq_(y_pred.shape, x.shape)
    assert np.allclose(y_true, y_pred, atol=1e-6)

