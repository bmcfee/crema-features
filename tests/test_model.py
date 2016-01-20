#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tests for model components'''

import numpy as np
import scipy
import scipy.stats
import tensorflow as tf
from nose.tools import eq_

from test_layers import new_graph
import crema



def test_gmean():


    @new_graph
    def __test(axis, keep_dims):
        x = np.abs(np.random.randn(5, 5, 5), dtype=np.float32)
        y_true = scipy.stats.gmean(x, axis=axis)

        x_in = tf.placeholder(tf.float32, shape=x.shape, name='x')

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


    @new_graph
    def __test(axis):

        x = 100 * np.abs(np.random.randn(5, 5, 5), dtype=np.float32)
        y_true = __softmax(x, axis)

        x_in = tf.placeholder(tf.float32, shape=x.shape, name='x')

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


def __whiten(x, idx, s_min=1e-10):

    mu = np.mean(x, axis=idx, keepdims=True)
    sig = np.std(x, axis=idx, keepdims=True)

    return (x - mu) / np.maximum(sig, s_min)


def test_whiten():
    x = 100 * np.abs(np.random.randn(5, 6, 7, 8), dtype=np.float32) + 30

    @new_graph
    def __test(idx):
        idx = tuple(idx)

        x_in = tf.placeholder(tf.float32, shape=x.shape, name='x')

        outvars = crema.model.ops.whiten(x_in, idx, s_min=1e-10)

        with tf.Session() as sess:
            y_pred = sess.run(outvars, feed_dict={x_in: x})

        y_true = __whiten(x, idx, s_min=1e-10)

        eq_(y_pred.shape, x.shape)
        assert np.allclose(y_true, y_pred, atol=1e-6)

    for idx in [ [1, 2, 3], [2, 3], [3], [1, 2]]:
        yield __test, idx

def __gain(x, default):

    x_pow = np.abs(x)

    weight = default * np.ones((1, 1, x.shape[2], x.shape[3]))

    w_mag = np.log1p(np.exp(weight))

    return np.log1p(x * w_mag) / np.log1p(w_mag)


@new_graph
def test_gain():
    x = 100 * np.abs(np.random.randn(16, 96, 36, 8), dtype=np.float32) + 30

    x_in = tf.placeholder(tf.float32, shape=x.shape, name='x')

    outvars = crema.model.ops.gain(x_in, default=1.0)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        y_pred = sess.run(outvars, feed_dict={x_in: x})

    y_true = __gain(x, default=1.0)

    eq_(y_pred.shape, x.shape)
    assert np.allclose(y_true, y_pred, atol=1e-6)


def test_he_std():

    @new_graph
    def __test(shape, sym, target):

        pred = crema.model.init.he_std(shape, sym)

        eq_(pred, target)


    yield __test, [1, 1, 1, 1], True, 1
    yield __test, [1, 1, 1, 1], False, np.sqrt(2)
    yield __test, [1, 1, 1, 10], True, 1
    yield __test, [1, 1, 1, 10], False, np.sqrt(2)
    yield __test, [3, 5, 2, 10], True, np.power(3 * 5 * 2, -0.5)
    yield __test, [3, 5, 2, 10], False, np.sqrt(2) * np.power(3 * 5 * 2, -0.5)


@new_graph
def test_constant():

    value = 3.0
    shape = (2, 4, 8, 16)

    w = crema.model.init.constant(shape, default=value)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        w_out = sess.run(w)

    eq_(w_out.shape, shape)
    assert np.allclose(w_out, value)


def test_he_normal():

    @new_graph
    def __test(shape, sym):
        w = crema.model.init.he_normal(shape, sym=sym)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            w_out = sess.run(w)

        if sym:
            gain = 1.0
        else:
            gain = np.sqrt(2.0)

        std_target = gain * np.prod(shape[:-1])**(-0.5)

        # Make sure that we're close to zero-mean
        assert np.abs(np.mean(w_out)) <= 1e-2
        assert np.abs(np.var(w_out) - std_target**2) <= 1e-2

    for n_w in [3, 9, 23]:
        for n_c in [3, 7]:
            for sym in [False, True]:
                yield __test, (n_w, n_w, n_c, 1000), sym


def test_he_uniform():

    @new_graph
    def __test(shape, sym):
        w = crema.model.init.he_uniform(shape, sym=sym)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            w_out = sess.run(w)

        if sym:
            gain = 1.0
        else:
            gain = np.sqrt(2.0)

        std_target = gain * np.prod(shape[:-1])**(-0.5)

        # Make sure that we're close to zero-mean
        assert np.abs(np.mean(w_out)) <= 1e-2
        assert np.abs(np.var(w_out) - std_target**2) <= 1e-2

    for n_w in [3, 9, 23]:
        for n_c in [3, 7]:
            for sym in [False, True]:
                yield __test, (n_w, n_w, n_c, 1000), sym


@new_graph
def test_expand_mask():

    x = np.random.randn(100) > 0.0

    x_in = tf.placeholder(tf.bool, shape=x.shape, name='x')

    outvars = crema.model.ops.expand_mask(x_in)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        y_pred = sess.run(outvars, feed_dict={x_in: x})

    eq_(y_pred.ndim, 3)
    eq_(y_pred.shape[0], x.shape[0])
    eq_(y_pred.size, x.shape[0])
    assert np.issubdtype(y_pred.dtype, np.float)
    assert np.allclose(y_pred.squeeze(), x.astype(float))


@new_graph
def test_logsigmoid():

    x = np.random.randn(50, 100)

    x_in = tf.placeholder(tf.float32, shape=x.shape, name='x')

    outvars = crema.model.ops.logsigmoid(x_in)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        y_pred = sess.run(outvars, feed_dict={x_in: x})

    y_true = - np.log1p(np.exp(-x))

    assert np.allclose(y_pred, y_true)


def test_ndsoftmax_xent():


    @new_graph
    def __test(dims):
        x = np.random.randn(50, 100, 30)
        y = (np.random.randn(*x.shape) > 0).astype(np.float)
        x_in = tf.placeholder(tf.float32, shape=x.shape, name='x')
        y_in = tf.placeholder(tf.float32, shape=x.shape, name='y')

        outvars = crema.model.ops.ndxent(x_in, y_in, dims)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            y_pred = sess.run(outvars, feed_dict={x_in: x, y_in: y})

        y_true = - np.sum(x * y, axis=tuple(dims))

        assert np.allclose(y_pred, y_true, atol=5)

    for dims in [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]:
        yield __test, dims
