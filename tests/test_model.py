#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tests for model components'''

from decorator import decorator
from nose.tools import eq_, raises
from nose.plugins.skip import SkipTest

import crema
import numpy as np
import scipy


def require_or_skip(module_name):

    def __wrapper(func, *args, **kwargs):
        try:
            __import__(module_name)
        except ImportError as err:
            raise SkipTest(str(err))

        return func(*args, **kwargs)

    return decorator(__wrapper)


@require_or_skip('tensorflow')
def test_gmean():

    import tensorflow as tf
    import crema.model.utils

    x = np.abs(np.random.randn(5, 5, 5), dtype=np.float32)

    def __test(axis, keep_dims):
        y_true = scipy.stats.gmean(x, axis=axis)

        x_in = tf.placeholder(tf.float32, shape=(5, 5, 5), name='x')

        outvar = crema.model.utils.reduce_gmean(x_in, reduction_indices=[axis],
                                                keep_dims=keep_dims)

        with tf.Session() as sess:
            y_pred = sess.run(outvar, feed_dict={x_in: x})

        if keep_dims:
            eq_(y_pred.shape, x.shape)
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


@require_or_skip('tensorflow')
def test_ndsoftmax():

    import tensorflow as tf
    import crema.model.utils

    x = 100 * np.abs(np.random.randn(5, 5, 5), dtype=np.float32)

    def __test(axis):

        y_true = __softmax(x, axis)

        x_in = tf.placeholder(tf.float32, shape=(5, 5, 5), name='x')

        outvar = crema.model.utils.ndsoftmax(x_in, reduction_indices=axis)

        with tf.Session() as sess:
            y_pred = sess.run(outvar, feed_dict={x_in: x})

        eq_(y_pred.shape, x.shape)

        assert np.allclose(y_true, y_pred)
        assert np.all(np.isfinite(y_pred))
        assert np.allclose(y_pred.sum(tuple(axis)), 1.0)

    for axis in [[0], [1], [2], [2, 3]]:
        yield __test, axis
