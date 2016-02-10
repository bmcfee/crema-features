#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Model layer constructors'''

import six
import tensorflow as tf
from tensorflow.python import control_flow_ops

from . import init
from . import ops

def dense_layer(input_tensor, n_units, name=None, nonlinearity=tf.nn.relu,
                reg=False, nl_kwargs=None):
    '''A dense layer

    Parameters
    ----------
    input_tensor : tf.Tensor, shape=(n_batch, n_input)
        The input tensor

    n_units : int
        The number of outputs for this layer

    name : str
        A name for this layer

    nonlinearity : function
        The nonlinearity function to apply

    reg : bool
        If true, apply l2 regularization to the weights in this layer

    nl_kwargs : dict
        If provided, additional keyword arguments to the nonlinearity

    Returns
    -------
    output : tf.Operator
        The output node of the layer
    '''

    x_shape = input_tensor.get_shape()
    layer_shape = [int(x_shape[-1]), n_units]

    if nonlinearity is None:
        nonlinearity = tf.identity

    with tf.name_scope(name):
        if nonlinearity in (tf.nn.relu, tf.nn.relu6):
            sym = False
            default_bias = init.he_std(layer_shape, sym=sym)
        else:
            sym = True
            default_bias = 0

        weight = init.he_uniform(layer_shape, name='weight', sym=sym)
        bias = init.constant([n_units], name='bias', default=default_bias)

        response = tf.matmul(input_tensor, weight) + bias

        activation = nonlinearity(response, **(nl_kwargs if nl_kwargs else {}))

        output = tf.identity(activation, name='activation')

        if reg:
            penalty = tf.reduce_sum(tf.square(weight), name='l2_penalty')
            tf.add_to_collection('penalty', penalty)

    return output


def conv2_layer(input_tensor, shape, n_filters,
                name=None,
                nonlinearity=tf.nn.relu,
                strides=None,
                mode='SAME',
                squeeze_dims=None,
                reg=False,
                batch_norm=False,
                nl_kwargs=None,
                tied_init=False):
    '''A 2-dimensional convolution layer.

    Parameters
    ----------
    input_tensor : tf.Tensor, shape=(n_batch, n_width, n_height, n_channels)
        The input tensor

    shape : list of int
        The dimensions of the convolutional filters [width, height]

    n_filters : int
        The number of filters for this layer

    name : str
        A name for this layer

    nonlinearity : function
        The nonlinearity function to apply

    strides : None or tuple of int
        Optional striding for the convolution.

    mode : str
        The convolution mode 'SAME' or 'VALID'

    squeeze_dims : None or tuple of int
        If provided, the specified dimensions will be squeezed

    reg : bool
        If true, apply l2 regularization to the weights in this layer

    batch_norm : bool
        If true, apply batch normalization

    nl_kwargs : dict
        If provided, additional keyword arguments to the nonlinearity

    Returns
    -------
    output : tf.Operator
        The output node of the layer

    '''
    x_shape = input_tensor.get_shape()

    # If a shape is None, make it span the full extent of that input dimension
    shape = list(shape)
    for i in [0, 1]:
        if shape[i] is None:
            shape[i] = int(x_shape[i+1])

    filter_shape = [shape[0], shape[1], int(x_shape[-1]), n_filters]

    if nonlinearity is None:
        nonlinearity = tf.identity
    elif isinstance(nonlinearity, six.string_types):
        nonlinearity = getattr(tf.nn, nonlinearity)


    if strides is None:
        strides = [1, 1, 1, 1]
    else:
        strides = [1, strides[0], strides[-1], 1]

    with tf.name_scope(name):
        if nonlinearity in (tf.nn.relu, tf.nn.relu6):
            sym = False
            default_bias = init.he_std(filter_shape, sym=sym)
        else:
            sym = True
            default_bias = 0


        if tied_init:
            weight = init.he_uniform_tied(filter_shape, name='weight', sym=sym)
        else:
            weight = init.he_uniform(filter_shape, name='weight', sym=sym)

        if batch_norm:
            _response = tf.nn.conv2d(input_tensor, weight, strides=strides, padding=mode)
            response = batch_norm_layer(_response, n_filters)

        else:
            bias = init.constant([n_filters], name='bias', default=default_bias)
            response = tf.nn.conv2d(input_tensor, weight, strides=strides, padding=mode) + bias

        activation = nonlinearity(response, **(nl_kwargs if nl_kwargs else {}))

        if squeeze_dims is None:
            output = tf.identity(activation, name='activation')
        else:
            output = tf.squeeze(activation, squeeze_dims=squeeze_dims, name='activation')

        if reg:
            penalty = tf.reduce_sum(tf.square(weight), name='l2_penalty')
            tf.add_to_collection('penalty', penalty)

    return output


def conv2_multilabel(input_tensor, n_classes, name=None, squeeze_dims=None, mode='SAME',
                     reg=True):
    '''Convolutional multi-label output.

    This is useful for doing time-varying multi-label predictions.

    Parameters
    ----------
    input_tensor : tf.Tensor
        The input tensor

    n_classes : int
        The number of classes to predict

    name : str
        Name for this layer

    squeeze_dims : None or list of int
        Dimensions which may be squeezed

    mode : str
        convolution mode 'SAME' or 'VALID'

    reg : bool
        If true, apply l2 regularization to the weights in this layer

    Returns
    -------
    logits : tf.Operator
        log-likelihood tensor for each class

    See Also
    --------
    conv2_layer
    ops.logsigmoid
    '''
    return conv2_layer(input_tensor, [1, None], n_classes,
                       name=name,
                       mode=mode,
                       nonlinearity=ops.logsigmoid,
                       squeeze_dims=squeeze_dims,
                       batch_norm=True,
                       reg=reg)


def conv2_softmax(x, n_classes, name=None, squeeze_dims=None, mode='SAME', reg=True):
    '''Convolutional multi-class output.

    This is useful for doing time-varying multi-class predictions.

    Parameters
    ----------
    input_tensor : tf.Tensor
        The input tensor

    n_classes : int
        The number of classes to predict

    name : str
        Name for this layer

    squeeze_dims : None or list of int
        Dimensions which may be squeezed

    mode : str
        convolution mode 'SAME' or 'VALID'

    reg : bool
        If true, apply l2 regularization to the weights in this layer

    Returns
    -------
    logits : tf.Operator
        log-likelihood tensor

    See Also
    --------
    conv2_layer
    ops.ndsoftmax
    '''

    return conv2_layer(x, [1, None], n_classes,
                       name=name,
                       mode=mode,
                       nonlinearity=ops.ndsoftmax,
                       nl_kwargs=dict(reduction_indices=[2, 3]),
                       squeeze_dims=squeeze_dims,
                       batch_norm=True,
                       reg=reg)


def __get_global(name, collection='global', scope=None):
    '''Get a variable by name from the global context'''

    collection = tf.get_collection(collection, scope=scope)

    ops = [_ for _ in collection if name in _.name]

    if not ops:
        raise ValueError('variable not found: {}'.format(name))

    return ops


def batch_norm_layer(x, n_out, decay=0.9, name='batchnorm', affine=True):
    """Batch normalization on convolutional maps.

    Parameters
    ----------
    x: Tensor, 4D
        BHWD input maps

    n_out: integer
        depth of input maps

    train_flag: boolean tf.Variable
        true indicates training phase

    name: string
        variable scope

    affine: bool
        whether to affine-transform outputs

    Returns
    -------
    normed
        batch-normalized maps

    Based on the implementation described at http://stackoverflow.com/a/34634291
    """

    train_flag = __get_global('is_training')[0]

    with tf.variable_scope(name):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma', trainable=affine)

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')

        ema = tf.train.ExponentialMovingAverage(decay=decay)

        ema_apply_op = ema.apply([batch_mean, batch_var])

        def __mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        def __mean_var_without_update():
            return ema.average(batch_mean), ema.average(batch_var)

        mean, var = control_flow_ops.cond(train_flag,
                                          __mean_var_with_update,
                                          __mean_var_without_update)

        normed = tf.nn.batch_norm_with_global_normalization(x, mean, var,
                                                            beta, gamma,
                                                            1e-3, affine)
    return normed
