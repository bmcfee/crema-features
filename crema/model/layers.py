#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Model layer constructors'''

import tensorflow as tf
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
                nl_kwargs=None):
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

    nl_kwargs : dict
        If provided, additional keyword arguments to the nonlinearity

    Returns
    -------
    output : tf.Operator
        The output node of the layer

    '''
    x_shape = input_tensor.get_shape()

    filter_shape = [shape[0], shape[1], int(x_shape[-1]), n_filters]

    if nonlinearity is None:
        nonlinearity = tf.identity

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


        weight = init.he_uniform(filter_shape, name='weight', sym=sym)
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
    return conv2_layer(input_tensor, [1, 1], n_classes,
                       name=name,
                       mode=mode,
                       nonlinearity=ops.logsigmoid,
                       squeeze_dims=squeeze_dims,
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

    return conv2_layer(x, [1, 1], n_classes,
                       name=name,
                       mode=mode,
                       nonlinearity=ops.ndsoftmax,
                       nl_kwargs=dict(reduction_indices=[2, 3]),
                       squeeze_dims=squeeze_dims,
                       reg=reg)

