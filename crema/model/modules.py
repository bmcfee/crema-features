#!/usr/bin/env python
'''Module constructors'''

import copy
import tensorflow as tf     # pylint: disable=import-error
import six

from . import ops
from . import layers
from .. import pre
from .. import task

def shared(inputs, layer_defs=None, name='shared'):
    '''Create a low-level feature extraction module

    Parameters
    ----------
    inputs : dict: str -> tf.Tensor
        A collection of input variables

    layer_defs : list
        A list of layer specifications

    name : str
        The name for this subgraph

    Returns
    -------
    features : tf.Tensor
        The output node of the final layer
    '''

    layer_defs = copy.deepcopy(layer_defs)

    for key in ['input_cqt', 'input_cqtensor']:
        if key in inputs:
            break

    variables = {key: inputs[key]}

    with tf.name_scope(name):
        for layer in layer_defs:

            for node, params in layer.items():
                layer_in = variables[params.pop('input')]
                layer_out = params.get('name')

                try:
                    operator = getattr(ops, node)
                except AttributeError:
                    operator = getattr(layers, node)

                variables[layer_out] = operator(layer_in, **params)

    # Add the last output to the outputs collection
    tf.add_to_collection('outputs', variables[layer_out])

    return variables[layer_out]


def chord(features, name='chord'):
    '''Construct the submodule for chord estimation

    Parameters
    ----------
    features : tf.Tensor
        The input tensor to the module

    name : str
        The name of this subgraph

    Returns
    -------
    pitches, root, bass : tf.Tensor
        Prediction nodes for pitches, root, and bass

        pitches are n-by-time-by-12, encoding the probability that each
        pitch class is active.

        root and bass are n-by-time-by-13, encoding the distribution over pitches,
        including an additional `N` coordinate for unpitched predictions.

    '''
    # Set up the targets

    target_pc = tf.placeholder(tf.bool, shape=[None, None, 12], name='output_pitches')
    target_root = tf.placeholder(tf.bool, shape=[None, None, 13], name='output_root')
    target_bass = tf.placeholder(tf.bool, shape=[None, None, 13], name='output_bass')
    mask_chord = tf.placeholder(tf.bool, shape=[None], name='mask_chord')

    with tf.name_scope(name):
        z_chord = ops.expand_mask(mask_chord, name='mask_chord')

        pitch_logit = layers.conv2_multilabel(features, 12,
                                              squeeze_dims=[2],
                                              name='pitches_module')

        root_logit = layers.conv2_softmax(features, 13,
                                          squeeze_dims=[2],
                                          name='root_module')

        bass_logit = layers.conv2_softmax(features, 13,
                                          squeeze_dims=[2],
                                          name='bass_module')

        pitches = tf.exp(pitch_logit, name='pitches')
        root = tf.exp(root_logit, name='root')
        bass = tf.exp(bass_logit, name='bass')

        # Set up the losses
        with tf.name_scope('loss'):
            f_mask = tf.inv(tf.reduce_mean(z_chord))
            with tf.name_scope('pitches'):
                pc_loss = tf.reduce_mean(z_chord *
                                         tf.nn.sigmoid_cross_entropy_with_logits(pitch_logit,
                                                                                 tf.to_float(target_pc)),
                                         name='loss')

            with tf.name_scope('root'):
                root_loss = tf.reduce_mean(z_chord *
                                           ops.ndxent(root_logit,
                                                      tf.to_float(target_root),
                                                      [2]),
                                           name='loss')

            with tf.name_scope('bass'):
                bass_loss = tf.reduce_mean(z_chord *
                                           ops.ndxent(bass_logit,
                                                      tf.to_float(target_bass),
                                                      [2]),
                                           name='loss')

    tf.add_to_collection('outputs', pitches)
    tf.add_to_collection('outputs', root)
    tf.add_to_collection('outputs', bass)

    tf.add_to_collection('inputs', target_pc)
    tf.add_to_collection('inputs', target_root)
    tf.add_to_collection('inputs', target_bass)
    tf.add_to_collection('inputs', mask_chord)

    tf.add_to_collection('loss', pc_loss)
    tf.add_to_collection('loss', root_loss)
    tf.add_to_collection('loss', bass_loss)

    tf.scalar_summary('{:s}/pitches'.format(name), f_mask * pc_loss)
    tf.scalar_summary('{:s}/root'.format(name), f_mask * root_loss)
    tf.scalar_summary('{:s}/bass'.format(name), f_mask * bass_loss)

    return pitches, root, bass


def tags_global(features, n_tags, name='tags_global'):
    '''Construct a global (time-independent) tag module

    Parameters
    ----------
    features : tf.Tensor
        The input features to predict from

    n_tags : int > 0
        The number of output tags

    name : str
        A name for this submodule


    Returns
    -------
    tags : tf.Tensor
        The prediction output for this module: probability for each tag being active.
    '''
    target_tags = tf.placeholder(tf.bool, shape=[None, n_tags], name='output_{:s}'.format(name))
    mask_tags = tf.placeholder(tf.bool, shape=[None], name='mask_{:s}'.format(name))

    with tf.name_scope(name):
        z_tag = ops.expand_mask(mask_tags, name='mask_tag')

        # Make the logits
        tag_logit = layers.conv2_multilabel(features, n_tags, squeeze_dims=[2], name='tag_module')

        # Mean-pool the logits over time
        global_logit = tf.reduce_mean(tag_logit, reduction_indices=[1], name='tag_pool')

        tag_predict = tf.exp(tag_logit, name='tags_{:s}'.format(name))

        # Set up the losses
        with tf.name_scope('loss'):
            f_mask = tf.inv(tf.reduce_mean(z_tag))

            with tf.name_scope('tag'):
                tag_loss = tf.reduce_mean(z_tag *
                                          tf.nn.sigmoid_cross_entropy_with_logits(global_logit,
                                                                                  tf.to_float(target_tags)),
                                          name='loss')

    tf.add_to_collection('outputs', tag_predict)

    tf.add_to_collection('inputs', target_tags)
    tf.add_to_collection('inputs', mask_tags)

    tf.add_to_collection('loss', tag_loss)

    tf.scalar_summary('tags/{:s}'.format(name), f_mask * tag_loss)

    return tag_predict


def tags(features, n_tags, name='tags'):
    '''Construct a time-varying tag module

    Parameters
    ----------
    features : tf.Tensor
        The input features to predict from

    n_tags : int > 0
        The number of output tags

    name : str
        A name for this submodule


    Returns
    -------
    tags : tf.Tensor
        The prediction output for this module: probability for each tag being active at
        each time.
    '''
    target_tags = tf.placeholder(tf.bool, shape=[None, None, n_tags], name='output_{:s}'.format(name))
    mask_tags = tf.placeholder(tf.bool, shape=[None], name='mask_{:s}'.format(name))

    with tf.name_scope(name):
        z_tag = ops.expand_mask(mask_tags, name='mask_tag')

        # Make the logits
        tag_logit = layers.conv2_multilabel(features, n_tags, squeeze_dims=[2],
                                            name='tag_module')



        # Mean-pool the logits over time
        tag_predict = tf.exp(tag_logit, name='tags_{:s}'.format(name))

        # Set up the losses
        with tf.name_scope('loss'):
            f_mask = tf.inv(tf.reduce_mean(z_tag))

            with tf.name_scope('tag'):
                tag_loss = tf.reduce_mean(z_tag *
                                          tf.nn.sigmoid_cross_entropy_with_logits(tag_logit,
                                                                                  tf.to_float(target_tags)),
                                          name='loss')

    tf.add_to_collection('outputs', tag_predict)

    tf.add_to_collection('inputs', target_tags)
    tf.add_to_collection('inputs', mask_tags)

    tf.add_to_collection('loss', tag_loss)

    tf.scalar_summary('tags/{:s}'.format(name), f_mask * tag_loss)

    return tag_predict


def regression(features, dimension, name='factor'):
    '''Construct a linear regression layer

    Parameters
    ----------
    features : tf.Tensor
        The input features

    dimension : int > 0
        The number of regression outputs

    name : str
        The name for this submodule

    Returns
    -------
    vec : tf.Tensor
        The predictor outputs
    '''
    target_vec = tf.placeholder(tf.float32, shape=[None, dimension],
                                name='output_{:s}'.format(name))

    mask_vec = tf.placeholder(tf.bool, shape=[None],
                              name='mask_{:s}'.format(name))

    with tf.name_scope(name):

        z_vec = ops.expand_mask(mask_vec, name='mask_vec')

        # One convolutional layer
        conv_layer = layers.conv2_layer(features,
                                        [1, 1],
                                        dimension * 8,
                                        mode='VALID',
                                        batch_norm=True,
                                        name='factor_conv')

        # Pool out the time dimension
        conv_pool = tf.reduce_mean(conv_layer, reduction_indices=[1, 2])

        vec_predict = layers.dense_layer(conv_pool,
                                         dimension,
                                         name='predictor',
                                         nonlinearity=None,
                                         reg=True)

        with tf.name_scope('loss'):
            f_mask = tf.inv(tf.reduce_mean(z_vec))
            with tf.name_scope('vec'):
                vec_loss = tf.reduce_mean(z_vec * tf.nn.l2_loss(vec_predict - target_vec),
                                          name='vec_loss')


    tf.add_to_collection('outputs', vec_predict)
    tf.add_to_collection('inputs', target_vec)
    tf.add_to_collection('inputs', mask_vec)

    tf.add_to_collection('loss', vec_loss)

    tf.scalar_summary('regression/{:s}'.format(name), f_mask * vec_loss)

    return vec_predict


def make_input(spec):
    '''Construct the pre-processor, input variable, and shared features

    Parameters
    ----------
    spec : dict
        An input specification dictionary

    Returns
    -------
    crema_input : crema.pre.CremaInput
        The input transformation object

    features : tf.Tensor
        The feature output node
    '''

    cls = getattr(pre, spec['input']['pre'])
    crema_input = cls(**spec['input'].get('kwargs', {}))

    in_vars = dict()

    for varname, varspec in six.iteritems(crema_input.var):
        in_vars[varname] = tf.placeholder(varspec.dtype,
                                          shape=[None,
                                                 varspec.width,
                                                 varspec.height,
                                                 varspec.channels],
                                          name=varname)

        tf.add_to_collection('inputs', in_vars[varname])

    features = shared(in_vars, layer_defs=spec['layers'], name=spec['name'])

    return crema_input, features

def make_output(features, spec):
    '''Construct an output module

    Parameters
    ----------
    features : tf.Tensor
        Shared input features

    spec : dict
        Output module specification

    Returns
    -------
    transformer : crema.task.BaseTransformer
        The task transformer object

    '''

    Task = getattr(task, spec['task'])
    transformer = Task(name=spec['name'], **spec.get('params', {}))

    kwargs = dict(name=transformer.name)
    if isinstance(transformer, task.ChordTransformer):
        builder = chord
    elif isinstance(transformer, task.TimeSeriesLabelTransformer):
        builder = tags
        kwargs['n_tags'] = len(transformer.encoder.classes_)
    elif isinstance(transformer, task.GlobalLabelTransformer):
        builder = tags_global
        kwargs['n_tags'] = len(transformer.encoder.classes_)
    elif isinstance(transformer, task.VectorTransformer):
        builder = regression
        kwargs['dimension'] = transformer.dimension
    else:
        raise TypeError('Unsupported transformer type: {:s}'.format(type(transformer)))

    return transformer, builder(features, **kwargs)
