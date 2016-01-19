#!/usr/bin/env python
'''Module constructors'''

import tensorflow as tf     # pylint: disable=import-error

from . import ops
from . import layers


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


def tags_global(features, n_tags, name='tags'):
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

        tags = tf.exp(tag_logit, name='tags_{:s}'.format(name))

        # Set up the losses
        with tf.name_scope('loss'):
            f_mask = tf.inv(tf.reduce_mean(z_tag))

            with tf.name_scope('tag'):
                tag_loss = tf.reduce_mean(z_tag *
                                          tf.nn.sigmoid_cross_entropy_with_logits(global_logit,
                                                                                  tf.to_float(target_tags)),
                                          name='tag_loss')

    tf.add_to_collection('outputs', tags)

    tf.add_to_collection('inputs', target_tags)
    tf.add_to_collection('inputs', mask_tags)

    tf.add_to_collection('loss', tag_loss)

    tf.scalar_summary('tags/{:s}'.format(name), f_mask * tag_loss)

    return tags


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
                                        name='factor_conv')

        # Pool out the time dimension
        with tf.name_scope('pooling'):
            conv_maxpool = tf.reduce_max(conv_layer, reduction_indices=[1, 2])
            conv_meanpool = tf.reduce_mean(conv_layer, reduction_indices=[1, 2])

            conv_pool = tf.concat(1, [conv_maxpool, conv_meanpool])

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
