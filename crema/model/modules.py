#!/usr/bin/env python

import tensorflow as tf

from . import ops
from . import layers

def chord_module(features, name='chord'):
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

        pitches are n-by-12, encoding the probability that each
        pitch class is active.

        root and bass are n-by-13, encoding the distribution over pitches,
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
