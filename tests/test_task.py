#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tests for first-level audio feature extraction'''

import crema
import numpy as np

import jams

from nose.tools import eq_


def test_task_chord_present():

    # Construct a jam
    jam = jams.JAMS(file_metadata=dict(duration=4.0))

    ann = jams.Annotation(namespace='chord')

    ann.append(time=0, duration=1.0, value='C:maj')
    ann.append(time=1, duration=1.0, value='C:maj/3')
    ann.append(time=3, duration=1.0, value='D:maj')

    jam.annotations.append(ann)

    # One second = one frame
    T = crema.task.ChordTransformer(sr=1, hop_length=1)

    y, mask = T.transform(jam)

    # Make sure we have the mask
    eq_(mask, True)

    # Check the shape
    assert np.allclose(y.shape, [3, 12, 4])

    pcp = y[0]
    root = y[1]
    bass = y[2]

    # Ideal vectors:
    # pcp = Cmaj, Cmaj, N, Dmaj
    # root: C, C, N, D
    # bass: C, E, N, D
    pcp_true = np.asarray([ [1, 1, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 0, 0],
                            [1, 1, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 1],
                            [1, 1, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])

    root_true = np.asarray([[1, 1, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])

    bass_true = np.asarray([[1, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])

    assert np.allclose(pcp, pcp_true)
    assert np.allclose(root, root_true)
    assert np.allclose(bass, bass_true)


def test_task_chord_absent():

    jam = jams.JAMS(file_metadata=dict(duration=4.0))
    T = crema.task.ChordTransformer(sr=1, hop_length=1)

    y, mask = T.transform(jam)

    # Mask should be false since we have no matching namespace
    eq_(mask, False)

    # Check the shape
    assert np.allclose(y.shape, [3, 12, 4])

    # Make sure it's empty
    assert not np.any(y)


def test_task_tslabel_present():
    labels = ['alpha', 'beta', 'psycho', 'aqua', 'disco']

    jam = jams.JAMS(file_metadata=dict(duration=4.0))

    ann = jams.Annotation(namespace='tag_open')

    ann.append(time=0, duration=1.0, value='alpha')
    ann.append(time=0, duration=1.0, value='beta')
    ann.append(time=1, duration=1.0, value='23')
    ann.append(time=3, duration=1.0, value='disco')

    jam.annotations.append(ann)
    T = crema.task.TimeSeriesLabelTransformer(namespace='tag_open',
                                              sr=1,
                                              hop_length=1,
                                              labels=labels)

    y, mask = T.transform(jam)

    # Mask should be true
    eq_(mask, True)

    # Check the shape
    assert np.allclose(y.shape, [len(labels), 4])

    # Decode the labels
    predictions = T.encoder.inverse_transform(y.T)

    true_labels = [['alpha', 'beta'], [], [], ['disco']]

    for t, p in zip(true_labels, predictions):
        eq_(set(t), set(p))


def test_task_tslabel_absent():
    labels = ['alpha', 'beta', 'psycho', 'aqua', 'disco']

    jam = jams.JAMS(file_metadata=dict(duration=4.0))
    T = crema.task.TimeSeriesLabelTransformer(namespace='tag_open',
                                              sr=1,
                                              hop_length=1,
                                              labels=labels)

    y, mask = T.transform(jam)

    # Mask should be false since we have no matching namespace
    eq_(mask, False)

    # Check the shape
    assert np.allclose(y.shape, [len(labels), 4])

    # Make sure it's empty
    assert not np.any(y)
