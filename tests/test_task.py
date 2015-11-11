#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tests for first-level audio feature extraction'''

import crema
import numpy as np

import jams

from nose.tools import eq_, raises


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

    output = T.transform(jam)

    # Make sure we have the mask
    eq_(output['z_pitches'], True)
    eq_(output['z_root'], True)
    eq_(output['z_bass'], True)

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

    assert np.allclose(output['y_pitches'], pcp_true)
    assert np.allclose(output['y_root'], root_true)
    assert np.allclose(output['y_bass'], bass_true)


def test_task_chord_absent():

    jam = jams.JAMS(file_metadata=dict(duration=4.0))
    T = crema.task.ChordTransformer(sr=1, hop_length=1)

    output = T.transform(jam)

    # Mask should be false since we have no matching namespace
    eq_(output['z_pitches'], False)
    eq_(output['z_root'], False)
    eq_(output['z_bass'], False)

    # Check the shape
    assert np.allclose(output['y_pitches'].shape, [12, 4])
    assert np.allclose(output['y_root'].shape, [12, 4])
    assert np.allclose(output['y_bass'].shape, [12, 4])

    # Make sure it's empty
    assert not np.any(output['y_pitches'])
    assert not np.any(output['y_root'])
    assert not np.any(output['y_bass'])


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
                                              name='madeup',
                                              labels=labels)

    output = T.transform(jam)

    # Mask should be true
    eq_(output['z_madeup'], True)

    y = output['y_madeup']

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
                                              name='madeup',
                                              labels=labels)

    output = T.transform(jam)

    # Mask should be false since we have no matching namespace
    eq_(output['z_madeup'], False)
    y = output['y_madeup']

    # Check the shape
    assert np.allclose(y.shape, [len(labels), 4])

    # Make sure it's empty
    assert not np.any(y)


def test_task_glabel_absent():
    labels = ['alpha', 'beta', 'psycho', 'aqua', 'disco']

    jam = jams.JAMS(file_metadata=dict(duration=4.0))
    T = crema.task.GlobalLabelTransformer(namespace='tag_open',
                                          name='madeup',
                                          labels=labels)

    output = T.transform(jam)

    # Mask should be false since we have no matching namespace
    eq_(output['z_madeup'], False)

    # Check the shape
    eq_(output['y_madeup'].ndim, 1)
    eq_(output['y_madeup'].shape[0], len(labels))

    # Make sure it's empty
    assert not np.any(output['y_madeup'])


def test_task_glabel_present():
    labels = ['alpha', 'beta', 'psycho', 'aqua', 'disco']

    jam = jams.JAMS(file_metadata=dict(duration=4.0))

    ann = jams.Annotation(namespace='tag_open')

    ann.append(time=0, duration=1.0, value='alpha')
    ann.append(time=0, duration=1.0, value='beta')
    ann.append(time=1, duration=1.0, value='23')
    ann.append(time=3, duration=1.0, value='disco')

    jam.annotations.append(ann)
    T = crema.task.GlobalLabelTransformer(namespace='tag_open',
                                          name='madeup',
                                          labels=labels)

    output = T.transform(jam)

    # Mask should be true
    eq_(output['z_madeup'], True)

    # Check the shape
    eq_(output['y_madeup'].ndim, 1)
    eq_(output['y_madeup'].shape[0], len(labels))

    # Decode the labels
    predictions = T.encoder.inverse_transform(output['y_madeup'].reshape((1, -1)))[0]

    true_labels = ['alpha', 'beta', 'disco']

    eq_(set(true_labels), set(predictions))


def test_task_vector_absent():

    def __test(dimension, name):

        var_name = 'y_{:s}'.format(name)
        mask_name = 'z_{:s}'.format(name)

        jam = jams.JAMS(file_metadata=dict(duration=4.0))
        T = crema.task.VectorTransformer(namespace='vector',
                                         dimension=dimension,
                                         name=name)

        output = T.transform(jam)

        # Mask should be false since we have no matching namespace
        eq_(output[mask_name], False)

        # Check the shape
        eq_(output[var_name].ndim, 1)
        eq_(output[var_name].shape[0], dimension)

        # Make sure it's empty
        assert not np.any(output[var_name])

    for dimension in [1, 2, 4]:
        yield __test, dimension, 'collab'
        yield __test, dimension, 'vec'


def test_task_vector_present():

    def __test(target_dimension, data_dimension, name):
        var_name = 'y_{:s}'.format(name)
        mask_name = 'z_{:s}'.format(name)

        jam = jams.JAMS(file_metadata=dict(duration=4.0))
        T = crema.task.VectorTransformer(namespace='vector',
                                         dimension=target_dimension,
                                         name=name)

        ann = jams.Annotation(namespace='vector')
        ann.append(time=0, duration=1,
                   value=list(np.random.randn(data_dimension)))

        jam.annotations.append(ann)

        output = T.transform(jam)

        # Mask should be false since we have no matching namespace
        eq_(output[mask_name], True)

        # Check the shape
        eq_(output[var_name].ndim, 1)
        eq_(output[var_name].shape[0], target_dimension)

        # Make sure it's empty
        assert np.allclose(output[var_name], ann.data.loc[0].value)

    for target_d in [1, 2, 4]:
        for data_d in [1, 2, 4]:
            if target_d != data_d:
                tf = raises(RuntimeError)(__test)
            else:
                tf = __test
            yield tf, target_d, data_d, 'collab'
            yield tf, target_d, data_d, 'vec'


def test_task_beat_present():

    # Construct a jam
    jam = jams.JAMS(file_metadata=dict(duration=4.0))

    ann = jams.Annotation(namespace='beat')

    ann.append(time=0, duration=0.0, value=1)
    ann.append(time=1, duration=0.0, value=2)
    ann.append(time=2, duration=0.0, value=3)
    ann.append(time=3, duration=0.0, value=1)

    jam.annotations.append(ann)

    # One second = one frame
    T = crema.task.BeatTransformer(sr=2, hop_length=1)

    output = T.transform(jam)

    # Make sure we have the masks
    eq_(output['z_beat'], True)
    eq_(output['z_downbeat'], True)

    # Check the shape: 4 seconds at 2 samples per second
    # The first channel measures beats
    # The second channel measures downbeats
    assert np.allclose(output['y_beat'].shape, [1, 8])
    assert np.allclose(output['y_downbeat'].shape, [1, 8])

    # Ideal vectors:
    #   a beat every second (two samples)
    #   a downbeat every three seconds (6 samples)

    beat_true = np.asarray([[1, 0, 1, 0, 1, 0, 1, 0]])
    downbeat_true = np.asarray([[1, 0, 0, 0, 0, 0, 1, 0]])

    assert np.allclose(output['y_beat'], beat_true)
    assert np.allclose(output['y_downbeat'], downbeat_true)


def test_task_beat_nometer():

    # Construct a jam
    jam = jams.JAMS(file_metadata=dict(duration=4.0))

    ann = jams.Annotation(namespace='beat')

    ann.append(time=0, duration=0.0)
    ann.append(time=1, duration=0.0)
    ann.append(time=2, duration=0.0)
    ann.append(time=3, duration=0.0)

    jam.annotations.append(ann)

    # One second = one frame
    T = crema.task.BeatTransformer(sr=2, hop_length=1)

    output = T.transform(jam)

    # Make sure we have the mask
    eq_(output['z_beat'], True)
    eq_(output['z_downbeat'], False)

    # Check the shape: 4 seconds at 2 samples per second
    assert np.allclose(output['y_beat'].shape, [1, 8])
    assert np.allclose(output['y_downbeat'].shape, [1, 8])

    # Ideal vectors:
    #   a beat every second (two samples)
    #   no downbeats

    beat_true = np.asarray([[1, 0, 1, 0, 1, 0, 1, 0]])
    downbeat_true = np.asarray([[0, 0, 0, 0, 0, 0, 0, 0]])

    assert np.allclose(output['y_beat'], beat_true)
    assert np.allclose(output['y_downbeat'], downbeat_true)


def test_task_beat_absent():

    # Construct a jam
    jam = jams.JAMS(file_metadata=dict(duration=4.0))

    # One second = one frame
    T = crema.task.BeatTransformer(sr=2, hop_length=1)

    output = T.transform(jam)

    # Make sure we have the mask
    eq_(output['z_beat'], False)
    eq_(output['z_downbeat'], False)

    # Check the shape: 4 seconds at 2 samples per second
    assert np.allclose(output['y_beat'].shape, [1, 8])
    assert np.allclose(output['y_downbeat'].shape, [1, 8])
    assert not np.any(output['y_beat'])
    assert not np.any(output['y_downbeat'])
