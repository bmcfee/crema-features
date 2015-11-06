#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tests for first-level audio feature extraction'''

import crema
import numpy as np

import jams

from nose.tools import raises, eq_


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

    # Ideal vectors
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
