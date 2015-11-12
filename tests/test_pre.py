#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tests for first-level audio feature extraction'''

import crema
import numpy as np

from nose.tools import eq_

TEST_FILE = 'data/test1_44100.wav'


def test_input_shape():

    def __test(n_octaves, over_sample):
        CQT = crema.pre.CQT(n_octaves=n_octaves, over_sample=over_sample)

        C = CQT.extract(TEST_FILE)

        eq_(C.shape[0], 12 * n_octaves * over_sample)

    for n_octaves in [3, 4, 5]:
        for over_sample in [1, 2, 3]:
            yield __test, n_octaves, over_sample


def test_octensor():

    def __test(n_octaves, over_sample, n_slice):
        CQT = crema.pre.CQT(n_octaves=n_octaves,
                            over_sample=over_sample,
                            n_slice=n_slice)

        cqspec = CQT.extract(TEST_FILE)
        octensor = CQT.octensor(cqspec)

        # Make sure our shapes line up
        eq_(octensor.ndim, 3)
        eq_(octensor.shape[0], cqspec.shape[1])
        eq_(octensor.shape[1], 12 * n_slice * over_sample)

        for i in range(octensor.shape[2]):
            # Compute the relevant slice from cqspec
            assert np.allclose(octensor[:, :, i].T,
                               cqspec[i * 12 * over_sample:
                                      (i + n_slice) * 12 * over_sample])

    for n_octaves in [3, 4]:
        for over_sample in [1, 2, 3]:
            for n_slice in [1, 2, 3]:
                yield __test, n_octaves, over_sample, n_slice
