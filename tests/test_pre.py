#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tests for first-level audio feature extraction'''

import crema
import numpy as np
import librosa

from nose.tools import raises, eq_

TEST_FILE = 'data/test1_44100.wav'


def test_input_shape():

    def __test(n_octaves, over_sample):
        
        CQT = crema.pre.CQT(n_octaves=n_octaves, over_sample=over_sample)

        C = CQT.extract(TEST_FILE)

        eq_(C.shape[0], 12 * n_octaves * over_sample)


    for n_octaves in [3, 4, 5]:
        for over_sample in [1, 2, 3]:
            yield __test, n_octaves, over_sample
