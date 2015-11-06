#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Feature pre-processing'''

import numpy as np

from .dsp import librosa


class CQT(object):

    def __init__(self, n_octaves=8, over_sample=3, fmin=None, dtype=np.float32):

        self.n_octaves = n_octaves
        self.over_sample = over_sample

        if fmin is None:
            fmin = librosa.note_to_hz('C1')

        self.fmin = fmin

        self.dtype = dtype

    def extract(self, infile):
        '''Extract Constant-Q spectra from an input file'''

        y, sr = librosa.load(infile)

        CQT = librosa.cqt(y,
                          n_bins=12 * self.n_octaves * self.over_sample,
                          bins_per_octave=12 * self.over_sample,
                          fmin=self.fmin).astype(self.dtype)

        n_frames = librosa.time_to_frames(librosa.get_duration(y))
        CQT = CQT[:n_frames]

        return CQT
