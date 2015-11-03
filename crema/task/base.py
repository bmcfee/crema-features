#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''The base class for task transformer objects'''

import numpy as np
from ..dsp import librosa


class BaseTaskTransformer(object):
    '''Base class for task transformer objects'''

    def __init__(self, namespace, sr, hop_length, fill_na):
        self.namespace = namespace
        self.sr = sr
        self.hop_length = hop_length

        if fill_na is None:
            fill_na = np.nan

        self.fill_na = fill_na

    def encode_intervals(self, intervals, values):

        frames = librosa.time_to_frames(intervals,
                                        sr=self.sr,
                                        hop_length=self.hop_length)

        n_total = np.max(frames)

        target = np.empty((values.shape[1], n_total),
                          dtype=values.dtype)

        target.fill(self.fill_na)

        for column, interval in zip(values, frames):
            target[:, interval[0]:interval[1]] += column[:, np.newaxis]

        return target
