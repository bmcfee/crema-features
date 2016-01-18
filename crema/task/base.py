#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''The base class for task transformer objects'''

import numpy as np

# Note: we use the local preset-bound librosa here
from ..dsp import librosa


class BaseTaskTransformer(object):
    '''Base class for task transformer objects'''

    def __init__(self, namespace, fill_na):
        self.namespace = namespace

        if fill_na is None:
            fill_na = np.nan

        self.fill_na = fill_na

    def find_annotation(self, jam):
        anns = jam.search(namespace=self.namespace)

        if anns:
            i = np.random.choice(len(anns))
            return anns[i]

        return None

    def encode_events(self, duration, events, values):

        frames = librosa.time_to_frames(events)

        n_total = librosa.time_to_frames(duration)

        target = np.empty((n_total, values.shape[1]),
                          dtype=values.dtype)

        target.fill(self.fill_na)

        for column, event in zip(values, frames):
            target[event] = column

        return target.astype(np.bool)

    def encode_intervals(self, duration, intervals, values):

        frames = librosa.time_to_frames(intervals)

        n_total = librosa.time_to_frames(duration)

        target = np.empty((n_total, values.shape[-1]),
                          dtype=values.dtype)

        target.fill(self.fill_na)

        for column, interval in zip(values, frames):
            target[interval[0]:interval[1]] += column

        return target.astype(np.bool)
