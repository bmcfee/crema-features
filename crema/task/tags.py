#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tag task transformers'''

import numpy as np

from .base import BaseTaskTransformer
from sklearn.preprocessing import MultiLabelBinarizer

# Top 15 instruments by popularity in medleydb

INSTRUMENTS = ['drum set',
               'electric bass',
               'piano',
               'male singer',
               'clean electric guitar',
               'vocalists',
               'synthesizer',
               'female singer',
               'acoustic guitar',
               'distorted electric guitar',
               'auxiliary percussion',
               'double bass',
               'violin',
               'cello',
               'flute']


class TimeSeriesLabelTransformer(BaseTaskTransformer):

    def __init__(self, namespace, sr, hop_length, name, labels=None):
        '''Initialize a time-series label transformer

        Parameters
        ----------
        jam : jams.JAMS
            The JAMS object container

        n_samples : int > 0
            The number of samples in the audio frame

        label_encoder : sklearn.preprocessing.MultiLabelBinarizer
            The (pre-constructed) label encoder
        '''

        super(TimeSeriesLabelTransformer, self).__init__(namespace,
                                                         sr, hop_length, 0)

        self.encoder = MultiLabelBinarizer()
        self.encoder.fit([labels])
        self._classes = set(self.encoder.classes_)
        self.name = name

    def transform(self, jam):

        anns = jam.search(namespace=self.namespace)

        intervals = np.asarray([[0.0, jam.file_metadata.duration]])
        values = [None]
        mask = False

        if anns:
            ann_int, ann_val = anns[0].data.to_interval_values()
            intervals = np.vstack([intervals, ann_int])
            values.extend(ann_val)
            mask = True

        # Suppress all intervals not in the encoder
        tags = []
        for v in values:
            if v in self._classes:
                tags.extend(self.encoder.transform([[v]]))
            else:
                tags.extend(self.encoder.transform([[]]))

        tags = np.asarray(tags)
        target = self.encode_intervals(jam.file_metadata.duration,
                                       intervals,
                                       tags)
        return {'y_{:s}'.format(self.name): target,
                'mask_{:s}'.format(self.name): mask}


class GlobalLabelTransformer(BaseTaskTransformer):

    def __init__(self, namespace, name, labels=None):
        '''Initialize a global label transformer

        Parameters
        ----------
        jam : jams.JAMS
            The JAMS object container
        '''

        super(GlobalLabelTransformer, self).__init__(namespace, 1, 1, 0)

        self.encoder = MultiLabelBinarizer()
        self.encoder.fit([labels])
        self._classes = set(self.encoder.classes_)
        self.name = name

    def transform(self, jam):

        anns = jam.search(namespace=self.namespace)

        intervals = np.asarray([[0, 1]])
        values = [None]
        mask = False

        if anns:
            values = list(anns[0].data.value)
            intervals = np.tile(intervals, [len(values), 1])
            mask = True

        # Suppress all intervals not in the encoder
        tags = [v for v in values if v in self._classes]
        if len(tags):
            target = self.encoder.transform([tags]).max(axis=0)
        else:
            target = np.zeros(len(self._classes), dtype=np.int)

        return {'y_{:s}'.format(self.name): target,
                'mask_{:s}'.format(self.name): mask}
