#!/usr/bin/env python
# -*- enconding: utf-8 -*-
'''Instantaneous event coding'''

import numpy as np

from .base import BaseTaskTransformer


class BeatTransformer(BaseTaskTransformer):

    def __init__(self, sr, hop_length):
        super(BeatTransformer, self).__init__('beat',
                                              sr, hop_length, 0)

    def transform(self, jam):

        anns = jam.search(namespace=self.namespace)

        if anns:
            mask = True
            intervals, values = anns[0].data.to_interval_values()
            values = np.asarray(values)

            beat_events = intervals[:, 0]
            beat_labels = np.ones((len(beat_events), 1))

            downbeat_events = beat_events[values == 1]
            downbeat_labels = np.ones((len(downbeat_events), 1))
        else:
            mask = False
            beat_events = np.zeros(0)
            beat_labels = np.zeros((0, 1))
            downbeat_events = beat_events
            downbeat_labels = beat_labels

        target_beat = self.encode_events(jam.file_metadata.duration,
                                         beat_events,
                                         beat_labels)

        target_downbeat = self.encode_events(jam.file_metadata.duration,
                                             downbeat_events,
                                             downbeat_labels)

        target = np.vstack([target_beat[np.newaxis, :],
                            target_downbeat[np.newaxis, :]])

        return target, mask
