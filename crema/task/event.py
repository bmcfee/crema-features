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

        mask_beat = False
        mask_downbeat = False

        if anns:
            mask_beat = True
            intervals, values = anns[0].data.to_interval_values()
            values = np.asarray(values)

            beat_events = intervals[:, 0]
            beat_labels = np.ones((len(beat_events), 1))

            idx = (values == 1)
            if np.any(idx):
                downbeat_events = beat_events[idx]
                downbeat_labels = np.ones((len(downbeat_events), 1))
                mask_downbeat = True
            else:
                downbeat_events = np.zeros(0)
                downbeat_labels = np.zeros((0, 1))
        else:
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

        return {'y_beat': target_beat,
                'z_beat': mask_beat,
                'y_downbeat': target_downbeat,
                'z_downbeat': mask_downbeat}
