#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Chord recognition task transformer'''

import numpy as np
import mir_eval

from .base import BaseTaskTransformer
from sklearn.preprocessing import MultiLabelBinarizer


class ChordTransformer(BaseTaskTransformer):

    def __init__(self, sr, hop_length):
        '''Initialize a chord task transformer'''

        super(ChordTransformer, self).__init__('chord|chord_harte',
                                               sr,
                                               hop_length,
                                               0)

        pitches = list(range(12))
        self.encoder = MultiLabelBinarizer()
        self.encoder.fit([pitches])
        self._classes = set(self.encoder.classes_)

    def transform(self, jam):

        anns = jam.search(namespace=self.namespace)

        if anns:
            intervals, chords = anns[0].data.to_interval_values()

            # Suppress all intervals not in the encoder
            pitch = []
            root = []
            bass = []

            for c in chords:
                # Encode the pitches
                r, s, b = mir_eval.chord.encode(c)
                s = np.roll(s, r)

                pitch.append(s)

                if r in self._classes:
                    root.extend(self.encoder.transform([[r]]))
                    bass.extend(self.encoder.transform([[(r+b) % 12]]))
                else:
                    root.extend(self.encoder.transform([[]]))
                    bass.extend(self.encoder.transform([[]]))

            mask = 1

        else:
            # Construct a blank annotation with mask = 0
            intervals = np.asarray([[0.0, jam.file_metadata.duration]])
            pitch = [[]]
            root = [[]]
            bass = [[]]
            mask = 0

        pitch = np.asarray(pitch)
        root = np.asarray(root)
        bass = np.asarray(bass)

        target_pitch = self.encode_intervals(intervals, pitch)
        target_root = self.encode_intervals(intervals, root)
        target_bass = self.encode_intervals(intervals, bass)

        target = np.stack([target_pitch, target_root, target_bass],
                          axis=2)

        return target, mask
