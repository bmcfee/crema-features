#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Chord recognition task transformer'''

import numpy as np
import mir_eval

from .base import BaseTaskTransformer
from sklearn.preprocessing import MultiLabelBinarizer


def _pad_nochord(target, axis=-1):

    ncmask = 1 - np.sum(target, axis=axis, keepdims=True)

    return np.concatenate([target, ncmask], axis=axis)


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

        ann = self.find_annotation(jam)

        # Construct a blank annotation with mask = 0
        intervals = np.asarray([[0.0, jam.file_metadata.duration]])
        chords = ['N']
        mask = False
        if ann:
            ann_ints, ann_chords = ann.data.to_interval_values()
            intervals = np.vstack([intervals, ann_ints])
            chords.extend(ann_chords)
            mask = True

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

        pitch = np.asarray(pitch)
        root = np.asarray(root)
        bass = np.asarray(bass)

        target_pitch = self.encode_intervals(jam.file_metadata.duration,
                                             intervals, pitch)
        target_root = self.encode_intervals(jam.file_metadata.duration,
                                            intervals, root)
        target_bass = self.encode_intervals(jam.file_metadata.duration,
                                            intervals, bass)

        return {'output_pitches': target_pitch,
                'output_root': _pad_nochord(target_root),
                'output_bass': _pad_nochord(target_bass),
                'mask_chord': mask}
