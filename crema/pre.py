#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Feature pre-processing'''

import numpy as np

from .dsp import librosa


class CQT(object):

    def __init__(self, n_octaves=8, over_sample=3, n_slice=2,
                 fmin=None, dtype=np.float32):

        self.n_octaves = n_octaves
        self.over_sample = over_sample

        if fmin is None:
            fmin = librosa.note_to_hz('C1')

        self.fmin = fmin

        self.n_slice = n_slice
        self.dtype = dtype

    def extract(self, infile):
        '''Extract Constant-Q spectra from an input file'''

        y, _ = librosa.load(infile)
        n_frames = librosa.time_to_frames(librosa.get_duration(y))

        cqspec = librosa.cqt(y,
                             n_bins=12 * self.n_octaves * self.over_sample,
                             bins_per_octave=12 * self.over_sample,
                             fmin=self.fmin).astype(self.dtype)

        return cqspec[:, :n_frames]

    def octensor(self, cqspec):
        '''Convert a constant-q spectrogram to an octave tensor

        Parameters
        ----------
        cqspec : np.ndarray, shape=(n_bins, t)
            Constant-Q spectrogram as produced by ``extract``

        Returns
        -------
        tensor : np.ndarray, shape=(t, n_bins_per_slice, n_octaves)
            cqspec re-arranged into (multi-)octave slices
        '''

        # 1. pad C up to an even multiple of bins_per_octave

        bins_per_octave = self.over_sample * 12

        # 2. carve C up into n_slice * bins_per_octave tiles,
        # strided by bins_per_octave
        n_octaves = self.n_octaves

        tensor = np.empty((cqspec.shape[1],
                           bins_per_octave * self.n_slice,
                           n_octaves - self.n_slice + 1),
                          dtype=cqspec.dtype)

        # Populate each sub-band
        for i in range(n_octaves - self.n_slice + 1):
            subband = slice(i * bins_per_octave,
                            i * bins_per_octave + tensor.shape[1])
            tensor[:, :, i] = cqspec[subband].T

        return tensor
