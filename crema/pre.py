#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Feature pre-processing'''

import numpy as np
from collections import namedtuple

from .dsp import librosa


Tensor = namedtuple('Tensor', ['dtype', 'width', 'height', 'channels'])


class CremaInput(object):
    def __init__(self):
        raise NotImplementedError

class CQTensor(CremaInput):

    def __init__(self, n_octaves=8, over_sample=3, n_slice=2,
                 fmin=None, dtype=np.float32):

        self.n_octaves = n_octaves
        self.over_sample = over_sample

        if fmin is None:
            fmin = librosa.note_to_hz('C1')

        self.fmin = fmin
        self.n_slice = n_slice
        self.dtype = dtype

        self.var = dict()

        self.var['input_cqt'] = Tensor(dtype,
                                       None,
                                       12 * self.over_sample * self.n_slice,
                                       self.n_octaves - 1)

    def extract(self, infile):
        '''Extract Constant-Q spectra from an input file'''

        y, _ = librosa.load(infile)

        # Construct the feature dictionary
        features = dict()

        # Extract the cqt spectrogram
        cqspec = self._cqt(y)

        # Populate the feature dictionary
        features['input_cqt'] = self._octensor(cqspec)
        return features

    def _cqt(self, y):
        '''Convert an audio time series to a CQT spectrogram

        Parameters
        ----------
        y : np.ndarray, shape=(t,)
            Audio time series

        Returns
        -------
        cqspec : np.ndarray, shape=(n_bins, t)
            Constant-Q spectrogram
        '''

        n_frames = librosa.time_to_frames(librosa.get_duration(y))

        cqspec = librosa.cqt(y,
                             n_bins=12 * self.n_octaves * self.over_sample,
                             bins_per_octave=12 * self.over_sample,
                             fmin=self.fmin).astype(self.dtype)

        # Max-normalize
        peak_energy = cqspec.max()
        if peak_energy >= 1e-10:
            cqspec /= peak_energy

        return cqspec[:, :n_frames]

    def _octensor(self, cqspec):
        '''Convert a constant-q spectrogram to an octave tensor

        Parameters
        ----------
        cqspec : np.ndarray, shape=(n_bins, t)
            Constant-Q spectrogram as produced by ``_cqt``

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
