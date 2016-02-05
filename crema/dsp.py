#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Wrap librosa for default DSP settings'''

import librosa as _librosa
import presets

# Global DSP parameters via presets

librosa = presets.Preset(_librosa)
librosa['sr'] = 32768
librosa['hop_length'] = 256
