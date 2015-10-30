#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Instrument recognition task transformer'''

import pandas as pd
from .base import BaseTaskTransformer


class InstrumentTransformer(BaseTaskTransformer):

    def __init__(self, jam, n_samples, label_encoder):
        '''Initialize an instrument transformer

        Parameters
        ----------
        jam : jams.JAMS
            The JAMS object container

        n_samples : int > 0
            The number of samples in the audio frame

        label_encoder : sklearn.preprocessing.MultiLabelBinarizer
            The (pre-constructed) label encoder
        '''

        # Grab the data frame from the first matching annotation
        # TODO
        #   support multiple annotations per track
        #   apply label encoder to values
        #   sample values according to the target sampling rate/shape
        data = jam.search(namespace='tag_medleydb_instruments')[0].data

        # Convert the instrument activations to a matrix of activation functions

    def __getitem__(self, key):
        pass
