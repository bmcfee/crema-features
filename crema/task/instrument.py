#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Instrument recognition task transformer'''

import pandas as pd
from .base import BaseTaskTransformer


class InstrumentTransformer(BaseTaskTransformer):

    def __init__(self, jam, label_encoder):
        '''Initialize an instrument transformer

        Parameters
        ----------
        jam : jams.JAMS
            The JAMS object container

        label_encoder : sklearn.preprocessing.MultiLabelBinarizer
            The (pre-constructed) label encoder
        '''

        # Grab the data frame from the first matching annotation
        data = jam.search(namespace='tag_medleydb_instruments')[0].data

        # Re-index the data frame at the target sampling rate
