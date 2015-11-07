#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Regression task transformers'''

import numpy as np

from .base import BaseTaskTransformer


class VectorTransformer(BaseTaskTransformer):

    def __init__(self, namespace, dimension):

        super(VectorTransformer, self).__init__(namespace, 1, 1, 0)

        self.dimension = dimension

    def transform(self, jam):

        anns = jam.search(namespace=self.namespace)

        if anns:
            vector = np.asarray(anns[0].data.values.iloc[0])
            if len(vector) != self.dimension:
                raise RuntimeError('vector dimension({:0}) '
                                   '!= self.dimension({:1})'
                                   .format(len(vector), self.dimension))
            mask = True
        else:
            vector = np.zeros(self.dimension, dtype=np.float32)
            mask = False

        return vector, mask
