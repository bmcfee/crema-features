#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''The base class for task transformer objects'''


class BaseTaskTransformer(object):

    def __init__(self, jam):
        raise NotImplementedError
