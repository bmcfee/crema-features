#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tests for model components'''

from decorator import decorator
from nose.tools import eq_, raises
from nose.plugins.skip import SkipTest

import crema
import numpy as np
import scipy


def require_or_skip(module_name):

    def __wrapper(func, *args, **kwargs):
        try:
            __import__(module_name)
        except ImportError:
            raise SkipTest

        return func(*args, **kwargs)

    return decorator(__wrapper)


@require_or_skip('tensorflow')
def test_gmean():

    pass
