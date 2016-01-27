#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tests for the data processing layer'''

import jams
import numpy as np
import joblib

from nose.tools import eq_, raises

import pandas as pd

import crema
import crema.data

TEST_FILE = 'data/test1_44100.wav'
TEST_JAMS = 'data/test1_44100.jams'


def test_data_duration_pass():

    data = dict(longest=np.empty((2, 50, 3, 4)),
                shortest=np.empty((2, 30, 5, 6)),
                threedim=np.empty((2, 31, 10)),
                flat=np.empty((2, 10)))

    n = crema.data.data_duration(data)

    eq_(n, 30)


@raises(RuntimeError)
def test_data_duration_fail():
    data = dict(longest=np.empty((2, 50)),
                shortest=np.empty((2, 30)),
                threedim=np.empty((2, 31)),
                flat=np.empty((2, 10)))

    crema.data.data_duration(data)


def test_slice_data():

    data = dict(ts1=np.random.randn(5, 50, 3, 3),
                ts2=np.random.randn(5, 50, 7, 1),
                flat1=np.random.randn(5),
                flat2=np.random.randn(5, 100))

    sample = slice(10, 20)

    subsample = crema.data.slice_data(data, sample)

    eq_(data.keys(), subsample.keys())
    assert np.all(subsample['ts1'] == data['ts1'][:, sample])
    assert np.all(subsample['ts2'] == data['ts2'][:, sample])
    assert np.all(subsample['flat1'] == data['flat1'])
    assert np.all(subsample['flat2'] == data['flat2'])


def test_jams_mapping():

    tasks = [crema.task.BeatTransformer(),
             crema.task.VectorTransformer('vector', 64)]

    jam = jams.load(TEST_JAMS)

    data = crema.data.jams_mapping(TEST_JAMS, tasks)

    for task in tasks:
        target = task.transform(jam)

        for key in target:
            assert key in data
            assert np.allclose(target[key], data[key][0])


def test_make_data():
    tasks = [crema.task.BeatTransformer(),
             crema.task.VectorTransformer('vector', 64)]

    crema_input = crema.pre.CQTensor()

    data = crema.data.make_task_data(TEST_FILE, TEST_JAMS, tasks, crema_input)

    feature = crema_input.extract(TEST_FILE)
    assert np.allclose(feature['input_cqtensor'], data['input_cqtensor'][0])

    jam = jams.load(TEST_JAMS)
    for task in tasks:
        target = task.transform(jam)

        for key in target:
            assert key in data
            assert np.allclose(target[key], data[key][0])


def test_sampler():
    tasks = [crema.task.BeatTransformer(),
             crema.task.VectorTransformer('vector', 64)]

    crema_input = crema.pre.CQTensor()
    all_data = crema.data.make_task_data(TEST_FILE, TEST_JAMS, tasks, crema_input)

    def __test(n_samples, n_duration):
        sampler = crema.data.sampler(TEST_FILE, TEST_JAMS, tasks, crema_input,
                                     n_samples, n_duration)

        for i, s in enumerate(sampler):
            eq_(set(s.keys()), set(all_data.keys()))

            for key in s:
                target_shape = list(all_data[key].shape)
                if len(target_shape) > 2:
                    target_shape[1] = n_duration

                eq_(list(s[key].shape), target_shape)

        eq_(i, n_samples-1)

    for n_samples in [1, 2, 5]:
        for n_duration in [2, 10, 30]:
            yield __test, n_samples, n_duration


def test_data_cache():
    # First, get the raw features
    crema_input = crema.pre.CQTensor()
    data = crema.data.make_task_data(TEST_FILE, TEST_JAMS, [], crema_input)

    # Then create a cache
    cache = joblib.Memory(cachedir='./crema_cache/', verbose=0)
    data2 = crema.data.make_task_data(TEST_FILE, TEST_JAMS, [], crema_input, cache=cache)
    data3 = crema.data.make_task_data(TEST_FILE, TEST_JAMS, [], crema_input, cache=cache)

    assert np.all(data['input_cqtensor'] == data2['input_cqtensor'])
    assert np.all(data2['input_cqtensor'] == data3['input_cqtensor'])


def test_create_stream():

    sources = pd.read_csv('data/test_index.csv')

    tasks = [crema.task.ChordTransformer()]

    crema_input = crema.pre.CQTensor()

    def __test(n_duration, keys, thread):
        streamer = crema.data.create_stream(sources, tasks, crema_input,
                                            n_duration=n_duration, keys=keys, thread=thread)

        # Bound the stream to 10 examples, otherwise we'll run forever
        for sample, _ in zip(streamer.generate(), range(10)):
            eq_(sample['input_cqtensor'].shape[1], n_duration)

    for n_duration in [1, 8, 16]:
        for keys in [None, [1]]:
            for thread in [False, True]:
                yield __test, n_duration, keys, thread

    # And test for an exception if we give a bogus key
    yield raises(RuntimeError)(__test), 8, ['bogus key'], False


def test_mux_streams():
    sources = pd.read_csv('data/test_index.csv')

    tasks = [crema.task.ChordTransformer()]

    crema_input = crema.pre.CQTensor()


    streams = [crema.data.create_stream(sources, tasks, crema_input, n_duration=8) 
               for _ in range(3)]

    def __test(n_samples, n_batch):

        mux = crema.data.mux_streams(streams, n_samples, n_batch)

        for n, batch in enumerate(mux.generate()):
            for key in batch:
                eq_(batch[key].shape[0], n_batch)

        eq_(n+1, n_samples//min(n_samples, n_batch))

    for n_samples in [10, 20, 40]:
        for n_batch in [5, 10]:
            yield __test, n_samples, n_batch


def test_split():
    
    sources = pd.read_csv('data/test_index2.csv')

    all_keys = set(sources['key'])
    for train, test in crema.data.split(sources, random_state=0):
        eq_(all_keys, train | test)
        assert not train & test
