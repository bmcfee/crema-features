#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tests for the data processing layer'''

import jams
import numpy as np

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

    tasks = [crema.task.BeatTransformer(1000, 100),
             crema.task.VectorTransformer('vector', 64)]

    jam = jams.load(TEST_JAMS)

    data = crema.data.jams_mapping(TEST_JAMS, tasks)

    for task in tasks:
        target = task.transform(jam)

        for key in target:
            assert key in data
            assert np.allclose(target[key], data[key][0])


def test_make_data():
    tasks = [crema.task.BeatTransformer(crema.dsp.librosa['sr'],
                                        crema.dsp.librosa['hop_length']),
             crema.task.VectorTransformer('vector', 64)]

    crema_input = crema.pre.CremaInput()

    data = crema.data.make_task_data(TEST_FILE, TEST_JAMS, tasks, crema_input)

    feature = crema_input.extract(TEST_FILE)
    assert np.allclose(feature['input_crema_input'], data['input_crema_input'][0])

    jam = jams.load(TEST_JAMS)
    for task in tasks:
        target = task.transform(jam)

        for key in target:
            assert key in data
            assert np.allclose(target[key], data[key][0])


def test_sampler():
    tasks = [crema.task.BeatTransformer(crema.dsp.librosa['sr'],
                                        crema.dsp.librosa['hop_length']),
             crema.task.VectorTransformer('vector', 64)]

    crema_input = crema.pre.CremaInput()
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
    crema_input = crema.pre.CremaInput()
    data = crema.data.make_task_data(TEST_FILE, TEST_JAMS, [], crema_input)

    # Then create a cache
    cache = crema.data.init_cache('simple://')

    data2 = crema.data.make_task_data(TEST_FILE, TEST_JAMS, [], crema_input, cache=cache)
    data3 = crema.data.make_task_data(TEST_FILE, TEST_JAMS, [], crema_input, cache=cache)

    assert np.all(data['input_cqt'] == data2['input_Cqt'])
    assert np.all(data2['input_cqt'] == data3['input_cqt'])


def test_create_stream():

    sources = pd.read_csv('data/test_index.csv')

    tasks = [crema.task.ChordTransformer(crema.dsp.librosa['sr'],
                                         crema.dsp.librosa['hop_length'])]

    crema_input = crema.pre.CremaInput()

    def __test(n_duration):
        streamer = crema.data.create_stream(sources, tasks, crema_input, n_duration=n_duration)

        # Bound the stream to 10 examples, otherwise we'll run forever
        for sample, _ in zip(streamer.generate(), range(10)):
            eq_(sample['input_cqt'].shape[1], n_duration)

    for n_duration in [1, 8, 16]:
        yield __test, n_duration


def test_mux_streams():
    sources = pd.read_csv('data/test_index.csv')

    tasks = [crema.task.ChordTransformer(crema.dsp.librosa['sr'],
                                         crema.dsp.librosa['hop_length'])]

    crema_input = crema.pre.CremaInput()


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

