#!/usr/bin/env python

import argparse
import os
import sys
import pandas as pd
import jams
import json


def process_args(args):
    '''Parse arguments'''

    parser = argparse.ArgumentParser(description='Index MedleyDB data')

    parser.add_argument('-k', '--key', dest='key_file',
                        type=str, help='Path to key file',
                        default='medley_artist_index.json')

    parser.add_argument(dest='source_dir', type=str, help='Path to data')
    parser.add_argument(dest='output_file', type=str, help='Path to save index file')

    return vars(parser.parse_args(args))


def index_track(fname, keys):

    dirname = os.path.dirname(fname)
    index = dirname.split(os.path.sep)[-1]

    return keys[index]


def index_data(source_dir, output_file, key_file):

    audio_files = jams.util.find_with_extension(source_dir, 'wav', depth=5)
    audio_files = sorted([_ for _ in audio_files if '_MIX' in _])

    ann_files = jams.util.find_with_extension(source_dir, 'jams', depth=5)

    with open(key_file, 'r') as fdesc:
        keys = json.load(fdesc)

    assert len(audio_files) == len(ann_files), len(ann_files)
    assert len(audio_files) == len(keys), len(keys)

    df = pd.DataFrame(columns=['audio', 'jams', 'key'])

    df['audio'] = audio_files
    df['jams'] = ann_files
    df['key'] = [index_track(fname, keys) for fname in audio_files]

    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    params = process_args(sys.argv[1:])
    index_data(**params)
