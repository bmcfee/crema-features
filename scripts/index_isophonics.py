#!/usr/bin/env python

import argparse
import os
import sys
import pandas as pd
import jams


def process_args(args):
    '''Parse arguments'''

    parser = argparse.ArgumentParser(description='Index isophonics data')

    parser.add_argument(dest='source_dir', type=str, help='Path to data')
    parser.add_argument(dest='output_file', type=str, help='Path to save index file')

    return vars(parser.parse_args(args))


def index_data(source_dir, output_file):

    audio_files = jams.util.find_with_extension(source_dir, 'flac', depth=5)
    ann_files = jams.util.find_with_extension(source_dir, 'jams', depth=5)

    assert len(audio_files) == len(ann_files), len(audio_files)

    df = pd.DataFrame(columns=['audio', 'jams', 'key'])

    df['audio'] = audio_files
    df['jams'] = ann_files
    df['key'] = [os.path.relpath(os.path.dirname(fname), source_dir) 
                 for fname in audio_files]

    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    params = process_args(sys.argv[1:])
    index_data(**params)
