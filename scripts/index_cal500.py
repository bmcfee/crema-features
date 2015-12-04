#!/usr/bin/env python

import argparse
import os
import sys
import pandas as pd
import jams
import json


def process_args(args):
    '''Parse arguments'''

    parser = argparse.ArgumentParser(description='Index CAL500 data')

    parser.add_argument(dest='source_dir', type=str, help='Path to data')
    parser.add_argument(dest='output_file', type=str, help='Path to save index file')

    return vars(parser.parse_args(args))


def jam_to_audio(jam_file):

    jam = jams.load(jam_file, validate=False)

    return jam.sandbox.content_path


def index_audio(audio_files):

    return {os.path.splitext(os.path.basename(fn))[0]: fn
            for fn in audio_files}


def index_data(source_dir, output_file):

    audio_files = jams.util.find_with_extension(source_dir, 'mp3', depth=5)
    ann_files = jams.util.find_with_extension(source_dir, 'jamz', depth=5)

    df = pd.DataFrame(columns=['audio', 'jams', 'key'])

    audio_index = index_audio(audio_files)
    keys = [jam_to_audio(jf) for jf in ann_files]

    df['audio'] = [audio_index[k] for k in keys]
    df['jams'] = ann_files
    df['key'] = keys

    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    params = process_args(sys.argv[1:])
    index_data(**params)
