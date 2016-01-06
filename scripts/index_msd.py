#!/usr/bin/env python

import argparse
import os
import sys
import pandas as pd
import jams
from tqdm import tqdm


def process_args(args):
    '''Parse arguments'''

    parser = argparse.ArgumentParser(description='Index MSD data')

    parser.add_argument('-a', '--audio-ext', type=str, default='wav', help='Audio file extension', dest='aud_ext')
    parser.add_argument('-j', '--jams-ext', type=str, default='jams', help='JAMS file extension', dest='jams_ext')
    parser.add_argument('-s', '--split-ext', default=False, help='Split audio extension for indexing', dest='split_ext', action='store_true')
    parser.add_argument(dest='source_dir', type=str, help='Path to data')
    parser.add_argument(dest='output_file', type=str, help='Path to save index file')

    return vars(parser.parse_args(args))


def jam_to_key(jam_file):

    jam = jams.load(jam_file)

    return jam.file_metadata.identifiers.artist_id


def jamfile_to_audfile(jam_file, aud_ext, jams_ext):

    audio_file = jam_file.replace('{0}jams{0}'.format(os.path.sep),
                                  '{0}MSD_30sec{0}songs_msd{0}'.format(os.path.sep))

    return audio_file.replace('{0}{1}'.format(os.path.extsep, jams_ext),
                              '{0}clip{0}{1}'.format(os.path.extsep, aud_ext))


def index_audio(audio_files, split_ext):

    if split_ext:
        idx = lambda x : os.path.splitext(x)[0]
    else:
        idx = lambda x : x
    return {idx(os.path.basename(fn)): fn
            for fn in audio_files}


def index_data(source_dir, output_file, aud_ext, jams_ext, split_ext):

    ann_files = jams.util.find_with_extension(source_dir, jams_ext, depth=5)
    audio_files = [jamfile_to_audfile(x, aud_ext, jams_ext) for x in ann_files]

    frame = pd.DataFrame(columns=['audio', 'jams', 'key', 'original'])

    audio_index = index_audio(audio_files, split_ext)
    keys = [jam_to_key(jf) for jf in tqdm(ann_files, desc='Indexing jams')]

    frame['audio'] = audio_files
    frame['jams'] = ann_files
    frame['key'] = keys
    frame['original'] = True

    frame.to_csv(output_file, index=False)


if __name__ == '__main__':
    params = process_args(sys.argv[1:])
    index_data(**params)
