import argparse
from collections import defaultdict
import os
import os.path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare OGI Multilanguage Corpus data for training / testing.')

    parser.add_argument('ogi_dir', help='Path to OGI corpus directory')

    parser.add_argument('languages', type=lambda s: s.split(','),
                        help=('Comma-separated lest of first two letters '
                              'of names of each language to retain'))

    arguments = parser.parse_args()

    arguments.ogi_dir = os.path.expanduser(arguments.ogi_dir)

    # Validate OGI dir
    ogi_files = os.listdir(arguments.ogi_dir)
    if 'seglola' not in ogi_files:
        raise ValueError('Provided directory is not valid OGI corpus directory. '
                         'Should contain a "seglola" subdirectory.')

    return arguments


def load_split_data(split_name, ogi_dir, languages):
    """
    Load a list of recording identifiers for the given dataset split
    from the provided OGI directory, grouped by language. Return value
    is of the form:

        {'en': ['en084nlg', 'en084clg', ...],
         'ge': ['ge126htl', 'ge131clg', ...]}

    where the keys of the map are OGI languages and the values
    correspond to individual recordings.
    """

    ret = defaultdict(list)

    # Load split data file
    split_path = os.path.join(ogi_dir, 'trn_test', split_name + '.lst')
    with open(split_path, 'r') as split_entries_f:
        split_entries = []
        for split_line in split_entries_f:
            data = split_line.split()

            # Find a matching language (field in data[1] stores unique
            # ID, beginning with recording language)
            for language in languages:
                if data[1].startswith(language):
                    # Matching language. Extract all recordings
                    recording_names = data[2:len(data) - 2]
                    ret[language].extend([data[1] + recording
                                          for recording in recording_names])

    return dict(ret)


def get_data_file(recording_id, data_type, ogi_dir):
    """
    A general utility procedure to retrieve the path to a data file
    related to the given recording.

    `recording_id` should be a unique recording identifier as provided
    by `load_split_data` (of the form `en000nlg`, etc.).

    `data_type` may match any of the categories provided by the OGI
    corpus directory:

    - `calls`
    - `logs`
    - `logs2`
    - `seglola`
    """

    type_path = os.path.join(ogi_dir, data_type)

    # Match language key with full language name (why do they have to
    # lay it out this way??)
    language_key = recording_id[:2]
    language = None
    for language_opt in os.listdir(type_path):
        if language_opt.startswith(language_key):
            language = language_opt
            break

    if language is None:
        raise ValueError("Invalid language in recording ID: %s" % recording_id)

    language_path = os.path.join(type_path, language)

    if data_type == 'calls':
        # Calls directory is further split by call ID
        call_folder = recording_id[2:4]
        language_path = os.path.join(language_path, call_folder)
        if not os.path.isdir(language_path):
            raise ValueError("Invalid call ID in recording ID: %s"
                             % recording_id)

    files = os.listdir(language_path)
    for file in files:
        if file.startswith(recording_id):
            return os.path.join(language_path, file)

    raise ValueError("Recording %s not found in directory %s"
                     % (recording_id, language_path))


if __name__ == '__main__':
    args = parse_args()

    splits = ['train', 'devtest', 'evaltest']
    filenames = {split: load_split_data(split, args.ogi_dir, args.languages)
                 for split in splits}

    rec = filenames['train']['en'][0]
    print get_data_file(rec, 'calls', args.ogi_dir)

    # TODO: gather files, split with pysox, remove splits which are too short / are empty
