import argparse
from collections import defaultdict
import os
import os.path
import re
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare OGI Multilanguage Corpus data for training / testing.')

    parser.add_argument('ogi_dir', help='Path to OGI corpus directory')

    parser.add_argument('languages', type=lambda s: s.split(','),
                        help=('Comma-separated lest of first two letters '
                              'of names of each language to retain'))

    parser.add_argument('-o', '--output-directory', default='prepared',
                        help=('Directory to which prepared files should be '
                              'output'))

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


# Expected file extensions associated with each section of the corpus
TYPE_EXTENSIONS = {
    'calls': 'wav',
    'logs': 'log',
    'logs2': 'lg2',
    'seglola': 'seg',
}


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

    filename = recording_id + '.' + TYPE_EXTENSIONS[data_type]
    file_path = os.path.join(language_path, filename)
    if os.path.isfile(file_path):
        return file_path
    else:
        raise ValueError("Recording file %s not found in directory %s"
                         % (filename, language_path))


def process_recording(recording_id, ogi_dir):
    """
    Generate features for the given recording.

    Returns a path to an ARFF file containing features for the given
    recording.
    """

    wav_path = get_data_file(recording_id, 'calls', ogi_dir)

    decoded_path = decode_call_file(wav_path)
    split_paths = split_call_file(decoded_path)

    print split_paths

    # TODO extract features from split audio files (openSMILE)
    # TODO synthesize extracted features into hierarchical features
    # TODO run openSMILE on entire call as well
    # TODO add features from seglola files
    # TODO put it all in an ARFF output for the given recording and
    #   return path to ARFF file


def add_suffix(filename, suffix):
    """
    Add a dotted suffix to the given filename.

    >>> add_suffix('foo.wav', 'bar')
    'foo.bar.wav'
    """

    parts = filename.rsplit('.', 1)
    parts.insert(len(parts) - 1, suffix)
    return '.'.join(parts)


def decode_call_file(call_path):
    """
    Decode a NIST SPHERE call file into a normal WAV file.
    """

    decoded_path = add_suffix(call_path, 'decoded')

    try:
        retval = subprocess.call(["w_decode", '-f', call_path, decoded_path])
    except OSError, e:
        raise RuntimeError("Decoding failed. Is NIST SPHERE on your "
                           "path? (Look for a `w_decode` binary.)", e)

    return decoded_path


def split_call_file(call_path, split_size=2):
    """
    Split the given call audio file into equally-sized parts. Returns a
    list of paths to the resultant splits (which are placed in the same
    directory as the provided file, with some new extension).
    """

    new_path = add_suffix(call_path, 'split')

    sox_params = 'trim 0 2 : newfile : restart'.split()
    retval = subprocess.call(['sox', call_path, new_path] + sox_params)
    if retval != 0:
        raise RuntimeError("sox error: retval %i" % retval)

    # TODO remove splits which are empty / short?

    # Filename prefix of split files
    split_prefix = os.path.basename(call_path).rsplit('.', 1)[0] + '.split'
    call_dir = os.path.dirname(call_path)
    return [os.path.join(call_dir, filename)
            for filename in os.listdir(call_dir)
            if filename.startswith(split_prefix)]


if __name__ == '__main__':
    args = parse_args()

    splits = ['train', 'devtest', 'evaltest']
    filenames = {split: load_split_data(split, args.ogi_dir, args.languages)
                 for split in splits}

    rec = filenames['train']['en'][0]

    print get_data_file(rec, 'calls', args.ogi_dir)
    arff_file = process_recording(rec, args.ogi_dir)
