# Preprocessing for CSLU foreign accented English database

# similar to prepare.py, except we don't need the SPHERE decoding step

import argparse
from collections import defaultdict, namedtuple
import logging
import os
import os.path
import pickle
import re
import subprocess
from tempfile import NamedTemporaryFile

from recording import Recording, Segment

from sklearn.cross_validation import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare CSLU Foreign Accented English data for training / testing.')

    parser.add_argument('cslu_dir', help='Path to CSLU corpus directory')

    parser.add_argument('languages', type=lambda s: s.split(','),
                        help=('Comma-separated list of two-letter '
                              'language abbreviations'))

    parser.add_argument('-s', '--segment-length', type=int, default=2,
                        help='Audio segment length')
    parser.add_argument('--drop-short-segments', default=False, action='store_true',
                        help=('Drop segments which are shorter than '
                              'the provided segment length'))

    parser.add_argument('--gain-level', default=None, type=float,
                        help=('Decibel level for normalizing sound '
                              'clips'))

    parser.add_argument('-o', '--output-directory', default='prepared',
                        help=('Directory to which prepared files should be '
                              'output'))

    parser.add_argument('-c', '--opensmile-config',
                        default=os.path.join(SCRIPT_DIR, 'config',
                                             'opensmile.001.conf'),
                        help=('Path to openSMILE configuration file'))

    arguments = parser.parse_args()

    arguments.cslu_dir = os.path.expanduser(arguments.cslu_dir)

    # Validate CSLU dir
    
    cslu_files = os.listdir(arguments.cslu_dir)
    if 'speech' not in cslu_files:
        raise ValueError('Provided directory is not valid CSLU corpus directory. '
                         'Should contain a "speech" subdirectory.')

    if not os.path.isdir(arguments.output_directory):
        raise ValueError('Output directory %s does not exist'
                         % arguments.output_directory)

    return arguments


# CSLU data is not already divided into train/dev/eval splits
# need to create splits (randomly)

# This is roughly the equivalent of `load_split_data` for OGI
def get_filenames(splits, split_sizes, cslu_dir, languages):
    """
    Load a list of recording identifiers for the given dataset split
    from the provided OGI directory, grouped by language. Return value
    is of the form:

    {'train': {'en': ['en084nlg', 'en084clg', ...],
               'ge': ['ge126htl', 'ge131clg', ...]},
     'devtest': {...},
     'evaltest': {...}}

    where the keys of the map are OGI languages and the values
    correspond to individual recordings.
    """
    assert len(splits) == len(split_sizes)
    assert len(splits) == 3 # this constraint can be relaxed later

    def good_recording(folder, filename):
        file_size = os.path.getsize(os.path.join(folder, filename))
        # TODO: extra feature (from flag) to use particular fluency levels
        return file_size > 1e5

    
    def split_in_three(good_recs):
        sizes = split_sizes
        train, test = train_test_split(good_recs, train_size=1.0*sizes[0])
        devtest, evaltest = train_test_split(test, train_size=1.0*sizes[1]/(sizes[1] + sizes[2]))
        return (train, devtest, evaltest)

    ret = {split:{} for split in splits}

    # Find a matching language
    for language in languages:
        lang_path = os.path.join(cslu_dir, 'speech', language.upper())
        good_recs = [os.path.splitext(rec)[0] # cut off ".wav"
                     for rec in os.listdir(lang_path) if good_recording(lang_path, rec)]
        split_recs = split_in_three(good_recs)
        for i,split in enumerate(splits):
            ret[split][language] = split_recs[i]

            
    return ret


# Expected file extensions associated with each section of the corpus
TYPE_EXTENSIONS = {
    'speech': 'wav',
    'misc': '',
    'trans': 'inf',
}


def get_data_file(recording_id, data_type, cslu_dir):
    """
    A general utility procedure to retrieve the path to a data file
    related to the given recording.

    `recording_id` should be a unique recording identifier as provided
    by `load_split_data` (of the form `en000nlg`, etc.).

    `data_type` may match any of the categories provided by the CSLU
    corpus directory:

    - `speech`
    - `misc`
    - `trans`
    """

    type_path = os.path.join(cslu_dir, data_type)

    # Match language key with full language name (why do they have to
    # lay it out this way??)
    language_key = recording_id[1:3] # e.g. FGE00010
    language = None
    for language_opt in os.listdir(type_path):
        if language_opt.startswith(language_key):
            language = language_opt
            break

    if language is None:
        raise ValueError("Invalid language in recording ID: %s" % recording_id)

    language_path = os.path.join(type_path, language)

    filename = recording_id + '.' + TYPE_EXTENSIONS[data_type]
    file_path = os.path.join(language_path, filename)
    if os.path.isfile(file_path):
        return file_path
    else:
        raise IOError("Recording file %s not found in directory %s"
                      % (filename, language_path))


def process_recording(recording_id, args):
    """
    Generate features for the given recording.

    Returns a path to an ARFF file containing features for the given
    recording.
    """
    
    # get decoded_path directly, since CSLU is already in regular .wav files
    decoded_path = get_data_file(recording_id, 'speech', args.cslu_dir)

    if args.gain_level is not None:
        decoded_path = normalize_call_file(decoded_path, args.gain_level)

    segment_paths = split_call_file(decoded_path, args.segment_length,
                                    args.drop_short_segments)

    segments = []
    for segment_path in segment_paths:
        features = extract_audio_features(segment_path, args)
        if features is not None:
            segments.append(Segment(segment_path, features))

    return Recording(recording_id, segments)


def add_suffix(filename, suffix):
    """
    Add a dotted suffix to the given filename.

    >>> add_suffix('foo.wav', 'bar')
    'foo.bar.wav'
    """

    parts = filename.rsplit('.', 1)
    parts.insert(len(parts) - 1, suffix)
    return '.'.join(parts)


def normalize_call_file(call_path, gain_level=-3):
    """Normalize the audio level in the given call file."""

    new_path = add_suffix(call_path, 'norm')

    retval = subprocess.call(['sox', call_path, new_path,
                              'gain', '-n', str(gain_level)])
    if retval != 0:
        raise RuntimeError("sox error (normalization): retval %i" % retval)

    return new_path


def split_call_file(call_path, split_size=2, drop_short_segments=False):
    """
    Split the given call audio file into equally-sized segments. Returns
    a list of paths to the resultant segments (which are placed in the
    same directory as the provided file, with some new extension).
    """
    
    new_path = add_suffix(call_path, 'split')

    sox_params_str = 'trim 0 %i : newfile : restart' % split_size
    sox_params = sox_params_str.split()
    retval = subprocess.call(['sox', call_path, new_path] + sox_params)
    if retval != 0:
        raise RuntimeError("sox error (splitting): retval %i" % retval)

    # Filename prefix of segment files
    split_prefix = os.path.basename(call_path).rsplit('.', 1)[0] + '.split'
    call_dir = os.path.dirname(call_path)
    seg_paths = [os.path.join(call_dir, filename)
                 for filename in os.listdir(call_dir)
                 if filename.startswith(split_prefix)]

    if args.drop_short_segments:
        good_files = []
        for seg_path in seg_paths:
            seg_length = float(subprocess.check_output(['soxi', '-D', seg_path]))

            if seg_length < split_size:
                logging.warn("Removing short (%fs) segment %s.", seg_length, seg_path)
                os.unlink(seg_path)
            else:
                good_files.append(seg_path)

        return good_files

    return seg_paths


def extract_audio_features(audio_path, args):
    """
    Extract openSMILE (and other?) features from the audio at the given
    path. May return `None` if the provided audio is too short or
    otherwise invalid.

    openSMILE feature extraction is parameterized entirely by the
    external openSMILE config, which is provided as a command-line
    option in `args`.

    This function assumes that the openSMILE configuration file directs
    output to CSV format.

    Returns a dictionary of string keys and numeric values.
    """

    # Get a temp path for openSMILE output
    with NamedTemporaryFile() as outfile:
        try:
            retval = subprocess.call(['SMILExtract', '-C', args.opensmile_config,
                                      '-I', audio_path,
                                      '-O', outfile.name])
        except OSError, e:
            raise RuntimeError("openSMILE execution failed. Is "
                               "openSMILE (e.g., the SMILExtract "
                               "binary) on your path?", e)
        else:
            if retval != 0:
                raise RuntimeError("openSMILE error: retval %i" % retval)

            data_lines = outfile.readlines()
            if len(data_lines) == 1:
                logging.warn("Audio file at %s too short -- skipping",
                             audio_path)
                return None

            if len(data_lines) != 2:
                outfile.delete = False

                raise RuntimeError("Unexpected SMILE CSV output: we "
                                   "just want a two-line CSV. Check "
                                   "output at %s" % outfile.name)

            keys = data_lines[0].split(';')
            values = data_lines[1].split(';')

            features = {}
            for key, value in zip(keys, values):
                try:
                    value = float(value)
                except ValueError:
                    pass
                else:
                    features[key] = value

            return features


if __name__ == '__main__':
    args = parse_args()

    splits = ['train', 'devtest', 'evaltest']
    split_sizes = [.6, .2, .2] # exact numbers up for debate

    filenames = get_filenames(splits, split_sizes, args.cslu_dir, args.languages)

    for split in filenames:
        for language in filenames[split]:
            recordings = [process_recording(rec, args)
                          for rec in filenames[split][language]]

            out_path = os.path.join(args.output_directory,
                                    '%s.%s.pkl' % (language, split))
            with open(out_path, 'wb') as out_f:
                pickle.dump(recordings, out_f)

            logging.info('Wrote data for language %s, split %s to %s'
                         % (language, split, out_path))
