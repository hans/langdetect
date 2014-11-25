import argparse
import ConfigParser
import cPickle as pickle
from functools import partial
import sys
import time

from sklearn import linear_model, svm, metrics, decomposition, preprocessing

from recording import Recording, Segment, Nodule
from collections import Counter, namedtuple
import nodule_features


class Model(object):
    """Defines a language detection model (mostly for serialization
    purposes)."""

    def __init__(self, languages, classifier, feature_extractors, nodule_keys):
        """Create a model for saving / loading.

        `feature_extractors` is a list of feature extractor functions as
        defined in the `nodule_features` module.

        `nodule_keys` is a list of keys to extract from the feature
        dictionary and provide to the classifier. The order of this list
        matches the order of features expected by the classifier."""

        self.languages = languages

        self.classifier = classifier

        self.feature_extractors = feature_extractors
        self.nodule_keys = nodule_keys

    def classify_nodule(self, nodule):
        # Extract proper keys from nodule
        example = [nodule.features[key] for key in self.nodule_keys]
        return self.classifier.predict(example)[0]


# let noduleK be the number of segments each nodule takes in
noduleK = 3

def makeNodule(segments, prevNodule, feature_extractors, args):
    """Create a new `Nodule` from the provided segments and given
    nodule history.

    `features` is a list of feature extractors (as defined in the
    `nodule_features` module)."""

    # TODO: remove assertion
    assert len(segments) == args.nodule_size

    # Build a more convenient structure for lookup: a map
    #
    #     fname -> [seg0[fname], seg1[fname], ...]
    segments_by_feature = {key: [segment.features[key] for segment in segments]
                           for key in segments[0].features}

    # TODO: after deadline, do a better job for when when prevNodules is None
    if prevNodule is None:
        # Build a dummy nodule with all features equal to zero.
        prevNodule = Nodule(features=Counter())

    noduleFeatures = {}

    for extractor in feature_extractors:
        noduleFeatures.update(extractor(segments, segments_by_feature, prev_nodule))

    return Nodule(features=noduleFeatures)


def classifyRecording(model, recording):
    """
    Use a trained model to classify the given recording.
    """

    nodules = createNodules(recording)

    votes = Counter()
    for nodule in nodules:
        noduleVote = model.classify_nodule(nodule)
        votes[noduleVote] += 1

    return votes.most_common(1)[0]


def createNodules(recording, feature_extractors, args):
    # loop and create nodules (assume for now we're stepping one-by-one)
    recordingId, segments = recording
    nNodules = len(segments) - args.nodule_size + 1 #number of nodules

    # Temporary fix: if can't form even a single nodule, repeat last segment
    # TODO: after milestone, find a better solution
    if nNodules <= 0:
        while len(segments) != args.nodule_size:
            segments.append(segments[-1])

        return [makeNodule(segments, None, feature_extractors, args)]

    noduleList = []
    prevNodule = None
    for idx in range(nNodules):
        nodule = makeNodule(segments[idx:idx + noduleK], prevNodule, feature_extractors, args)
        noduleList.append(nodule)

        prevNodule = nodule

    return noduleList


CLASSIFIER_TYPES = {
    'logistic': partial(linear_model.LogisticRegression, C=1e5),
    'svm': svm.SVC,
}


def train(args):
    noduleKeys = None # we need to be consistent in how we order them for the classifier
    noduleX = [] # input nodule features
    noduleY = [] # output classifications

    train_path = '%s/%%s.train.pkl' % args.data_dir

    # TODO don't set in stone this configuration!
    feature_extractors = [nodule_features.avg_segment_features,
                          nodule_features.delta_segment_features,
                          nodule_features.previous_average]

    # Synthesize training examples
    for langIndex, lang in enumerate(args.languages):
        with open(train_path % lang, 'r') as data_f:
            recordings = pickle.load(data_f)
        print 'unpickled',lang

        # Build training data: just a big collection of nodules (not
        # grouped by recording)
        nodules = []
        for recording in recordings:
            nodules.extend(createNodules(recording, feature_extractors, args))

        if noduleKeys == None and len(recordings) != 0:
            noduleKeys = sorted([key for key in nodules[0].features])

        # Training set is just this standard feature set for every
        # nodule
        noduleXNew = [[nodule.features[key] for key in noduleKeys]
                      for nodule in nodules]

        #print noduleXNew[0]

        noduleX.extend(noduleXNew)

        # Labels for this language
        noduleY.extend([langIndex] * len(noduleXNew))

        print 'created nodules for', lang

    print ('Normalizing all examples and all features (%i examples, %i features)..'
           % (len(noduleX), len(noduleX[0])))
    noduleX = preprocessing.Normalizer().fit_transform(noduleX)

    if args.pca is not None:
        print 'Using PCA to reduce data to %i components' % args.pca
        pca = decomposition.PCA(n_components=args.pca, copy=False)
        noduleX = pca.fit_transform(noduleX)
        print 'Design matrix is now ', noduleX.shape

    for classifier_name, classifier_class in CLASSIFIER_TYPES.items():
        print 'Training model %s on %i examples..' % (classifier_name, len(noduleX))
        classifier = classifier_class()
        classifier.fit(noduleX, noduleY)

        print '\t', classifier

        model_path = 'data/model.%s.%s.pkl' % (classifier_name,
                                               time.strftime('%Y%m%d-%H%M%S'))
        with open(model_path, 'w') as data_f:
            model = Model(languages, classifier, feature_extractors, noduleKeys)
            pickle.dump(model, data_f)

        print 'Saved model to %s.' % model_path


def evaluate(golds, guesses):
    # TODO build confusion matrix as well?

    return metrics.classification_report(golds, guesses)


def test(model, args):
    dev_path = '%s/%%s.devtest.pkl' % args.data_dir

    golds, guesses = [], []
    for langIndex, lang in enumerate(model.languages):
        with open(dev_path % lang, 'r') as data_f:
            recordings = pickle.load(data_f)

        for recording in recordings:
            nodules = createNodules(recording)
            guess = classifyRecording(model, recording)[0]

            golds.append(langIndex)
            guesses.append(guess)

    print evaluate(golds, guesses)

if __name__ == '__main__':
    # Build an ArgumentParser just for matching config file arguments
    conf_section = 'Main'
    conf_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False)
    conf_parser.add_argument('-c', '--config-file', metavar='FILE',
                             help=('Path to configuration file, which '
                                   'has keys which match possible '
                                   'long-form argument names of the '
                                   'program (see --help). Properties '
                                   'should be under a section named '
                                   '[%s].' % conf_section))

    # Try to grab just the config file param; leave rest untouched
    args, remaining_argv = conf_parser.parse_known_args()

    defaults = None
    if args.config_file:
        config = ConfigParser.SafeConfigParser()
        config.read([args.config_file])
        defaults = dict(config.items(conf_section))

    # Parse rest of arguments
    parser = argparse.ArgumentParser(parents=[conf_parser])

    parser.add_argument('mode', choices=['train', 'test'],
                        help=('Program mode. Different options apply to '
                              'each mode -- see below.'))

    parser.add_argument('-d', '--data-dir',
                        help=('Directory containing preprocessed data '
                              '(as output by `prepare` module)'))

    model_options = parser.add_mutually_exclusive_group(required=True)
    model_options.add_argument('--model-out-dir',
                               help=('Directory to which model files '
                                     'should be saved (training only)'))
    model_options.add_argument('--model-in-file',
                               help=('Trained model file to use for '
                                     'testing'))

    train_options = parser.add_argument_group('Training options')
    train_options.add_argument('-l', '--languages', type=lambda s: s.split(','),
                               help=('Comma-separated list of first two '
                                     'letters of names of each language '
                                     'to retain'))
    train_options.add_argument('--pca', type=int,
                               help=('Run PCA on training examples '
                                     'before training, retaining `n` '
                                     'components'))
    train_options.add_argument('--nodule-size', type=int, default=3,
                               help=('Number of segments which each '
                                     'nodule should cover'))

    args = parser.parse_args(remaining_argv)

    # Validate arguments
    if args.mode == 'train':
        if args.languages is None:
            raise ValueError('--languages option required for training '
                             '(see --help)')

    ### Launch

    if args.mode == 'test':
        with open(args.model_in_file, 'r') as model_f:
            model = pickle.load(model_f)

        # Model provided -- test.
        test(model, args)
    else:
        train(args)



### Scratch notes
'''
Each nodule obviously has to take into account information from its corresponding segments.
Each nodule also takes in information from the previous one (that makes sense, right? This is a little like a "bigram" model: is it expressive enough?)

Our nodules learn weights that they can use in a classification problem.
'''
