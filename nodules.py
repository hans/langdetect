from __future__ import print_function
import argparse
import ConfigParser
import cPickle as pickle
from functools import partial
import itertools
import random
import sys
import time

import numpy as np
from sklearn import (# models
                     linear_model, svm,
                     # training utilities
                     metrics, cross_validation, grid_search,
                     # preprocessing
                     decomposition, preprocessing)
from sklearn.externals import joblib

from recording import Recording, Segment, Nodule
from collections import Counter, namedtuple
import nodule_features


class Model(object):
    """Defines a language detection model (mostly for serialization
    purposes)."""

    def __init__(self, languages, classifier, nodule_size, feature_extractors,
                 nodule_keys, transformers):
        """Create a model for saving / loading.

        `feature_extractors` is a list of feature extractor functions as
        defined in the `nodule_features` module.

        `nodule_keys` is a list of keys to extract from the feature
        dictionary and provide to the classifier. The order of this list
        matches the order of features expected by the classifier."""

        self.languages = languages

        self.classifier = classifier

        self.nodule_size = nodule_size
        self.feature_extractors = feature_extractors
        self.nodule_keys = nodule_keys

        self.transformers = transformers

    def _make_example(self, nodule):
        data = [nodule.features[key] for key in self.nodule_keys]

        # Backwards-compatible with old code (pre-transformers)
        if hasattr(self, 'transformers'):
            for transformer in self.transformers:
                data = transformer.transform(data)

        return data

    def classify_nodule(self, nodule):
        return self.classifier.predict(self._make_example(nodule))[0]

    def class_confidences(self, nodule):
        """Return a list of confidences for each class given the nodule
        data."""

        confidences = self.classifier.decision_function(self._make_example(nodule))

        # scikit-learn linear models do this weird thing for n_classes = 2
        # where they return a single confidence score.. need to fix that
        if len(confidences) == 1:
            confidence = confidences[0]
            if confidence > 0:
                return np.array([0, confidence])
            else:
                return np.array([-confidence, 0])

        return confidences

    def compatible_with(self, other_model):
        """Determine if this model is compatible with another model for
        ensembling purposes (i.e., if the two can be successfully
        ensembled at test time)."""

        return ([lang.lower() for lang in self.languages] == [lang.lower() for lang in other_model.languages]
                and self.nodule_size == other_model.nodule_size
                and self.feature_extractors == other_model.feature_extractors)


def makeNodule(segments, prev_nodule, feature_extractors, nodule_size):
    """Create a new `Nodule` from the provided segments and given
    nodule history.

    `features` is a list of feature extractors (as defined in the
    `nodule_features` module)."""

    # TODO: remove assertion
    assert len(segments) == nodule_size

    # Build a more convenient structure for lookup: a map
    #
    #     fname -> [seg0[fname], seg1[fname], ...]
    segments_by_feature = {key: [segment.features[key] for segment in segments]
                           for key in segments[0].features}

    # TODO: after deadline, do a better job for when when prevNodules is None
    if prev_nodule is None:
        # Build a dummy nodule with all features equal to zero.
        prev_nodule = Nodule(features=Counter())

    noduleFeatures = {}

    for extractor in feature_extractors:
        noduleFeatures.update(extractor(segments, segments_by_feature, prev_nodule))

    return Nodule(features=noduleFeatures)


def classifyRecording(models, languages, recording, args):
    """
    Use (a) trained model(s) to classify the given recording. The
    provided models must be compatible (i.e., m1.compatible_with(m2) for
    all m1, m2).

    Returns the index of the most likely language. (Index into
    `languages`.)
    """

    # Models all share feature extractors + nodule size (compatible by
    # contract)
    feature_extractors = models[0].feature_extractors
    nodule_size = models[0].nodule_size

    nodules = createNodules(recording, feature_extractors, nodule_size)

    votes = np.zeros(len(languages))
    for nodule in nodules:
        for model_weight, model in zip(args.model_weights, models):
            if args.soft_votes:
                # confidences of each class applying to the given nodule
                lang_confidences = model.class_confidences(nodule)
                votes += model_weight * lang_confidences
            else:
                decision = model.classify_nodule(nodule)
                votes[decision] += model_weight

    return np.argmax(votes)


def createNodules(recording, feature_extractors, nodule_size):
    # loop and create nodules (assume for now we're stepping one-by-one)
    recordingId, segments = recording
    nNodules = len(segments) - nodule_size + 1 #number of nodules

    # Temporary fix: if can't form even a single nodule, repeat last segment
    # TODO: after milestone, find a better solution
    if nNodules <= 0:
        if len(segments) == 0:
            # Really in trouble now.
            return []

        while len(segments) != nodule_size:
            segments.append(segments[-1])

        return [makeNodule(segments, None, feature_extractors, nodule_size)]

    noduleList = []
    prevNodule = None
    for idx in range(nNodules):
        nodule = makeNodule(segments[idx:idx + nodule_size], prevNodule,
                            feature_extractors, nodule_size)
        noduleList.append(nodule)

        prevNodule = nodule

    return noduleList


CLASSIFIER_TYPES = {
    'logistic': partial(linear_model.LogisticRegression, C=1e5),
    'svm': svm.SVC,
}


def train(args, do_grid_search=False):
    noduleKeys = None # we need to be consistent in how we order them for the classifier
    noduleX = [] # input nodule features
    noduleY = [] # output classifications

    train_path = '%s/%%s.train.pkl' % args.data_dir

    # Synthesize training examples
    for langIndex, lang in enumerate(args.languages):
        with open(train_path % lang, 'rb') as data_f:
            recordings = pickle.load(data_f)
        print('unpickled', lang)

        # Build training data: just a big collection of nodules (not
        # grouped by recording)
        nodules = []
        for recording in recordings:
            nodules.extend(createNodules(recording, args.feature_extractors,
                                         args.nodule_size))

        if noduleKeys == None and len(recordings) != 0:
            noduleKeys = sorted([key for key in nodules[0].features])

        # Training set is just this standard feature set for every
        # nodule
        noduleXNew = [[nodule.features[key] for key in noduleKeys]
                      for nodule in nodules]

        #print(noduleXNew[0])

        noduleX.extend(noduleXNew)

        # Labels for this language
        noduleY.extend([langIndex] * len(noduleXNew))

        print('created nodules for', lang)

    print('Normalizing all examples and all features (%i examples, %i features)..'
          % (len(noduleX), len(noduleX[0])))
    scaler = preprocessing.StandardScaler().fit(noduleX)
    noduleX = scaler.transform(noduleX)

    transformers = [scaler]

    if args.pca is not None:
        print('Using PCA to reduce data to %i components' % args.pca)
        pca = decomposition.PCA(n_components=args.pca, copy=False)
        noduleX = pca.fit_transform(noduleX)
        print('Design matrix is now ', noduleX.shape)

        transformers.append(pca)

    if do_grid_search:
        print('Performing grid search on logistic regression '
              'regularization parameter')

        classifier = logistic_grid_search(noduleX, noduleY)
        model_path = ('data/model.logistic.gridsearch.%s.pkl'
                      % time.strftime('%Y%m%d-%H%M%S'))

        with open(model_path, 'wb') as model_f:
            model = Model(languages=args.languages,
                          classifier=classifier,
                          nodule_size=args.nodule_size,
                          feature_extractors=args.feature_extractors,
                          nodule_keys=noduleKeys,
                          transformers=transformers)

            pickle.dump(model, model_f)

        print('Saved grid-search model to %s.' % model_path)
        return


    for classifier_name, classifier_class in CLASSIFIER_TYPES.items():
        print('Training model %s on %i examples..' % (classifier_name, len(noduleX)))
        classifier = classifier_class()
        classifier.fit(noduleX, noduleY)

        print('\t', classifier)

        model_path = '%s/model.%s.%s%s.jbl' % (args.model_out_dir,
                                               classifier_name,
                                               time.strftime('%Y%m%d-%H%M%S'),
                                               random.randint(0, 1000))

        model = Model(languages=args.languages,
                      classifier=classifier,
                      nodule_size=args.nodule_size,
                      feature_extractors=args.feature_extractors,
                      nodule_keys=noduleKeys,
                      transformers=transformers)

        joblib.dump(model, model_path)

        print('Saved model to %s.' % model_path)


def logistic_grid_search(X, y, cross_validation_proportion=0.25):
    """Perform grid search to find the best regularization parameter
    for the given data.

    Will partition the data and train on one split, cross-validate on
    other. (The data provided should be drawn exclusively from a
    training set.) `cross_validation_proportion` specifies the
    proportion of the provided data which should be held out for
    cross-validation.

    Returns the optimal logistic regression classifier.
    """

    X_train, X_val, y_train, y_val = cross_validation.train_test_split(
        X, y, test_size=cross_validation_proportion)

    parameters = {'C': [0.1, 1, 10, 100, 1000, 10000]}
    classifier = grid_search.GridSearchCV(linear_model.LogisticRegression(C=1),
                                          parameters)
    classifier.fit(X_val, y_val)

    print("Best parameters found on validation set:\n%s", classifier.best_estimator_)
    print("Grid scores:\n%s", classifier.grid_scores_)

    y_pred = classifier.predict(X_val)
    print("Classification report for classifier trained on validation set:")
    print(evaluate(y_val, y_pred))

    print("Now retraining on training set.")
    classifier = classifier.best_estimator_.fit(X_train, y_train)
    print('\t', classifier)

    return classifier


def evaluate(golds, guesses):
    confusion = metrics.confusion_matrix(golds, guesses)
    report = metrics.classification_report(golds, guesses)

    return "Confusion matrix:\n\n%s\n\n%s" % (confusion, report)


def test(models, languages, args):
    if args.test_on_train:
        dev_path = '%s/%%s.train.pkl' % args.data_dir
    else:
        dev_path = '%s/%%s.devtest.pkl' % args.data_dir

    golds, guesses = [], []
    for langIndex, lang in enumerate(languages):
        with open(dev_path % lang, 'rb') as data_f:
            recordings = pickle.load(data_f)

        for recording in recordings:
            guess = classifyRecording(models, languages, recording, args)

            if args.verbose:
                result = 'RIGHT' if guess == langIndex else 'WRONG'
                print('{}\t{}'.format(result, recording.id))

            golds.append(langIndex)
            guesses.append(guess)

    print(evaluate(golds, guesses))


def load_model(path):
    if path.endswith('jbl'):
        return joblib.load(path)
    else:
        with open(path, 'rb') as model_f:
            return pickle.load(model_f)


def load_models(paths):
    models = [load_model(path) for path in paths]

    # Verify that models are pairwise compatible
    for model1, model2 in itertools.combinations(models, 2):
        if not model1.compatible_with(model2):
            raise ValueError("Provided models not compatible.\n\t"
                             "Model 1: %s\n\n\tModel 2: %s"
                             % (model1, model2))

    return models


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

    parser.add_argument('mode', choices=['train', 'test', 'grid_search'],
                        help=('Program mode.'))

    parser.add_argument('-d', '--data-dir',
                        help=('Directory containing preprocessed data '
                              '(as output by `prepare` module)'))

    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('--soft-votes', default=False, action='store_true',
                        help=('Aggregate "soft votes" (confidences) per '
                              'nodule'))
    parser.add_argument('--test-on-train', default=False, action='store_true',
                        help=('Test on training data'))

    model_options = parser.add_mutually_exclusive_group(required=True)
    model_options.add_argument('--model-out-dir',
                               help=('Directory to which model files '
                                     'should be saved (training only)'))
    model_options.add_argument('--model-in-file', action='append',
                               help=('Trained model file to use for '
                                     'testing. If multiple models are '
                                     'provided, they will be ensembled '
                                     'during testing (must have same '
                                     'feature sets)'))
    model_options.add_argument('--model-weights',
                               type=lambda s: [float(f) for f in s.split(',')],
                               help=('Comma-separated list of float '
                                     'model weights for adjusting '
                                     'per-model votes'))

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
    train_options.add_argument('--feature-extractors',
                               default=[nodule_features.avg_segment_features,
                                        nodule_features.delta_segment_features,
                                        nodule_features.previous_average],
                               type=lambda fs_str: [getattr(nodule_features, f) for f in fs_str.split(',')],
                               help=('Comma-separated list of feature '
                                     'extractors to use'))

    if defaults is not None:
        parser.set_defaults(**defaults)

    args = parser.parse_args(remaining_argv)

    # Validate arguments
    if args.mode in ['train', 'grid_search']:
        if args.languages is None:
            raise ValueError('--languages option required for training '
                             '(see --help)')

    if args.model_weights is None:
        args.model_weights = np.ones(len(args.model_in_file))
    elif len(args.model_weights) != len(args.model_in_file):
        raise ValueError("Number of model weights provided must match "
                         "number of models")

    ### Launch

    if args.mode == 'test':
        models = load_models(args.model_in_file)
        test(models, models[0].languages, args)
    else:
        train(args, args.mode == 'grid_search')



### Scratch notes
'''
Each nodule obviously has to take into account information from its corresponding segments.
Each nodule also takes in information from the previous one (that makes sense, right? This is a little like a "bigram" model: is it expressive enough?)

Our nodules learn weights that they can use in a classification problem.
'''
