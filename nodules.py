import argparse
import ConfigParser
import cPickle as pickle
from functools import partial
import sys

from sklearn import linear_model, svm, metrics, preprocessing

from recording import Recording, Segment, Nodule
from collections import Counter, namedtuple


# let noduleK be the number of segments each nodule takes in
noduleK = 3

def makeNodule(segments, prevNodule):
    # Takes in noduleK features from consecutive segments
    # currently, these don't talk to each other yet, i.e. we're ignoring prevFeatures
    # for the future, we have to make sure we take care of the case where prevFeatures is None
    assert len(segments) == noduleK

    # TODO: after deadline, do a better job for when when prevNodules is None
    if prevNodule is None:
        prevNodule = Nodule(features = Counter()) #i.e. assume everything is 0

    noduleFeatures = {}
    for featureKey in segments[0].features:
        featureValues = [segment.features[featureKey] for segment in segments]

        # here go all the features
        noduleFeatures[('avg', featureKey)] = sum(featureValues) / float(noduleK)
        noduleFeatures[('delta', featureKey)] = featureValues[-1] - featureValues[1]

        # insert features using prevFeatures
        #print ('avg',featureKey) in prevNodule.features, prevNodule.features[('avg',featureKey)]
        noduleFeatures[('prev avg', featureKey)] = prevNodule.features[('avg',featureKey)]

    return Nodule(features=noduleFeatures)


def classifyNodule(model, nodule):
    # Extract features in the same order used during training
    keys = sorted(nodule.features.keys())

    example = [nodule.features[key] for key in keys]
    return model.predict(example)[0]


def classifyRecording(model, recording):
    """
    Use a trained model to classify the given recording.
    """

    nodules = createNodules(recording)

    votes = Counter()
    for nodule in nodules:
        noduleVote = classifyNodule(model, nodule)
        votes[noduleVote] += 1

    return votes.most_common(1)[0]


def createNodules(recording):
    # loop and create nodules (assume for now we're stepping one-by-one)
    recordingId, segments = recording
    nNodules = len(segments) - noduleK + 1 #number of nodules

    # Temporary fix: if can't form even a single nodule, repeat last segment
    # TODO: after milestone, find a better solution
    if nNodules <= 0:
        while len(segments) != noduleK:
            segments.append(segments[-1])

        return [makeNodule(segments, None)]

    noduleList = []
    prevNodule = None
    for idx in range(nNodules):
        nodule = makeNodule(segments[idx:idx + noduleK], prevNodule)
        noduleList.append(nodule)

        prevNodule = nodule

    return noduleList


MODEL_TYPES = {
    'logistic': partial(linear_model.LogisticRegression, C=1e5),
    'svm': svm.SVC,
}


def train(languages):
    # Sort language labels so we know outputs are consistent among train, test
    languages.sort()

    noduleKeys = None # we need to be consistent in how we order them for the classifier
    noduleX = [] # input nodule features
    noduleY = [] # output classifications

    train_path = 'decoded/%s.train.pkl'
    for langIndex, lang in enumerate(languages):
        with open(train_path % lang, 'r') as data_f:
            recordings = pickle.load(data_f)
        print 'unpickled',lang

        # Build training data: just a big collection of nodules (not
        # grouped by recording)
        nodules = []
        for recording in recordings:
            nodules.extend(createNodules(recording))

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
    print 'Now %i examples, %i features' % (len(noduleX), len(noduleX[0]))

    for model_type, model_class in MODEL_TYPES.items():
        print 'Training model %s on %i examples..' % (model_type, len(noduleX))
        model = model_class()
        model.fit(noduleX, noduleY)

        print model

        model_path = 'data/model.%s.pkl' % model_type
        with open(model_path, 'w') as data_f:
            pickle.dump(model, data_f)

        print 'Saved model to %s.' % model_path


def evaluate(golds, guesses):
    # # inputs as lists of 0s or 1s
    # # return accuracy, precision, F1
    # assert len(guesses) == len(golds)
    # nLangs = 2 # could eventually increase this

    return metrics.classification_report(golds, guesses)

    # # build confusion matrix [guess][gold]
    # confusionMat = [[0 for j in range(nLangs)] for i in range(nLangs)]
    # for guess, gold in zip(guesses,golds):
    #     confusionMat[guess][gold] += 1

    # nCorrect = sum([confusionMat[i][i] for i in range(nLangs)])
    # accuracy = nCorrect/float(len(guesses))

    # return (confusionMat, accuracy) # returns a confusion matrix




def test(model, languages):
    # Sort language labels so we know outputs are consistent among train, test
    languages.sort()

    dev_path = 'decoded/%s.devtest.pkl'

    golds, guesses = [], []
    for langIndex, lang in enumerate(languages):
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
    parser.add_argument('languages', type=lambda s: s.split(','),
                        help=('Comma-separated list of first two '
                              'letters of names of each language to '
                              'retain'))

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

    args = parser.parse_args(remaining_argv)

    if args.mode == 'test':
        with open(args.model_in_file, 'r') as model_f:
            model = pickle.load(model_f)

        # Model provided -- test.
        test(model, args.languages)
    else:
        train(args.languages)



### Scratch notes
'''
Each nodule obviously has to take into account information from its corresponding segments.
Each nodule also takes in information from the previous one (that makes sense, right? This is a little like a "bigram" model: is it expressive enough?)

Our nodules learn weights that they can use in a classification problem.
'''
