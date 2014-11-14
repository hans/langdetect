import cPickle as pickle
import sys

from recording import Recording, Segment, Nodule
from collections import Counter, namedtuple
from sklearn import linear_model

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
    else:
        print type(prevNodule)
        print prevNodule.key
        print prevNodule.features.keys()

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
    return model.predict(example)


def classifyRecording(model, recording):
    """
    Use a trained model to classify the given recording.
    """

    nodules = createNodules(recording)

    votes = Counter()
    for nodule in nodules:
        noduleVote = classifyNodule(model, nodule)
        votes[noduleVote] += 1

    return votes.most_common()[1]


def createNodules(recording):
    # loop and create nodules (assume for now we're stepping one-by-one)
    recordingId, segments = recording
    nNodules = len(segments) - noduleK + 1 #number of nodules

    # Temporary fix: if can't form even a single nodule, repeat last segment
    # TODO: after milestone, find a better solution
    if nNodules == 0:
        while len(segments) != noduleK:
            segments.append(segments[-1])

        return [makeNodule(segments, None)]

    noduleList = []
    prevNodule = None
    for idx in range(nNodules):
        print 'prevNodule: type',type(prevNodule)
        nodule = makeNodule(segments[idx:idx + noduleK], prevNodule)
        print type(nodule)
        noduleList.append(nodule)

        prevNodule = nodule

    return noduleList


def train(languages):
    noduleKeys = None # we need to be consistent in how we order them for the classifier
    noduleX = [] # input nodule features
    noduleY = [] # output classifications

    train_path = 'decoded/%s.train.pkl'
    for lang in languages:
        with open(train_path % lang, 'r') as data_f:
            recordings = pickle.load(data_f)
        print 'unpickled',lang

        # Build training data: just a big collection of nodules (not
        # grouped by recording)
        nodules = []
        for recording in recordings:
            nodules.extend(createNodules(recording))

        print nodules[1]
        print type(nodules[1])

        if noduleKeys == None and len(recordings) != 0:
            noduleKeys = sorted([key for key in nodules[0].features])
        print 'nodule features: ', noduleKeys

        # Training set is just this standard feature set for every
        # nodule
        noduleXNew = [[nodule.features[key] for key in noduleKeys]
                      for nodule in nodules]

        #print noduleXNew[0]

        noduleX.extend(noduleXNew)

        # Labels for this language
        noduleY.extend([lang] * len(noduleXNew))

        print 'created nodules for', lang

    logistic = linear_model.LogisticRegression(C=1e5)
    logistic.fit(noduleX, noduleY)

    print 'logistic', logistic

    model_path = 'data/model.logistic.pkl'
    with open(model_path, 'w') as data_f:
        pickle.dump(logistic, data_f)

    print 'Saved model to %s.' % model_path


def test(model, languages):
    dev_path = 'decoded/%s.devtest.pkl'

    for lang in languages:
        with open(dev_path % lang, 'r') as data_f:
            recordings = pickle.load(data_f)

        for recording in recordings:
            nodules = createNodules(recording)
            guess = classifyRecording(model, recording)

            print 'guess', guess, 'gold', lang



if __name__ == '__main__':
    languages = ['ge', 'ma']

    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as model_f:
            model = pickle.load(model_f)

        # Model provided -- test.
        test(model, languages)
    else:
        train(languages)



### Scratch notes
'''
Each nodule obviously has to take into account information from its corresponding segments.
Each nodule also takes in information from the previous one (that makes sense, right? This is a little like a "bigram" model: is it expressive enough?)

Our nodules learn weights that they can use in a classification problem.
'''
