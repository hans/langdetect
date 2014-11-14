# Disclaimer: this is still very rough

import pickle

from recording import Recording, Segment, Nodule

from collections import namedtuple

with open('data/dummy_data.pkl', 'r') as data_f:
    recording = pickle.load(data_f)

recordingId, segments = recording

segmentFeatures = [key for key in segments[0].features]

# let K be the number of segments each nodule takes in
noduleK = 3
nNodules = len(segments) - noduleK + 1

# list of features
def noduleFeatures(featureDicts, prevFeatures):
    # takes in noduleK features from consecutive segments
    # assuming "dense" dict of features (or use of Counters)
    # currently, these don't talk to each other yet, i.e. we're ignoring prevFeatures
    # for the future, we have to make sure we take care of the case where prevFeatures is None
    assert len(featureDicts) == noduleK
    dictFeatures = {}
    for featureKey in segmentFeatures:
        vals = [fDict[featureKey] for fDict in featureDicts]
        # here go all the features
        dictFeatures[('avg',featureKey)] = sum(vals)/float(noduleK)
        print 'dictFeatures',dictFeatures

    return dictFeatures

def featuresToClassification(noduleFeatures):
    # dummy
    return 'Deutsch'

# loop and create nodules
noduleList = []
for idx in range(nNodules):
    print idx
    prevFeatures = noduleList[-1].features if noduleList != [] else None
    featureDicts = [segments[i].features for i in range(idx, idx + noduleK)]
    print 'featureDicts',featureDicts
    features = noduleFeatures(featureDicts, prevFeatures)
    noduleList.append(Nodule(features=features))

print noduleList

with open('data/dummy_nodules.pkl', 'w') as data_f:
    pickle.dump(noduleList, data_f)


### Scratch notes
'''
Each nodule obviously has to take into account information from its corresponding segments.
Each nodule also takes in information from the previous one (that makes sense, right? This is a little like a "bigram" model: is it expressive enough?)

Our nodules learn weights that they can use in a classification problem.
'''
