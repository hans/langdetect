# Disclaimer: this is still very rough
import pickle

from recording import Recording, Segment, Nodule
from collections import namedtuple
from sklearn import linear_model

# let noduleK be the number of segments each nodule takes in
noduleK = 3

def noduleFeatures(localFeatures, prevFeatures):
    # Takes in noduleK features from consecutive segments
    # currently, these don't talk to each other yet, i.e. we're ignoring prevFeatures
    # for the future, we have to make sure we take care of the case where prevFeatures is None
    assert len(localFeatures) == noduleK
    dictFeatures = {}
    for featureKey in localFeatures[0]:
        vals = [fDict[featureKey] for fDict in localFeatures]

        # here go all the features
        dictFeatures[('avg',featureKey)] = sum(vals)/float(noduleK)
        dictFeatures[('delta',featureKey)] = vals[-1] - vals[1]
        # TODO: insert features using prevFeatures

    return dictFeatures

def featuresToClassification(noduleFeatures):
    # dummy
    return 'Deutsch'

def createNodules(recording):
    # loop and create nodules (assume for now we're stepping one-by-one)
    recordingId, segments = recording
    nNodules = len(segments) - noduleK + 1 #number of nodules

    if nNodules == 0:
        print 'nNodules == 0', len(segments)

    noduleList = []
    for idx in range(nNodules):
        prevFeatures = noduleList[-1].features if noduleList != [] else None
        localFeatures = [segments[i].features for i in range(idx, idx + noduleK)]
        features = noduleFeatures(localFeatures, prevFeatures)
        noduleList.append(Nodule(features=features))
    return noduleList


if __name__ == '__main__':
    # Open a pickled recording
    # (this should be a list of pickled recordings in the real version)
    languages = ['ge','ma']

    #langNodules = {}
    noduleKeys = None # we need to be consistent in how we order them for the classifier
    noduleX = [] # input nodule features
    noduleY = [] # output classifications
    for lang in languages:
        with open('decoded/'+lang+'.devtest.pkl', 'r') as data_f:
            recordings = pickle.load(data_f)
        print 'unpickled',lang

        print recordings[:1]

        nodules = [createNodules(rec) for rec in recordings]

        #print nodules[:10]
        
        if noduleKeys == None and len(recordings) != 0:
            noduleKeys = sorted([key for key in nodules[0]])
        print noduleKeys

        noduleXNew = [[nodule.features[key] for key in noduleKeys]
                      for nodule in nodules]
        #print noduleXNew[0]
        print [lang]*10

        noduleX += noduleXNew
        noduleY += [lang]*len(noduleXNew)
        print 'created nodule list'
        
    logistic = linear_model.LogisticRegression(C=1e5)
    logistic.fit(noduleX, noduleY)

    print 'logistic',logistic

    with open('data/nodules.devtest.pkl', 'w') as data_f:
        pickle.dump((noduleX, noduleY), data_f)



### Scratch notes
'''
Each nodule obviously has to take into account information from its corresponding segments.
Each nodule also takes in information from the previous one (that makes sense, right? This is a little like a "bigram" model: is it expressive enough?)

Our nodules learn weights that they can use in a classification problem.
'''
