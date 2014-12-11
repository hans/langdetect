"""Analyze the weights of a logistic regression model."""

import argparse
from collections import Counter
from pprint import pprint

from sklearn.externals import joblib
from sklearn.linear_model.base import LinearClassifierMixin

from nodules import Model


def main(args):
    classifier = args.model.classifier

    if not isinstance(classifier, LinearClassifierMixin):
        raise ValueError("The model provided is not a linear model. "
                         "This script is only defined for linear "
                         "models (with interpretable real-valued "
                         "coefficients associated with each model "
                         "feature).")

    assert len(classifier.coef_[0]) == len(args.model.nodule_keys)

    counter = Counter(dict(zip(args.model.nodule_keys,
                               classifier.coef_[0])))

    print '==== Most positive weights'
    pprint(counter.most_common(args.num_weights))

    print '\n\n==== Most negative weights'
    pprint(sorted(counter.items(), key=lambda i: i[1])[:args.num_weights])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=joblib.load)
    parser.add_argument('-n', '--num-weights', default=50, type=int)

    main(parser.parse_args())
