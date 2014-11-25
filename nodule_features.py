"""Defines nodule-level feature extractors.

All feature extractors are functions which accept arguments

    cur_segments, segments_by_feature, prev_nodule

and return a dictionary mapping from new feature keys to new feature
values.

`cur_segments` is a list of `Segment` tuples. `segments_by_feature` is a
dict mapping from feature key to list of feature values for each
segment. (`segments_by_feature` is this just a rearrangement of data in
`cur_segments`).

Ideally each feature extractor should operate independently --- i.e., it
can assume that the previous nodule has values corresponding to its own
features, but it shouldn't assume the presence of features in the nodule
that were produced by other feature extractors."""

import numpy as np
from scipy import stats


def avg_segment_features(segments, segments_by_feature, prev_nodule):
    """Build averages for each feature among the segments."""

    n_segments = float(len(segments))
    return {('avg', key): sum(segments_by_feature[key]) / n_segments
            for key in segments_by_feature}


def delta_segment_features(segments, segments_by_feature, prev_nodule):
    """Compute deltas from first to last segment for each feature."""

    return {('delta', key): segments_by_feature[key][-1] - segments_by_feature[key][0]
            for key in segments_by_feature}


def median_segment_features(segments, segments_by_feature, prev_nodule):
    return {('median', key): np.median(segments_by_feature[key])
            for key in segments_by_feature}


def std_segment_features(segments, segments_by_feature, prev_nodule):
    """Compute standard deviation among segments for each feature."""

    return {('std', key): np.std(segments_by_feature[key])
            for key in segments_by_feature}


def previous_average(segments, segments_by_feature, prev_nodule):
    """Assumes feature extractor `avg_segment_features` is enabled."""

    return {('prev avg', key): prev_nodule.features[('avg', key)]
            for key in segments[0].features}


def delta_previous_average(segments, segments_by_feature, prev_nodule):
    """Assumes feature extractor `avg_segment_features` is enabled."""

    n_segments = float(len(segments))
    return {('delta prev avg', key):
            sum(segments_by_feature[key]) / n_segments - prev_nodule.features[('avg', key)]
            for key in segments[0].features}
