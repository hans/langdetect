# Defines the Recording type.

from collections import namedtuple

Recording = namedtuple('Recording', ['id', 'segments'])

# Defines a particular segment of a call recording. Segments are usually
# fixed-length; a recording can have an arbitrary number of segments.
Segment = namedtuple('Segment', ['path', 'features'])

# Defines a "nodule" that estimates language probabilities based on
# features from a local set of (say 3) nodes and the nodule before it
Nodule = namedtuple('Nodule', ['features'])
