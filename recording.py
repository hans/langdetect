# Defines the Recording type.

from collections import namedtuple

Recording = namedtuple('Recording', ['id', 'segments'])

# Defines a particular segment of a call recording. Segments are usually
# fixed-length; a recording can have an arbitrary number of segments.
Segment = namedtuple('Segment', ['path', 'features'])
