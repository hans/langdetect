# Defines the Recording type.

from collections import namedtuple

Recording = namedtuple('Recording', ['id', 'segments'])
Segment = namedtuple('Segment', ['features'])
