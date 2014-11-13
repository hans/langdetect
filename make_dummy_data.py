# This file defines some dummy data.
#
# This is a stopgap schema just so that Arthur can begin working on the
# model before Jon actually gets the data processed properly.

import pickle

from recording import Recording, Segment

segments = [
    Segment(path=None, features={'a': 1.0, 'b': 0.2}),
    Segment(path=None, features={'a': 0.3, 'b': 0.2}),
    Segment(path=None, features={'a': 0.3, 'b': 0.19}),
    Segment(path=None, features={'a': 0.4, 'b': 0.18}),
    Segment(path=None, features={'a': 0.6, 'b': 0.19}),
    Segment(path=None, features={'a': 0.5, 'b': 0.2}),
    Segment(path=None, features={'a': 0.7, 'b': 0.3}),
    Segment(path=None, features={'a': 0.5, 'b': 0.4})
]

recording = Recording('fake', segments)

if __name__ == '__main__':
    with open('data/dummy_data.pkl', 'w') as data_f:
	pickle.dump(recording, data_f)
