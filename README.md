## Installation

Clone this repo, then run:

    # Set up virtualenv
    virtualenv . && source bin/activate

    # Install package requirements
    pip install -r requirements.txt

To build the training data (with `prepare.py`), you'll need the following
packages:

- [SPHERE][2], for decoding from the OGI sound formats
- [SoX][3], for sound preprocessing
- [openSMILE][4], for audio feature extraction

The binaries from these pacakages need to be on your `PATH` for the corpus
handling scripts to work properly.

## Data

We use the [OGI Multilanguage Corpus][1] for training and evaluation. You can
acquire a copy of this corpus through the LDC.

With a fresh download of the OGI corpus, run `prepare.py` to get the corpus
into a format usable with this software:

    python prepare.py <OGI_PATH> ge,ma

In this command the argument `ge,ma` specifies that we'd like to perform
language detection with German and Mandarin recordings.

[1]: https://catalog.ldc.upenn.edu/LDC94S17
[2]: http://www.nist.gov/itl/iad/mig/tools.cfm
[3]: http://sox.sourceforge.net/
[4]: http://opensmile.sourceforge.net/
