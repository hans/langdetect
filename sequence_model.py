from recording import Recording, Segment


def generate_training_examples(recordings):
    """
    Generate a collection of training examples for the given raw
    training data.

    Our classifier makes a decision at each nodule in a sequence of
    nodules. From each recording we can generate an example for a given
    nodule given the past nodule features.
    """

    # TODO
    pass


def train_model(recordings):
    data = generate_training_examples(recordings)

    # TODO train something!


def classify_recording(model, recording):
    # Accumulate predictions for each nodule
    predictions = []
    previous_features = {}

    for segment_group in make_windows(recording.segments):
        # TODO generate features for given segment group, along with
        # previous features

        # TODO do classification with trained model; accumulate
        # prediction as a vote?
        pass

    # return final prediction after examining sequence of predictions
    return None
