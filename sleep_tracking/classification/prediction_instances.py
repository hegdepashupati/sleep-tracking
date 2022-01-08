class PredictionInstances(object):
    def __init__(self, true_labels, predicted_labels, predicted_probs):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        self.predicted_probs = predicted_probs
