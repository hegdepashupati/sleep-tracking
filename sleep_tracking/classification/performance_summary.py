import numpy as np

class ROCPerformance(object):

    def __init__(self, false_positive_rates: np.ndarray, true_positive_rates: np.ndarray):
        self.false_positive_rates = false_positive_rates
        self.true_positive_rates = true_positive_rates


class ClassificationPerformanceSummary(object):
    def __init__(self, accuracy, wake_correct, rem_correct, nrem_correct, kappa):
        self.accuracy = accuracy
        self.wake_correct = wake_correct
        self.rem_correct = rem_correct
        self.nrem_correct = nrem_correct
        self.kappa = kappa
