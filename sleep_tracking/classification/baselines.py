from sleep_tracking.classification.data_loader import SleepDataset

import time
import numpy as np
from sklearn.utils import compute_class_weight
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RandomForest

__all__ = ['RandomForestClassifier', 'LogisticRegressionClassifier']


class BaselineClassifier(object):
    def __init__(self, classifier, param_grid):
        self.classifier = classifier
        self.param_grid = param_grid

    def fit(self, training_x: np.ndarray, training_y: np.ndarray, scoring='neg_log_loss'):
        print("Training started ...")
        start_time = time.time()
        self.classifier.class_weight = self._compute_class_weight_dictionary(training_y)
        best_parameters = self._gridsearch(training_x, training_y, scoring)
        self.classifier.set_params(**best_parameters)
        self.classifier.fit(training_x, training_y)
        print("Training completed in {:.3f} min".format((time.time() - start_time) / 60.))

    def _gridsearch(self, training_x, training_y, scoring):
        grid_search = GridSearchCV(self.classifier, self.param_grid, scoring=scoring, cv=3)
        grid_search.fit(training_x, training_y)
        return grid_search.best_params_

    @staticmethod
    def _compute_class_weight_dictionary(training_y):
        classes = np.unique(training_y)
        class_weight = compute_class_weight("balanced", classes=classes, y=training_y)
        class_weight_dict = dict(zip(classes, class_weight))
        return class_weight_dict


class RandomForestClassifier(BaselineClassifier):
    def __init__(self):
        classifier = RandomForest(n_estimators=500, max_features=1.0, max_depth=10,
                                  min_samples_split=10, min_samples_leaf=1)
        param_grid = {'max_depth': [10, 50, 100]}
        super(RandomForestClassifier, self).__init__(classifier, param_grid)


class LogisticRegressionClassifier(BaselineClassifier):
    def __init__(self):
        classifier = LogisticRegression(penalty='l1', solver='liblinear', verbose=0)
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
        super(LogisticRegressionClassifier, self).__init__(classifier, param_grid)
