from sleep_tracking.classification.data_split import SubjectSplits

import numpy as np
from sklearn.utils import compute_class_weight


def compute_class_weights(data_labels):
    class_labels = np.unique(data_labels)
    class_weights = compute_class_weight("balanced", classes=class_labels, y=data_labels)
    return class_weights.astype(np.float32)


def split_subjects_into_train_test_validation(subject_ids, test_fraction, validation_fraction, num_splits, seed=121):
    rng = np.random.RandomState(seed)
    num_testing_subjects = int(np.round(test_fraction * len(subject_ids)))
    num_validation_subjects = int(np.round(validation_fraction * len(subject_ids)))

    splits = []
    for _ in range(num_splits):
        random_choices = rng.permutation(len(subject_ids))
        testing_set = [subject_ids[idx] for idx in
                       random_choices[:num_testing_subjects]]
        validation_set = [subject_ids[idx] for idx in
                          random_choices[num_testing_subjects:(num_testing_subjects + num_validation_subjects)]]
        training_set = [subject_ids[idx] for idx in
                        random_choices[(num_testing_subjects + num_validation_subjects):]]
        splits.append(SubjectSplits(training_set=training_set, testing_set=testing_set, validation_set=validation_set))

    return splits


def split_subjects_into_train_test(subject_ids, test_fraction, num_splits, seed=121):
    rng = np.random.RandomState(seed)
    num_testing_subjects = int(np.round(test_fraction * len(subject_ids)))

    splits = []
    for _ in range(num_splits):
        random_choices = rng.permutation(len(subject_ids))
        testing_set = [subject_ids[idx] for idx in random_choices[:num_testing_subjects]]
        training_set = [subject_ids[idx] for idx in random_choices[num_testing_subjects:]]
        splits.append(SubjectSplits(training_set=training_set, testing_set=testing_set, validation_set=None))

    return splits
