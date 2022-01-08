from sleep_tracking.classification import utils
from sleep_tracking.classification.data_split import Datasetplits, SleepDataset, SubjectSplits
from sleep_tracking.features.psg_features import PSGLabels
from sleep_tracking.features.heartrate_features import HeartRateFeatures
from sleep_tracking.features.motion_features import MotionFeatures
from sleep_tracking.features.time_features import TimeFeatures

import numpy as np
import pandas as pd
from functools import reduce
from typing import List


class SleepDatasetLoader(object):

    def __init__(self, num_splits=5, validation_fraction=0.05, test_fraction=0.3, seed=121):
        self.num_splits = num_splits
        self.test_fraction = test_fraction
        self.validation_fraction = validation_fraction
        self.seed = seed

    def generate_subject_splits_with_validation(self, subject_ids: List[int]) -> List[SubjectSplits]:
        subject_splits = utils.split_subjects_into_train_test_validation(subject_ids=subject_ids,
                                                                         num_splits=self.num_splits,
                                                                         validation_fraction=self.validation_fraction,
                                                                         test_fraction=self.test_fraction,
                                                                         seed=self.seed)
        return subject_splits

    def generate_subject_splits_without_validation(self, subject_ids: List[int]) -> List[SubjectSplits]:
        subject_splits = utils.split_subjects_into_train_test(subject_ids=subject_ids,
                                                              num_splits=self.num_splits,
                                                              test_fraction=self.test_fraction,
                                                              seed=self.seed)
        return subject_splits

    def generate_data_splits_without_validation(self, subject_ids: List[int], features: List[str]) -> List[
        Datasetplits]:
        subject_splits = self.generate_subject_splits_without_validation(subject_ids)

        data_splits = []
        for subject_split in subject_splits:
            split_subject_ids = list(
                set(subject_split.training_set) | set(subject_split.testing_set))
            split_data_df = self.load_modeling_data(split_subject_ids)
            assert set(features).issubset(split_data_df.columns.to_list()), "wrong features passed"

            columns = ['subject', 'timestamp'] + features + ['psg_labels']
            training_df = split_data_df.loc[np.isin(split_data_df['subject'], subject_split.training_set), columns]
            testing_df = split_data_df.loc[np.isin(split_data_df['subject'], subject_split.testing_set), columns]

            data_split = Datasetplits(training_set=SleepDataset(df=training_df,
                                                                features=features,
                                                                subjects=subject_split.training_set),
                                      validation_set=None,
                                      testing_set=SleepDataset(df=testing_df,
                                                               features=features,
                                                               subjects=subject_split.testing_set))
            data_splits.append(data_split)
        return data_splits

    def generate_data_splits_with_validation(self, subject_ids: List[int], features: List[str]) -> List[Datasetplits]:
        subject_splits = self.generate_subject_splits_with_validation(subject_ids)

        data_splits = []
        for subject_split in subject_splits:
            split_subject_ids = list(
                set(subject_split.training_set) | set(subject_split.testing_set) | set(subject_split.validation_set))
            split_data_df = self.load_modeling_data(split_subject_ids)
            assert set(features).issubset(split_data_df.columns.to_list()), "wrong features passed"

            columns = ['subject', 'timestamp'] + features + ['psg_labels']
            training_df = split_data_df.loc[np.isin(split_data_df['subject'], subject_split.training_set), columns]
            validation_df = split_data_df.loc[np.isin(split_data_df['subject'], subject_split.validation_set), columns]
            testing_df = split_data_df.loc[np.isin(split_data_df['subject'], subject_split.testing_set), columns]

            data_split = Datasetplits(training_set=SleepDataset(df=training_df,
                                                                features=features,
                                                                subjects=subject_split.training_set),
                                      validation_set=SleepDataset(df=validation_df,
                                                                  features=features,
                                                                  subjects=subject_split.validation_set),
                                      testing_set=SleepDataset(df=testing_df,
                                                               features=features,
                                                               subjects=subject_split.testing_set))
            data_splits.append(data_split)
        return data_splits

    @staticmethod
    def load_modeling_data(subject_ids: List[int]) -> pd.DataFrame:
        psg_labels = PSGLabels().load(subject_ids)
        hr_features = HeartRateFeatures().load(subject_ids)
        motion_features = MotionFeatures().load(subject_ids)
        time_features = TimeFeatures().load(subject_ids)
        combined = reduce(lambda df1, df2: pd.merge(df1, df2, on=['subject', 'timestamp']),
                          [psg_labels, hr_features, motion_features, time_features])
        return combined
