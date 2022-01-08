from sleep_tracking.utils import get_root_directory

import os
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from typing import List


class AbstractFeature(ABC):
    WINDOW_SIZE = 10 * 30 - 15
    EPOCH_DURATION = 30

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def build(self, subject):
        pass

    def get_file_location(self, subject):
        return os.path.join(get_root_directory(),
                            "data/features/%s_%s.out" % (subject, self.name))

    def write(self, subject_ids: List[int]) -> None:
        for subject in subject_ids:
            features_df = self.build(subject)
            features_df.to_csv(self.get_file_location(subject), index=False)

    def load(self, subject_ids: List[int]) -> pd.DataFrame:
        features = []
        for subject in subject_ids:
            feature_df = pd.read_csv(self.get_file_location(subject))
            feature_df['subject'] = subject
            feature_df = feature_df.reindex(columns=['subject'] + feature_df.columns[:-1].to_list())
            features.append(feature_df)
        return pd.concat(features, ignore_index=True)

    def find_epoch_window(self, psg_epoch_timestamp, measurement_timestamps):
        start_time = psg_epoch_timestamp - self.WINDOW_SIZE
        end_time = psg_epoch_timestamp + self.EPOCH_DURATION + self.WINDOW_SIZE
        timestamps_ravel = measurement_timestamps.ravel()
        indices_in_range = np.unravel_index(np.where((timestamps_ravel > start_time) & (timestamps_ravel < end_time)),
                                            measurement_timestamps.shape)
        return indices_in_range[0][0]

    @staticmethod
    def interpolate(measurement_timestamps, measurement_values):
        interpolated_timestamps = np.arange(np.amin(measurement_timestamps),
                                            np.amax(measurement_timestamps), 1)
        interpolated_values = np.apply_along_axis(
            lambda x: np.interp(interpolated_timestamps, measurement_timestamps, x), 0,
            np.atleast_2d(measurement_values.T).T)
        return interpolated_timestamps, interpolated_values
