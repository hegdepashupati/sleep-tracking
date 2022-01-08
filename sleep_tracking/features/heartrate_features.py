import pandas as pd

from sleep_tracking.features.base_features import AbstractFeature
from sleep_tracking.preprocessing.cropped_data_loader import CroppedDatasetLoader
from sleep_tracking.features import utils
import numpy as np


class HeartRateFeatures(AbstractFeature):
    WINDOW_SIZE = 10 * 30 - 15

    def __init__(self):
        super(HeartRateFeatures, self).__init__(name='hr_features')

    def build(self, subject):
        psg_df = CroppedDatasetLoader.load_psg_valid_epochs([subject])
        hr_df = CroppedDatasetLoader.load_hr([subject])

        interpolated_timestamps, interpolated_heartrate = self.interpolate(
            measurement_timestamps=hr_df['timestamp'].to_numpy()[1:],
            measurement_values=hr_df['hr'].to_numpy()[1:])

        normalized_heartrate_vals = self.compute_normalized_heartrate(
            interpolated_heartrate)

        normalized_heartrate_diff = self.compute_normalized_heartrate_diff(
            interpolated_heartrate)

        feature_timestamp, feature_normalized_hr_avg, feature_hr_avg, feature_hr_variation = [], [], [], []
        for epoch_timestamp in psg_df.loc[psg_df['epoch_valid'], 'timestamp'].to_numpy():
            epoch_indices = self.find_epoch_window(epoch_timestamp, interpolated_timestamps)
            hr_interpolated_values = interpolated_heartrate[epoch_indices]
            hr_normalized_values = normalized_heartrate_vals[epoch_indices]
            hr_normalized_diffs = normalized_heartrate_diff[epoch_indices]
            feature_hr_avg.append(np.mean(hr_interpolated_values))
            feature_normalized_hr_avg.append(np.mean(hr_normalized_values))
            feature_hr_variation.append(np.std(hr_normalized_diffs))
            feature_timestamp.append(epoch_timestamp)
        feature_df = pd.DataFrame({'timestamp': feature_timestamp,
                                   'hr_avg': (feature_hr_avg - np.mean(feature_hr_avg)) / np.std(feature_hr_avg),
                                   'hr_avg_normalized': (feature_normalized_hr_avg - np.mean(
                                       feature_normalized_hr_avg)) / np.std(feature_normalized_hr_avg),
                                   'hr_variation': (feature_hr_variation - np.mean(feature_hr_variation)) / np.std(
                                       feature_hr_variation)})
        return feature_df

    def compute_normalized_heartrate(self, heartrate_values):
        return heartrate_values - np.min(heartrate_values)

    def compute_normalized_heartrate_diff(self, heartrate_values):
        heartrate_diffs = utils.convolve_with_dog(heartrate_values.flatten(), self.WINDOW_SIZE)
        scalar = np.percentile(np.abs(heartrate_diffs), 90)
        heartrate_diffs = heartrate_diffs / scalar
        return heartrate_diffs
