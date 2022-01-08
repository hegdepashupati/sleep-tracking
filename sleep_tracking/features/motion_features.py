import pandas as pd

from sleep_tracking.features.base_features import AbstractFeature
from sleep_tracking.preprocessing.cropped_data_loader import CroppedDatasetLoader
import numpy as np


class MotionFeatures(AbstractFeature):
    WINDOW_SIZE = 10 * 30 - 15

    def __init__(self):
        super(MotionFeatures, self).__init__(name='motion_features')

    def build(self, subject):
        psg_df = CroppedDatasetLoader.load_psg_valid_epochs([subject])
        motion_df = CroppedDatasetLoader.load_motion([subject])

        interpolated_timestamps, interpolated_motions = self.interpolate(
            measurement_timestamps=motion_df['timestamp'].to_numpy()[1:],
            measurement_values=motion_df[['accx', 'accy', 'accz']].to_numpy()[1:])

        motion_magnitudes = self.compute_motion_magnitude(
            interpolated_motions)

        feature_timestamp, feature_motion_avg, feature_motion_std = [], [], []
        for epoch_timestamp in psg_df.loc[psg_df['epoch_valid'], 'timestamp'].to_numpy():
            epoch_indices = self.find_epoch_window(epoch_timestamp, interpolated_timestamps)
            motion_values = motion_magnitudes[epoch_indices]
            feature_motion_avg.append(np.mean(motion_values))
            feature_motion_std.append(np.std(motion_values))
            feature_timestamp.append(epoch_timestamp)
        feature_df = pd.DataFrame({'timestamp': feature_timestamp,
                                   'motion_avg': (feature_motion_avg - np.mean(feature_motion_avg))/np.std(feature_motion_avg),
                                   'motion_std': (feature_motion_std - np.mean(feature_motion_std))/np.std(feature_motion_std)})
        return feature_df

    def compute_motion_magnitude(self, motion_values):
        return np.square(motion_values).sum(1)

