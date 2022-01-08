from sleep_tracking.features.base_features import AbstractFeature
from sleep_tracking.preprocessing.cropped_data_loader import CroppedDatasetLoader

import numpy as np


class TimeFeatures(AbstractFeature):
    SECONDS_PER_DAY = 3600 * 24
    SECONDS_PER_HOUR = 3600

    def __init__(self):
        super(TimeFeatures, self).__init__(name='time_features')

    def build(self, subject):
        psg_df = CroppedDatasetLoader.load_psg_valid_epochs([subject])
        time_df = psg_df.loc[psg_df.epoch_valid, ['timestamp']]
        timestamp = time_df['timestamp'].to_numpy()
        timestamp = timestamp - timestamp[0]
        time_df['cosine_proxy'] = self.compute_cosine_proxy(timestamp)
        return time_df

    def compute_cosine_proxy(self, time):
        sleep_drive_cosine_shift = 5
        return -1 * np.cos((time - sleep_drive_cosine_shift * self.SECONDS_PER_HOUR) *
                           2 * np.pi / self.SECONDS_PER_DAY)

