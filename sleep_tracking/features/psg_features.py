from sleep_tracking.features.base_features import AbstractFeature
from sleep_tracking.preprocessing.cropped_data_loader import CroppedDatasetLoader


class PSGLabels(AbstractFeature):
    WINDOW_SIZE = 10 * 30 - 15

    def __init__(self):
        super(PSGLabels, self).__init__(name='psg_labels')

    def build(self, subject):
        psg_df = CroppedDatasetLoader.load_psg_valid_epochs([subject])
        psg_df = psg_df.loc[psg_df.epoch_valid, ['timestamp', 'psg_three']]
        psg_df = psg_df.rename(columns={'psg_three': 'psg_labels'})
        return psg_df
