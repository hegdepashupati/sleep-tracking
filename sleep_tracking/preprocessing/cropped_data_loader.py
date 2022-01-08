from sleep_tracking import utils
from sleep_tracking.preprocessing.sleep_stages import sleep_stages_all, sleep_stages_four

import os
import numpy as np
import pandas as pd


class CroppedDatasetLoader:
    DATA_PATH = utils.get_root_directory().joinpath('data/cropped/')
    EPOCH_DURATION = 30

    @staticmethod
    def load_motion(subject_ids: list):
        colnames = ['timestamp', 'accx', 'accy', 'accz']
        data = []
        for subject in subject_ids:
            # df = pd.read_csv(os.path.join(CroppedDataLoader.DATA_PATH, "%s_cleaned_motion.out" % subject),
            #                  names=colnames, header=None, delim_whitespace=True)
            df = pd.read_csv(os.path.join(CroppedDatasetLoader.DATA_PATH, "%s_cleaned_motion.out" % subject),
                             names=colnames, delim_whitespace=True)
            df['subject'] = subject
            data.append(df)
        data = pd.concat(data)
        data = data.reindex(columns=['subject'] + colnames)
        return data

    @staticmethod
    def load_hr(subject_ids: list):
        colnames = ['timestamp', 'hr']
        data = []
        for subject in subject_ids:
            # df = pd.read_csv(os.path.join(CroppedDataLoader.DATA_PATH, "%s_cleaned_hr.out" % subject),
            #                  names=colnames, header=None, delim_whitespace=True)
            df = pd.read_csv(os.path.join(CroppedDatasetLoader.DATA_PATH, "%s_cleaned_hr.out" % subject),
                             names=colnames, delim_whitespace=True)
            df['subject'] = subject
            data.append(df)
        data = pd.concat(data)
        data = data.reindex(columns=['subject'] + colnames)
        return data

    @staticmethod
    def load_psg(subject_ids: list):
        colnames = ['timestamp', 'psg_all']
        data = []
        for subject in subject_ids:
            # df = pd.read_csv(os.path.join(CroppedDataLoader.DATA_PATH, "%s_cleaned_psg.out" % subject),
            #                  names=colnames, header=None, delim_whitespace=True)
            df = pd.read_csv(os.path.join(CroppedDatasetLoader.DATA_PATH, "%s_cleaned_psg.out" % subject),
                             names=colnames, delim_whitespace=True)
            df['subject'] = subject
            data.append(df)
        data = pd.concat(data)
        data = data.reindex(columns=['subject'] + colnames)
        data['psg_all'] = data['psg_all'].astype(int)
        data['psg_labels_all'] = pd.Categorical(data['psg_all'].map(sleep_stages_all),
                                                list(sleep_stages_all.values()),
                                                ordered=True)

        sleep_stages_combined_map = {-1: [-1], 0: [0], 1: [1, 2, 3, 4], 2: [5]}
        sleep_stages_combined_map = {v: k for k, vv in sleep_stages_combined_map.items() for v in vv}
        data['psg_three'] = data['psg_all'].map(sleep_stages_combined_map).astype(int)
        data['psg_labels_three'] = pd.Categorical(data['psg_three'].map(sleep_stages_four),
                                                  list(sleep_stages_four.values()),
                                                  ordered=True)

        return data

    @staticmethod
    def load_psg_valid_epochs(subject_ids: list):
        """
        Returns PSG dataframe with a column tagging valid epochs
        """
        psg_df = CroppedDatasetLoader.load_psg(subject_ids)
        motion_df = CroppedDatasetLoader.load_motion(subject_ids)
        hr_df = CroppedDatasetLoader.load_hr(subject_ids)

        # TODO: remove loop over subjects
        psg_df['epoch_valid'] = np.nan

        for subject in subject_ids:
            sub_psg_df = psg_df.loc[psg_df['subject'] == subject]
            sub_hr_df = hr_df.loc[hr_df['subject'] == subject]
            sub_motion_df = motion_df.loc[motion_df['subject'] == subject]

            psg_df.loc[psg_df['subject'] == subject, 'epoch_valid'] = CroppedDatasetLoader.find_valid_epochs(
                psg_timestamps=sub_psg_df['timestamp'].to_numpy(),
                psg_labels=sub_psg_df['psg_labels_all'].to_numpy(),
                hr_timestamps=sub_hr_df['timestamp'].to_numpy(),
                motion_timestamps=sub_motion_df['timestamp'].to_numpy(),
                start_time=sub_psg_df['timestamp'].min())

        psg_df['epoch_valid'] = psg_df['epoch_valid'].astype('boolean')

        return psg_df

    @staticmethod
    def find_valid_epochs(psg_timestamps, psg_labels, hr_timestamps, motion_timestamps, start_time):
        """
        A timestamp corresponding to PSG sequence is tagged `valid_epoch` if:
            - There is at least one HR and MOTION reading in the epoch window
            - Sleep stage corresponding to the timestamp is not  "unscored"
        """
        floored_hr_timestamps = hr_timestamps - np.mod(hr_timestamps - start_time,
                                                       CroppedDatasetLoader.EPOCH_DURATION)
        floored_motion_timestamps = motion_timestamps - np.mod(motion_timestamps - start_time,
                                                               CroppedDatasetLoader.EPOCH_DURATION)
        psg_valid = np.logical_and.reduce([np.isin(psg_timestamps, floored_hr_timestamps),
                                           np.isin(psg_timestamps, floored_motion_timestamps),
                                           psg_labels != "unscored"])
        return psg_valid
