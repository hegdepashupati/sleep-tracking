from typing import List, Union
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SleepDataset(Dataset):
    def __init__(self, df: pd.DataFrame, features: List[str], subjects: List[int]):
        self.df = df
        self.features = features
        self.subjects = subjects
        self.target = 'psg_labels'

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        # return observations in the exact order of subjects as defined by idx
        if isinstance(idx, int):
            subjects = list([self.subjects[idx]])
        else:
            subjects = [self.subjects[_idx] for _idx in idx]
        x, y = [], []
        for subject in subjects:
            x.append(self.df.loc[self.df['subject'] == subject, self.features].to_numpy())
            y.append(self.df.loc[self.df['subject'] == subject, self.target].to_numpy())
        return np.concatenate(x).astype(np.float32), np.concatenate(y).astype(np.int64)
        # x = self.df.loc[np.isin(self.df['subject'], subjects), self.features].to_numpy()
        # y = self.df.loc[np.isin(self.df['subject'], subjects), self.target].to_numpy()
        # return x.astype(np.float32), y.astype(np.int64)


class SubjectSplits(object):
    def __init__(self, training_set: List[int], testing_set: List[int], validation_set: Union[None, List[int]]):
        self.training_set = training_set
        self.testing_set = testing_set
        self.validation_set = validation_set


class Datasetplits(object):
    def __init__(self, training_set: SleepDataset,
                 testing_set: SleepDataset,
                 validation_set: Union[None, SleepDataset]):
        self.training_set = training_set
        self.testing_set = testing_set
        self.validation_set = validation_set
