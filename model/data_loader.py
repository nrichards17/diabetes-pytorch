import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class DiabetesDataset(Dataset):
    """
    docstring - todo
    """
    def __init__(self, df, continuous_features=None,
                 categorical_features=None, output_features=None):
        """
        docstring - todo

        Args:
            todo
        """

        continuous_features = continuous_features if continuous_features else []
        categorical_features = categorical_features if categorical_features else []
        output_features = output_features if output_features else []

        # assert columns in data
        for var in continuous_features + categorical_features + output_features:
            assert var in df.columns, f'Feature not in data.columns: {var}'

        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.output_features = output_features

        # dataset size
        self.n = df.shape[0]

        if self.output_features:
            self.y = df[output_features].astype(np.float32).values.reshape(-1, 1)
        else:
            raise ValueError('Must specify output feature(s) in list.')

        if self.continuous_features:
            self.cont_X = df[self.continuous_features].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.n, 1))

        if self.categorical_features:
            self.cat_X = df[categorical_features].astype(np.int64).values
        else:
            self.cat_X = np.zeros((self.n, 1))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return [self.y[idx], self.cont_X[idx], self.cat_X[idx]]


def fetch_dataloaders(data_dir, features, params):
    """
    docstring - todo

    Args:
        todo
    """

    dataloaders = {}

    for split in ['train', 'val', 'test']:
        name = '{}.csv'.format(split)
        FILE = os.path.join(data_dir, params['dataset'], name)
        df = pd.read_csv(FILE)

        dataset = DiabetesDataset(df,
                                  continuous_features=features['continuous'],
                                  categorical_features=features['categorical'],
                                  output_features=features['output'])

        if split == 'train':
            loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
        else:
            loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=False)

        dataloaders[split] = loader

    return dataloaders
