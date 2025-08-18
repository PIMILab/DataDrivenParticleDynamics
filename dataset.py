"""dataset.py"""

import os
import torch
from torch.utils.data import Dataset


class MolecularDataset(Dataset):
    def __init__(self, dset, idx = None):
        'Initialization'

        # Load data
        load_dir = os.path.join('data', dset, dset + '.pt')
        self.data = torch.load(load_dir, weights_only=False)
        self.data_list = [self.data['data_list'][i] for i in idx] if idx else self.data['data_list']

        # x = (r, v, s) = [D, D, 1]
        self.dims = (self.data_list[0].x.size(1) - 1) // 2
        self.name = dset

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)


def load_dataset(dset, N_train = None):

    if N_train:
        indices = torch.randperm(N_train).tolist()
        train_idx = indices[:3*N_train//4]
        val_idx = indices[3*N_train//4:]

        train_set = MolecularDataset(dset, train_idx)       
        val_set = MolecularDataset(dset, val_idx)
        return train_set, val_set
    else:
        test_set = MolecularDataset(dset)
        return test_set


if __name__ == '__main__':
    pass
