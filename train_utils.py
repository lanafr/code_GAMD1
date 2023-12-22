import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import dgl


class LJDataNew(Dataset):
    def __init__(self,
                 dataset_path,
                 sample_num,   # per seed
                 case_prefix='data_',
                 seed_num=10,
                 split=(0.9, 0.1),
                 mode='train',
                 ):
        self.dataset_path = dataset_path
        self.sample_num = sample_num
        self.case_prefix = case_prefix
        self.seed_num = seed_num

        self.mode = mode
        assert mode in ['train', 'test']
        idxs = np.arange(seed_num*sample_num)
        #np.random.seed(0)   # fix same random seed: Setting a random seed ensures that the random shuffling of idxs will be the same every time you run your code, making your results reproducible.
        np.random.shuffle(idxs)
        ratio = split[0]
        if mode == 'train':
            self.idx = idxs[:int(len(idxs)*ratio)]
        else:
            self.idx = idxs[int(len(idxs)*ratio):]

    def __len__(self):
        return len(self.idx)

    """

    def __getitem__(self, idx, get_path_name=False):
        idx = self.idx[idx]
        sample_to_read = idx % self.sample_num
        seed = idx // self.sample_num
        fname = f'data_{seed}_{sample_to_read}'#f'seed_{seed_to_read}_data_{sample_to_read}'
        data_path = os.path.join(self.dataset_path, fname)

        data = {}
        with np.load(data_path + '.npz', 'rb') as raw_data:
            pos = raw_data['pos'].astype(np.float32)
            data['pos'] = pos
            forces = raw_data['forces'].astype(np.float32)
            data['forces'] = forces
        if get_path_name:
            return data, data_path
        return data

    """

    def __getitem__(self, idx, get_path_name=False):
        idx = self.idx[idx]
        sample_to_read = idx % self.sample_num
        sample_to_read_next = sample_to_read+1
        seed = idx // self.sample_num
        
        
        fname = f'data_{seed}_{sample_to_read+500}'#f'seed_{seed_to_read}_data_{sample_to_read}'
        fname_next = f'data_{seed}_{sample_to_read_next+500}'

        #fname = f'data_6_555'
        #fname_next = f'data_4_368'

        data_path = os.path.join(self.dataset_path, fname)
        data_path_next = os.path.join(self.dataset_path, fname_next)

        data = {}
        with np.load(data_path + '.npz', 'rb') as raw_data:
            pos = raw_data['pos'].astype(np.float32)
            data['pos'] = pos
            forces = raw_data['forces'].astype(np.float32)
            data['forces'] = forces
        with np.load(data_path_next + '.npz', 'rb') as raw_data_next:
            pos_next = raw_data_next['pos'].astype(np.float32)
            data['pos_next'] = pos_next
            forces_next = raw_data_next['forces'].astype(np.float32)
            data['forces_next'] = forces_next

        if get_path_name:
            return data, data_path

        return data

class Graphs_data(Dataset):
    def __init__(self, sample_num):
        self.sample_num = sample_num

        idxs = [i for i in range(sample_num)]
        assert mode in ['train', 'test']
        #np.random.seed(0)   # fix same random seed: Setting a random seed ensures that the random shuffling of idxs will be the same every time you run your code, making your results reproducible.
        np.random.shuffle(idxs)
        ratio = split[0]
        if mode == 'train':
            self.idx = idxs[:int(len(idxs)*ratio)]
        else:
            self.idx = idxs[int(len(idxs)*ratio):]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx, get_path_name=False):
        sample_to_read = self.idx[idx]

        #fname = f'graph{idx}.dgl'
        fname = 'graph3498.dgl'

        data_path = os.path.join(self.dataset_path, fname)
        
        graph = dgl.load_graphs(data_path)

        return graph