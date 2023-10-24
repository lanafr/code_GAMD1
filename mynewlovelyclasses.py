import numpy as np
import torch
import torch.nn as nn
import dgl.nn
import dgl.function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
import time
from md_module import get_neighbor
from sklearn.preprocessing import StandardScaler

from typing import List, Set, Dict, Tuple, Optional

from torchdyn.core import NeuralODE
from torchdyn import *  
from torchdyn.models import *
from torchdyn.nn import *

## here we do the class for RNN (LSTM) and NeuralODE


"""
Things to do:

1. Change dataloader so that we can get a sequence of elements to feed to the
   neural network instead of a single element (don't remove this one just also write another one)

2. Add an RNN to the network, which can later be replaces by a NeuralODE
"""

class LJDataSequence(Dataset):
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
        if mode == 'train':
            self.seed = [0:8]
        else:
            self.seed = [9]

    def __len__(self):
        return 999
 """

    def __getitem__(self, idx, get_path_name=False):
        idx = self.idx[idx]
        sample_to_read = idx % self.sample_num
        sample_to_read_next = sample_to_read+1
        seed = idx // self.sample_num
        
        
        fname = f'data_{seed}_{sample_to_read}'#f'seed_{seed_to_read}_data_{sample_to_read}'
        fname_next = f'data_{seed}_{sample_to_read_next}'

        #fname = f'data_4_555'
        #fname_next = f'data_4_555'

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

    """
    
    def __getitem__(self, seed, get_path_name=False):
        sample_num = self.sample_num

        sequence = []
        
        for i in range (sample_num):
            fname = f'data_{seed}_{i}'

            data_path = os.path.join(self.dataset_path, fname)

            with np.load(data_path + '.npz', 'rb') as raw_data:
                pos = raw_data['pos'].astype(np.float32)
                data['pos'] = pos
            
            sequence.append(pos)

        return sequence