import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import dgl

import sys, os
sys.path.append(os.path.join('../',os.path.dirname(os.path.abspath(''))))
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
sys.path.append('/home/guests/lana_frkin/GAMDplus/code')
sys.path.append('/home/guests/lana_frkin/GAMDplus/code/LJ')
print(sys.path)

from nn_module import SimpleMDNetNew
from train_endtoend_autoencoder_nice import ParticleNetLightning, NUM_OF_ATOMS
import torch
import numpy as np
from torch.utils.data import DataLoader
from types import SimpleNamespace
import torch.nn as nn

class Sequential_data(Dataset):
    def __init__(self,
                 sample_num=1000,
                 ):

        self.sample_num = sample_num

    def __len__(self):
        return self.sample_num

    def __getitem__(self, seed_num, get_path_name=False):
        PATH = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/autoencoder_for_graphs_withpos_and_no_reg_just_rec_loss/checkpoint_29.ckpt'
        SCALER_CKPT = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/autoencoder_for_graphs_withpos_and_no_reg_just_rec_loss/scaler_29.npz'
        args = SimpleNamespace(use_layer_norm=False,
                            encoding_size=256,
                            hidden_dim=128,
                            edge_embedding_dim=1,
                            drop_edge=False,
                            conv_layer=4,
                            rotate_aug=False,
                            update_edge=False,
                            use_part=False,
                            data_dir='',
                            loss='mae')
        model = ParticleNetLightning(args).load_from_checkpoint(PATH, args=args)
        model.load_training_stats(SCALER_CKPT)
        model.cuda()
        model.eval()

        embed_lst = [0]*1000

        for i in range (1000):

            data = np.load(f'md_dataset/lj_data/data_{seed_num}_{i}.npz')
            pos_data = data['pos']
            pos_hopefully_same, graph1, graph2, embed = model.predict_nextpos(pos_data)
            embed_lst[i] = embed

        embed_lst_next = embed_lst.copy()
        embed_lst_next.pop(0)
        embed_lst.pop(999)
        brand_new_data = list(zip(embed_lst, embed_lst_next))
        
        return brand_new_data