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

from nn_module import SimpleMDNetNew_GAMD, SimpleMDNetNew
from actual_gamd import ParticleNetLightning_GAMD, NUM_OF_ATOMS
from train_endtoend_autoencoder_nice import ParticleNetLightning
import torch
import numpy as np
from torch.utils.data import DataLoader
from types import SimpleNamespace
import torch.nn as nn

from einops import rearrange

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
        #return brand_new_data
        without_first = embed_lst[1:]
        without_last = embed_lst[:-1]
        x = np.linspace(0,1,999)

        return without_last, x, without_first



class Some_seq_data(Dataset):
    def __init__(self, seed_num, device='cuda'):
        self.device = device
        self.seed_num = seed_num
        all_files = np.arange(0, 1000)

        ## self.rng = np.random.RandomState(891374)
        ## np.random.RandomState(125487).shuffle(all_files)
        # 15% test set, 10% validation set, the rest is for training
        test_n = int(0.15 * len(all_files))
        valid_n = int((0.15 + 0.1) * len(all_files))
        test_files = all_files[:test_n]
        valid_files = all_files[test_n:valid_n]
        train_files = all_files[valid_n:]

        self.train_x, self.train_times, self.train_y = self.get_it(seed_num, train_files)
        self.valid_x, self.valid_times, self.valid_y = self.get_it(seed_num, valid_files)
        self.test_x, self.test_times, self.test_y = self.get_it(seed_num, test_files)

        print("x is:")
        print(self.train_x)

    def _load_files(self, files):
        all_x = []
        all_t = []
        all_y = []
        for f in files:
            arr = np.load(f)
            x_state = arr[:-1, :].astype(np.float32)
            y = arr[1:, :].astype(np.float32)

            x_times = np.ones(x_state.shape[0])
            all_x.append(x_state)
            all_t.append(x_times)
            all_y.append(y)

            # print("Loaded file '{}' of length {:d}".format(f, x_state.shape[0]))
        return all_x, all_t, all_y

    def get_it(self, seed_num, idxs):
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

        embed_lst = [0]*len(idxs)

        j=0

        for i in idxs:

            data = np.load(f'md_dataset/lj_data/data_{seed_num}_{i}.npz')
            pos_data = data['pos']
            pos_hopefully_same, graph1, graph2, embed, force = model.predict_nextpos(pos_data)
            embed_lst[j] = embed
            j=j+1

        #return brand_new_data
        embed_numpy = [tensor.detach().cpu().numpy() for tensor in embed_lst]

        without_first = embed_numpy[1:]
        without_first = np.stack(without_first, axis=0)
        without_first_simple = without_first
        without_first = without_first[np.newaxis, ...]
        without_last = embed_numpy[:-1]
        without_last = np.stack(without_last, axis=0)
        without_last_simple = without_last
        without_last = without_last[np.newaxis, ...]

        t = np.linspace(0,len(idxs),1)

        without_first[0] = t
        without_last[0] = t

        t_new = t[np.newaxis, ...]
        t[0] = t

        #return without_last, t_new, without_first
        return without_last_simple, t_new, without_first_simple

class just_a_sequence(Dataset):
    def __init__(self, seed_num, device='cuda'):
        self.device = device
        self.seed_num = seed_num
        all_files = np.arange(0, 500)

        ## self.rng = np.random.RandomState(891374)
        ## np.random.RandomState(125487).shuffle(all_files)
        # 15% test set, 10% validation set, the rest is for training

        #valid_n = int((1-0.15) * len(all_files))
        #train_files = all_files[:valid_n]
        #valid_files = all_files[valid_n:]

        #self.train_x = self.get_it(seed_num, train_files)
        #self.valid_x = self.get_it(seed_num, valid_files)

        self.train_x = self.get_it(seed_num, all_files)

    

    def get_it(self, seed_num, idxs):
        PATH = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/autoencoder_prvipravi/checkpoint_29.ckpt'
        SCALER_CKPT = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/autoencoder_prvipravi/scaler_29.npz'
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
                            loss='mse')
        model = ParticleNetLightning(args).load_from_checkpoint(PATH, args=args)
        model.load_training_stats(SCALER_CKPT)
        model.cuda()
        model.eval()

        embed_lst = [0]*len(idxs)

        j=0

        for i in idxs:

            data = np.load(f'md_dataset/lj_data/data_{seed_num}_{i+500}.npz')
            pos_data = data['pos']
            #pos_hopefully_same, graph1, graph2, embed, force = model.predict_nextpos(pos_data)
            pred, graphem1, graphem2, emb = model.predict_nextpos(pos_data)
            embed_lst[j] = emb
            j=j+1

        #return brand_new_data
        embed_numpy = [tensor.detach().cpu().numpy() for tensor in embed_lst]

        return embed_numpy

    """
    ## for the gamd-like autoencoder
    def get_it(self, seed_num, idxs):
        PATH = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/GAMD_like/checkpoint_29.ckpt'
        SCALER_CKPT = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/GAMD_like/scaler_29.npz'
        args = SimpleNamespace(use_layer_norm=False,
                            encoding_size=128,
                            hidden_dim=128,
                            edge_embedding_dim=256,
                            drop_edge=False,
                            conv_layer=4,
                            rotate_aug=False,
                            update_edge=False,
                            use_part=False,
                            data_dir='',
                            loss='mse')
        model = ParticleNetLightning_GAMD(args).load_from_checkpoint(PATH, args=args)
        model.load_training_stats(SCALER_CKPT)
        model.cuda()
        model.eval()

        embed_lst = [0]*len(idxs)

        j=0

        for i in idxs:

            data = np.load(f'md_dataset/lj_data/data_{seed_num}_{i+500}.npz')
            pos_data = data['pos']
            #pos_hopefully_same, graph1, graph2, embed, force = model.predict_nextpos(pos_data)
            forces, pos, emb = model.predict_forces(pos_data)
            embed_lst[j] = emb
            j=j+1

        #return brand_new_data
        embed_numpy = [tensor.detach().cpu().numpy() for tensor in embed_lst]

        return embed_numpy
        """