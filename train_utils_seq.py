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
    def __init__(self, seq_len, seed_num, device='cuda'):
        self.seq_len = seq_len
        self.device = device
        self.seed_num = seed_num
        all_files = np.arange(0, 1000)

        self.rng = np.random.RandomState(891374)
        np.random.RandomState(125487).shuffle(all_files)
        # 15% test set, 10% validation set, the rest is for training
        test_n = int(0.15 * len(all_files))
        valid_n = int((0.15 + 0.1) * len(all_files))
        test_files = all_files[:test_n]
        valid_files = all_files[test_n:valid_n]
        train_files = all_files[valid_n:]

        train_x, train_t, train_y = self.get_it(seed_num, train_files)
        valid_x, valid_t, valid_y = self.get_it(seed_num, valid_files)
        test_x, test_t, test_y = self.get_it(seed_num, test_files)

        
        train_x, train_t, train_y = self.perturb_sequences(train_x, train_t, train_y)
        valid_x, valid_t, valid_y = self.perturb_sequences(valid_x, valid_t, valid_y)
        test_x, test_t, test_y = self.perturb_sequences(test_x, test_t, test_y)


        self.train_x, self.train_times, self.train_y = self.align_sequences(
            train_x, train_t, train_y
        )
        self.valid_x, self.valid_times, self.valid_y = self.align_sequences(
            valid_x, valid_t, valid_y
        )
        self.test_x, self.test_times, self.test_y = self.align_sequences(
            test_x, test_t, test_y
        )
        self.input_size = self.train_x.shape[-1]

        # print("train_times: ", str(self.train_times.shape))
        # print("train_x: ", str(self.train_x.shape))
        # print("train_y: ", str(self.train_y.shape))

    

    def autoencode(file):
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

        data = np.load(file)
        pos_data = data['pos']
        pos_hopefully_same, graph1, graph2, embed = model.predict_nextpos(pos_data)

        return embed


    def align_sequences(self, set_x, set_t, set_y):

        times = []
        x = []
        y = []
        for i in range(len(set_y)):

            seq_x = set_x[i]
            seq_t = set_t[i]
            seq_y = set_y[i]

            for t in range(0, seq_y.shape[0] - self.seq_len, self.seq_len // 4):
                x.append(seq_x[t: t + self.seq_len])
                times.append(seq_t[t: t + self.seq_len])
                y.append(seq_y[t: t + self.seq_len])
        L = (
            np.stack(x, axis=0),
            np.stack(times, axis=0),
            np.stack(y, axis=0),
        )

        return [rearrange(torch.Tensor(i), 'b t ... -> t b ...').to(self.device) for i in L]

    def perturb_sequences(self, set_x, set_t, set_y):

        x = []
        times = []
        y = []
        for i in range(len(set_y)):

            seq_x = set_x[i]
            seq_y = set_y[i]

            new_x, new_times = [], []
            new_y = []

            skip = 0
            for t in range(seq_y.shape[0]):
                skip += 1
                if self.rng.rand() < 0.9:
                    new_x.append(seq_x[t])
                    new_times.append(skip)
                    new_y.append(seq_y[t])
                    skip = 0

            #x.append(np.stack(new_x, axis=0))
            x.append(np.stack([t.detach().cpu().numpy() for t in new_x], axis=0))
            times.append(np.stack(new_times, axis=0))
            y.append(np.stack([t.detach().cpu().numpy() for t in new_y], axis=0))

        return x, times, y

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
            pos_hopefully_same, graph1, graph2, embed = model.predict_nextpos(pos_data)
            embed_lst[j] = embed
            j=j+1
        #return brand_new_data
        without_first = embed_lst[1:]
        without_last = embed_lst[:-1]
        x = np.linspace(0,1,len(idxs))

        return without_last, x, without_first