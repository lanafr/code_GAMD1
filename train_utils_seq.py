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
from train_endtoend_autoencoder_nice import ParticleNetLightning
import torch
import numpy as np
from torch.utils.data import DataLoader
from types import SimpleNamespace
import torch.nn as nn

from einops import rearrange

NUM_OF_ATOMS = 258

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
        all_files = np.arange(0, 500)

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
        idxs = np.arange(0, 500)

        self.sequence = self.get_it(seed_num, idxs)

        positions = []

        for i in idxs:
            data = np.load(f'md_dataset/lj_data/data_{seed_num}_{i+500}.npz')
            pos_data = data['pos']
            positions.append(pos_data)

        self.positions = positions
    

    def get_it(self, seed_num, idxs):
        PATH = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/autoencoder_velocities_works?/checkpoint_29.ckpt'
        SCALER_CKPT = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/autoencoder_velocities_works?/scaler_29.npz'
        args = SimpleNamespace(use_layer_norm=False,
                            encoding_size=32,
                            hidden_dim=128,
                            edge_embedding_dim=32,
                            drop_edge=True,
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

        with torch.no_grad():
            for i in idxs:

                data = np.load(f'md_dataset/lj_data/data_{seed_num}_{i+500}.npz')
                pos_data = data['pos']
                vel_data = data['vel']
                #pos_hopefully_same, graph1, graph2, embed, force = model.predict_nextpos(pos_data)
                pred, emb, vel = model.embed_pos(pos_data, vel_data)
                embed_lst[j] = emb
                j=j+1

        #return brand_new_data
        embed_numpy = [tensor.detach().cpu().numpy() for tensor in embed_lst]

        return embed_numpy

    def get_molecule_by_molecule(self, seed_num, idxs):
        PATH = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/autoencoder_wellsee/checkpoint_29.ckpt'
        SCALER_CKPT = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/autoencoder_wellsee/scaler_29.npz'
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
            pred, graphem1, graphem2, emb = model.embed_pos(pos_data)
            embed_lst[j] = emb
            j=j+1

        #return brand_new_data
        embed_numpy = [tensor.detach().cpu().numpy() for tensor in embed_lst]

        return embed_numpy

class sequence_of_pos(Dataset):
    def __init__(self,
            dataset_path,
            sample_num,   # per seed
            device='cuda',
            split=(0.9, 0.1),
            seed_num=10,
            mode='train',):

        self.device = device
        self.seed_num = seed_num
        self.dataset_path = dataset_path
        self.sample_num = sample_num
        self.case_prefix = 'data_'
        self.seed_num = seed_num

        self.mode = mode
        assert mode in ['train', 'test']
        #np.random.seed(0)   # fix same random seed: Setting a random seed ensures that the random shuffling of idxs will be the same every time you run your code, making your results reproducible.
        seed_seq = [i for i in range(seed_num)]
        ratio = split[0]
        if mode == 'train':
            self.index = seed_seq[:int(seed_num*ratio)]
        else:
            self.index = seed_seq[int(seed_num*ratio):]

    def __getitem__(self, index):

        data = []

        data.append(self.get_sequence(index))

        return data

    def __len__(self):
        return len(self.index)
    

    def get_sequence(self, index):
        idxs = np.arange(0, 200)

        pos_lst = []
        vel_lst = []

        for i in idxs:

            current_pos = self.get_one(i, index)['pos']
            pos_lst.append(current_pos)

            current_vel = self.get_one(i, index)['vel']
            vel_lst.append(current_vel)

        dictionary = {'pos': pos_lst, 'vel': vel_lst}

        return dictionary

    def get_one(self, idx, seed, get_path_name=False):
        
        fname = f'data_{seed}_{idx+400}'#f'seed_{seed_to_read}_data_{sample_to_read}'
        #fname = f'data_5_{idx+400}'#f'seed_{seed_to_read}_data_{sample_to_read}'
        #fname_next = f'data_{seed+200}_{sample_to_read_next}'

        #fname = f'data_6_555'
        #fname_next = f'data_4_368'

        data_path = os.path.join(self.dataset_path, fname)
        #data_path_next = os.path.join(self.dataset_path, fname_next)

        data = {}
        with np.load(data_path + '.npz', 'rb') as raw_data:
            pos = raw_data['pos'].astype(np.float32)
            data['pos'] = pos
            forces = raw_data['forces'].astype(np.float32)
            data['forces'] = forces
            vel = raw_data['vel'].astype(np.float32)
            data['vel'] = vel
            """
        with np.load(data_path_next + '.npz', 'rb') as raw_data_next:
            pos_next = raw_data_next['pos'].astype(np.float32)
            data['pos_next'] = pos_next
            forces_next = raw_data_next['forces'].astype(np.float32)
            data['forces_next'] = forces_next
            """

        if get_path_name:
            return data, data_path

        return data


class shorter_sequences_of_pos(Dataset):
    def __init__(self,
            dataset_path,
            sample_num,   # per seed
            device='cuda',
            split=(0.9, 0.1),
            seed_num=10,
            mode='train',
            sequence_lenght = 8,
            use_all = True,):

        self.device = device
        self.seed_num = seed_num
        self.dataset_path = dataset_path
        self.sample_num = int(sample_num/2)
        self.case_prefix = 'data_'
        self.seed_num = seed_num
        self.sequence_lenght = sequence_lenght
        self.use_all = use_all

        self.mode = mode
        assert mode in ['train', 'test']
        #np.random.seed(0)   # fix same random seed: Setting a random seed ensures that the random shuffling of idxs will be the same every time you run your code, making your results reproducible.
        seed_seq = [i for i in range(sample_num+1-sequence_lenght)]
        ratio = split[0]
        if mode == 'train':
            self.index = seed_seq[:int(len(seed_seq)*ratio)]
        else:
            self.index = seed_seq[int(len(seed_seq)*ratio):]

    def __getitem__(self, index):

        data = []
        idxs = self.index

        for i in idxs:

            data.append(self.get_sequence(i,index))

        return data

    def __len__(self):
        return len(self.index)
    

    def get_sequence(self, start, seed):
        short_seq = [i for i in range(start, start+self.sequence_lenght)]

        pos_lst = []
        vel_lst = []

        for i in short_seq:

            current_pos = self.get_one(i, seed)['pos']
            pos_lst.append(current_pos)

            current_vel = self.get_one(i, seed)['vel']
            vel_lst.append(current_vel)

        dictionary = {'pos': pos_lst, 'vel': vel_lst}

        return dictionary

    def get_one(self, idx, seed, get_path_name=False):
        
        fname = f'data_{seed}_{idx+self.sample_num}'

        data_path = os.path.join(self.dataset_path, fname)

        data = {}
        with np.load(data_path + '.npz', 'rb') as raw_data:
            pos = raw_data['pos'].astype(np.float32)
            data['pos'] = pos
            forces = raw_data['forces'].astype(np.float32)
            data['forces'] = forces
            vel = raw_data['vel'].astype(np.float32)
            data['vel'] = vel

        if get_path_name:
            return data, data_path

        return data