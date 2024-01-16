import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import dgl
import matplotlib.pyplot as plt


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
from the_sequential_network import Learner

from einops import rearrange
from scipy.spatial import cKDTree
from sklearn.metrics import mean_squared_error


PATH = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/autoencoder_prvipravi/checkpoint_29.ckpt'
SCALER_CKPT = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/autoencoder_prvipravi/scaler_29.npz'
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

j=0

embed_lst = []

with torch.no_grad():

    for i in range(400):

        data = np.load(f'md_dataset/lj_data/data_{1}_{i+600}.npz')
        pos_data = data['pos']
        #pos_hopefully_same, graph1, graph2, embed, force = model.predict_nextpos(pos_data)
        pred, emb = model.predict_nextpos(pos_data)

        embed_lst.append(emb)
        j=j+1
        print("I got to i")
        print(i)


print (embed_lst)