import argparse
import os, sys
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import jax
import jax.numpy as jnp
import cupy
from pytorch_lightning.loggers import WandbLogger
from torchdiffeq import odeint, odeint_adjoint

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from nn_module import SimpleMDNetNew
from train_utils_seq import Sequential_data, Some_seq_data, just_a_sequence
from graph_utils import NeighborSearcher, graph_network_nbr_fn
import time

import torchcde

import wandb

from torchdyn.core import NeuralODE
from torchdyn.datasets import *
from torchdyn import *
from torchdyn.models.hybrid import HybridNeuralDE
from torchdyn.models import *

import torch.utils.data as data

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # just to test if it works w/o gpu
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

device = torch.device("cuda") # all of this works in GPU as well :)

class MLP(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 hidden_dim=128,
                 hidden_layer=3,
                 activation_first=False,
                 activation='relu',
                 init_param=False):
        super(MLP, self).__init__()
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.2)
        elif activation == 'sigmoid':
            act_fn = nn.Sigmoid()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'elu':
            act_fn = nn.ELU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'silu':
            act_fn = nn.SiLU()
        elif activation == 'softplus':
            act_fn = nn.Softplus()
        else:
            raise Exception('Only support: relu, leaky_relu, sigmoid, tanh, elu, as non-linear activation')

        mlp_layer = []
        for l in range(hidden_layer):
            if l != hidden_layer-1 and l != 0:
                mlp_layer += [nn.Linear(hidden_dim, hidden_dim), act_fn]
            elif l == 0:
                if hidden_layer == 1:
                    if activation_first:
                        mlp_layer += [act_fn, nn.Linear(in_feats, out_feats)]
                    else:
                        print('Using MLP with no hidden layer and activations! Fall back to nn.Linear()')
                        mlp_layer += [nn.Linear(in_feats, out_feats)]
                elif not activation_first:
                    mlp_layer += [nn.Linear(in_feats, hidden_dim), act_fn]
                else:
                    mlp_layer += [act_fn, nn.Linear(in_feats, hidden_dim), act_fn]
            else:   # l == hidden_layer-1
                mlp_layer += [nn.Linear(hidden_dim, out_feats)]
        self.mlp_layer = nn.Sequential(*mlp_layer)
        if init_param:
            self._init_parameters()

    def _init_parameters(self):
        for layer in self.mlp_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, feat):
        return self.mlp_layer(feat)

def build_model(args, ckpt=None):

    param_dict = {
                  }

    print("Using following set of hyper-parameters")
    print(param_dict)

    encoding_size = args.encoding_size
    
    model = RNN(input_size = 32, hidden_size = 256, num_layers=8)

    model.cuda()

    if ckpt is not None:
        print('Loading model weights from: ', ckpt)
        model.load_state_dict((torch.load(ckpt)))
    return model


class Learner(pl.LightningModule):
    def __init__(self, args, epoch_num=100, batch_size=1, learning_rate=3e-4, log_freq=1,
                 model_weights_ckpt=None, scaler_ckpt=None):
        super().__init__()
        self.model = build_model(args, model_weights_ckpt)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.log_freq = log_freq
        self.epoch_num = epoch_num
        #self.mlp1 = nn.Linear(128, 512)
        #self.mlp2 = nn.Linear(512, 128)

    def forward(self, x):
        #x = self.mlp1(x)

        x = self.model(x)
        #x = self.mlp2(x)
        return x

    def train_dataloader(self):

        dataset = []

        n=2

        for i in range(9):
            X_train = torch.Tensor(just_a_sequence(9).sequence[0:n]).to(device)
            #for j in range(258):
            #    dataset.append(X_train[:, j, :])
            #X_train = torch.rand((2,258,32))
            dataset.append(X_train)

        return DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=False)

    

    def training_step(self, batch, batch_idx):

        loss = 0
        
        for i in range (len(batch)):
            x = batch[i]

            print("Hejhej friend")

            y_hat = self.model(x)

            one_loss = nn.MSELoss()(y_hat, x)

            l1_reg = torch.sum(torch.abs(y_hat))/(y_hat.size()[0]*y_hat.size()[1]*y_hat.size()[2])

            #multiplier = torch.arange(x.size()[0])/x.size()[0].to(device)

            #loss = loss + 0.01*l1_reg + one_loss

            loss = loss + one_loss


            self.log('training loss', loss, on_step=True, prog_bar=True, logger=True)

        print("I'm here yey")

        return loss

    def val_dataloader(self):

        dataset = []

        for i in range(1):
            X_val = torch.Tensor(just_a_sequence(9).sequence[0:2]).to(device)
            dataset.append(X_val)

        return DataLoader(dataset, batch_size=1, shuffle = True, pin_memory=False)

    
    def validation_step(self, batch, batch_idx):

        with torch.no_grad():
            x = batch[0]
            loss = 0
            """ for  molecule by molecule
            for i in range(258):
                integration_time = torch.arange(x.size()[0]).to("cuda")
                integration_time = integration_time.float()
                y_hat = self.model(x[:,i,:][0], integration_time)
                one_loss = nn.MSELoss()(y_hat,x[:,i,:])
                loss = loss + one_loss
            """
            y_hat = self.model(x)
            print(y_hat.size())
            print(x.size())
            one_loss = nn.MSELoss()(y_hat,x)
            loss = loss + one_loss

            #loss = loss/x.size()[0]

            self.log('val loss', loss, on_step=True, prog_bar=True, logger=True)

    def ode_embed_func(self, embedding_start: torch.Tensor, t):
        y_hat = []

        out = embedding_start

        y_hat.append(out)
        
        for i in range (t-1):
            out = self.model(out)
            y_hat.append(out)
        
        return y_hat

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        sched = StepLR(optim, step_size=20, gamma=0.001**(10/500)) ## changed 5 to 10
        return [optim], [sched]

class ModelCheckpointAtEpochEnd(pl.Callback):
    """
       Save a checkpoint at epoch end
    """
    def __init__(
            self,
            filepath,
            save_step_frequency,
            prefix="checkpoint",
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
        """
        self.filepath = filepath
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: Learner):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        if epoch % self.save_step_frequency == 0 or epoch == pl_module.epoch_num -1:
            filename = os.path.join(self.filepath, f"{self.prefix}_{epoch}.ckpt")
            scaler_filename = os.path.join(self.filepath, f"scaler_{epoch}.npz")

            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)
            np.savez(scaler_filename)
            # joblib.dump(pl_module.train_data_scaler, scaler_filename)


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity='tanh',
        )

        self.linear = nn.Linear(256, 32)

    def forward(self, x):
        output, hn = self.rnn(x)
        output = self.linear(output)

        return output      

def train_model(args):
    lr = args.lr
    num_gpu = args.num_gpu
    check_point_dir = args.cp_dir
    min_epoch = args.min_epoch
    max_epoch = args.max_epoch
    weight_ckpt = args.state_ckpt_dir
    batch_size = args.batch_size
    encoding_size = args.encoding_size
    wandb_logger = WandbLogger()
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    wandb.init(project="NeuralODE jej", name="Sequential model")

    #lstm_cell = nn.LSTMCell(input_size=128, hidden_size=512).to(device)

    #model = NeuralODE(f1(encoding_size), sensitivity='adjoint', solver='rk4', solver_adjoint='dopri5', atol_adjoint=1e-4, rtol_adjoint=1e-4).to(device)
    #model = HybridNeuralDE(flow = flow_NODE(512), jump = lstm_cell, out= f2(512), last_output=False, reverse=False).to(device)

    learn = Learner(epoch_num=max_epoch,
                                 learning_rate=lr,
                                 model_weights_ckpt=weight_ckpt,
                                 batch_size=batch_size,
                                 args=args)

    
    cwd = os.getcwd()
    model_check_point_dir = os.path.join(cwd, check_point_dir)
    os.makedirs(model_check_point_dir, exist_ok=True)
    epoch_end_callback = ModelCheckpointAtEpochEnd(filepath=model_check_point_dir, save_step_frequency=10)
    checkpoint_callback = pl.callbacks.ModelCheckpoint()

    trainer = pl.Trainer(logger = WandbLogger(),
                        min_epochs=min_epoch,
                        max_epochs=max_epoch,
                        callbacks=[epoch_end_callback, checkpoint_callback],
                        log_every_n_steps=1)
    trainer.fit(learn)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_epoch', default=100, type=int)
    parser.add_argument('--max_epoch', default=110, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--cp_dir', default='./model_ckpt/sequential_network_RNN')
    parser.add_argument('--state_ckpt_dir', default=None, type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--encoding_size', default=32, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--use_layer_norm', action='store_true')
    parser.add_argument('--num_gpu', default=-1, type=int)
    args = parser.parse_args()
    train_model(args)


if __name__ == '__main__':
    main()