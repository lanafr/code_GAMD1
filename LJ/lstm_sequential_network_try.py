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
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # just to test if it works w/o gpu
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

## try number 100

import torch
import torchcde

from heavyballNODE import *

import wandb

wandb.init(project="NeuralODE jej", name="Sequential model")

from torchdyn.core import NeuralODE
from torchdyn.datasets import *
from torchdyn import *
from torchdyn.models.hybrid import HybridNeuralDE
from torchdyn.models import *

import torch
import torch.utils.data as data
device = torch.device("cuda") # all of this works in GPU as well :)

import torch.nn as nn
import pytorch_lightning as pl

import torch
import torch.nn as nn

def build_model(args, ckpt=None):

    param_dict = {
                  'encoding_size': args.encoding_size,
                  }

    print("Using following set of hyper-parameters")
    print(param_dict)

    res = True
    cont = True
    
    model = MODEL(res=res, cont=cont)

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

    def forward(self, x, integration_time):
        #x = self.mlp1(x)

        x = self.model(x, integration_time)
        #x = self.mlp2(x)
        return x

    def train_dataloader(self):

        dataset = []

        # data dim [timestamps, batch, channels (derivatives), feature dimension]

        n=100

        for i in range(9):
            X_train = torch.Tensor(just_a_sequence(2).train_x[0:n]).to(device)
            #for j in range(258):
            #    dataset.append(X_train[:, j, :])

            timestamps = X_train.size()[0]
            batch_size = 1
            channels = X_train.size()[1]  # For example, representing different derivatives
            feature_dim = X_train.size()[2]

            # Create a tensor with random values
            tensor_shape = (timestamps, batch_size, channels, feature_dim)
            new_X = torch.zeros(tensor_shape)

            integration_time = torch.arange(X_train.size()[0]).to(device)
            integration_time = integration_time.float()

            new_X[:,0] = integration_time
            new_X[:,2] = X_train[:,1]
            new_X[:,3] = X_train[:,2]

            dataset.append(new_X)

        return DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=False)

    

    def training_step(self, batch, batch_idx):

        loss = 0
        
        for i in range (len(batch)):
            x = batch[i]

            x[:,1] = batch_idx

            tensor_shape = (x.shape()[0], 1)
            t = torch.zeros(tensor_shape)
            t[:,0] = x[:,0]
            t[:,1] = x[:,1]


            y_hat = self.model(x, t)

            one_loss = nn.MSELoss()(y_hat, x)

            l1_reg = torch.sum(torch.abs(y_hat))/(y_hat.size()[0]*y_hat.size()[1]*y_hat.size()[2])

            #loss = loss + 0.01*l1_reg + one_loss
            loss = loss + one_loss


            self.log('training loss', loss, on_step=True, prog_bar=True, logger=True)

        print("I'm here yey")

        return loss

    def val_dataloader(self):

        dataset = []

        for i in range(1):
            X_val = torch.Tensor(just_a_sequence(2).train_x[0:100]).to(device)
            dataset.append(X_val)

            timestamps = X_val.size()[0]
            batch_size = 1
            channels = X_val.size()[1]  # For example, representing different derivatives
            feature_dim = X_val.size()[2]

            # Create a tensor with random values
            tensor_shape = (timestamps, batch_size, channels, feature_dim)
            new_X = torch.zeros(tensor_shape)
            print(new_X[:,0].size())
            print(new_X[:,...].size())

            integration_time = torch.arange(X_val.size()[0]).to(device)
            integration_time = integration_time.float()

            new_X[:,0] = integration_time
            new_X[:,2] = X_train[:,1]
            new_X[:,3] = X_train[:,2]

            dataset.append(new_X)

        return DataLoader(dataset, batch_size=1, shuffle = True, pin_memory=False)

    
    def validation_step(self, batch, batch_idx):

        with torch.no_grad():
            x = batch[0]

            x[:,1] = batch_idx

            tensor_shape = (x.shape()[0], 1)
            t = torch.zeros(tensor_shape)
            t[:,0] = x[:,0]
            t[:,1] = x[:,1]


            y_hat = self.model(x, t)

            one_loss = nn.MSELoss()(y_hat, x)

            #loss = loss/x.size()[0]

            self.log('val loss', loss, on_step=True, prog_bar=True, logger=True)

    def ode_embed_func(self, embedding_start: torch.Tensor, t):
        x = embedding_start.to(device)
        integration_time = torch.arange(t).to(device)
        y_hat = self.model(x, integration_time)
        return y_hat

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        sched = StepLR(optim, step_size=10, gamma=0.001**(10/500)) ## changed 5 to 10
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

class tempf(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.actv = nn.Tanh()
        self.dense1 = nn.Linear(in_channels, in_channels)
        self.dense2 = nn.Linear(in_channels, out_channels)
        # self.dense3 = nn.Linear(out_channels, out_channels)

    def forward(self, h, x):
        out = self.dense1(x)
        out = self.actv(out)
        out = self.dense2(out)
        out = self.actv(out)
        # out = self.dense3(out)
        return out


class temprnn(nn.Module):
    def __init__(self, in_channels, out_channels, nhidden, res=False, cont=False):
        super().__init__()
        self.actv = nn.Tanh()
        self.dense1 = nn.Linear(in_channels + 2 * nhidden, 2 * nhidden)
        self.dense2 = nn.Linear(2 * nhidden, 2 * nhidden)
        self.dense3 = nn.Linear(2 * nhidden, 2 * out_channels)
        self.cont = cont
        self.res = res

    def forward(self, h, x):
        print(h[:, 0].size())
        print(x.size())
        out = torch.cat([h[:, 0], h[:, 1], x], dim=1)
        print(h[:, 0].size())
        print(x.size())
        out = self.dense1(out)
        out = self.actv(out)
        out = self.dense2(out)
        out = self.actv(out)
        out = self.dense3(out).reshape(h.shape)
        out = out + h
        return out


class MODEL(nn.Module):
    def __init__(self, res=False, cont=False):
        super(MODEL, self).__init__()
        nhid = 128
        self.cell = HeavyBallNODE(tempf(nhid, nhid), corr=0, corrf=True)
        # self.cell = HeavyBallNODE(tempf(nhid, nhid))
        #self.try1 = NODEintegrate(self.cell)
        self.rnn = temprnn(32, nhid, nhid, res=res, cont=cont)
        self.ode_rnn = ODE_RNN_with_Grad_Listener(self.cell, self.rnn, (2, nhid), None, tol=1e-5)
        self.outlayer = nn.Linear(nhid, 32)

    def forward(self, x, t):
        x = torch.tensor(x).to('cuda')
        t = torch.tensor(t).to('cuda')
        out = self.ode_rnn(t, x, retain_grad=True)[0]
        #out = self.try1(t, x)
        out = self.outlayer(out[:, :, 0])[1:]
        return out
        

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
    parser.add_argument('--min_epoch', default=50, type=int)
    parser.add_argument('--max_epoch', default=150, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--cp_dir', default='./model_ckpt/sequential_network_withprvipravi')
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