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

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from nn_module import SimpleMDNetNew
from train_utils_seq import Sequential_data, Some_seq_data, just_a_sequence
from graph_utils import NeighborSearcher, graph_network_nbr_fn
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "" # just to test if it works w/o gpu
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

class Learner(pl.LightningModule):
    def __init__(self, model:nn.Module, epoch_num=100, batch_size=1, learning_rate=3e-4, log_freq=1,
                 model_weights_ckpt=None, scaler_ckpt=None):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.log_freq = log_freq
        #self.mlp1 = nn.Linear(128, 512)
        #self.mlp2 = nn.Linear(512, 128)

    def forward(self, x):
        #x = self.mlp1(x)
        x = self.model(x)
        #x = self.mlp2(x)
        return x

    def train_dataloader(self):

        dataset = []

        n=100

        for i in range(1):
            X_train = torch.Tensor(just_a_sequence(0).train_x[0:n]).to(device)
            dataset.append(X_train)

        return DataLoader(dataset, batch_size=1, shuffle = False)

    

    def training_step(self, batch, batch_idx):

        print("I'm hereee")
        print(torch.cuda.current_device())

        x = batch[0]
        y_hat = self.model(x[0:-1])
        loss = nn.MSELoss()(y_hat, x[1:])

        #loss = loss/x.size()[0]

        self.log('training loss', loss, on_step=True, prog_bar=True, logger=True)

        return loss

    """
    def training_step(self, batch, batch_idx):

        x = batch[0]

        m = x.size()[0]

        t_span = torch.arange(0, m/1000, 0.001)
        t_span = t_span.to(device)
        t, y = self.model(x[0:-1],t_span)
        y_hat = y[-1]

        #Y = self.out(Y)

        #y_hat = self.model(x[0:-1])
        #loss = nn.MSELoss()(y_hat, x[1:])
        loss = nn.MSELoss()(y_hat, x[-1])

        #loss = loss/x.size()[0]

        self.log('training loss', loss, on_step=True, prog_bar=True, logger=True)

        return loss

    """

    def val_dataloader(self):

        dataset = []

        for i in range(1):
            X_val = torch.Tensor(just_a_sequence(0).train_x[0:100]).to(device)
            dataset.append(X_val)

        return DataLoader(dataset, batch_size=1, shuffle = True)

        
    
    def validation_step(self, batch, batch_idx):

        with torch.no_grad():
            x = batch[0]
            y_hat = self.model(x[0:-1])
            loss = nn.MSELoss()(y_hat, x[1:])

            #loss = loss/x.size()[0]

            self.log('validation loss', loss, on_step=True, prog_bar=True, logger=True)

            return loss

    """
    def validation_step(self, batch, batch_idx):

        with torch.no_grad():

            print("Here jej")
            x = batch[0]

            print("No actually here jej")

            m = x.size()[0]

            t_span = torch.arange(0, m/1000, 0.001)
            t_span = t_span.to(device)

            print("Lol at least somewhere")
            t, y = self.model(x[0:-1],t_span)

            #Y = self.out(Y)

            #y_hat = self.model(x[0:-1])
            #loss = nn.MSELoss()(y_hat, x[1:])
            print("Who knows")
            loss = nn.MSELoss()(y, x[1:])

            #loss = loss/x.size()[0]

            self.log('val loss', loss, on_step=True, prog_bar=True, logger=True)

            return loss

    """



    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)

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

class f1(nn.Module):
    def __init__(self, encoding_size):
        super().__init__()
        nhidden = 512
        self.actv = nn.Tanh()
        self.dense1 = nn.Linear(encoding_size, nhidden)
        self.dense2 = nn.Linear(nhidden, nhidden)
        #self.dense3 = nn.Linear(1024, 1024)
        #self.dense4 = nn.Linear(1024, nhidden)
        self.dense5 = nn.Linear(nhidden, encoding_size)


    def forward(self, x):
        out = self.dense1(x)
        out = self.actv(out)
        out = self.dense2(out)
        #out = self.actv(out)
        #out = self.dense3(out)
        #out = self.actv(out)
        #out = self.dense4(out)
        out = self.actv(out)
        out = self.dense5(out)
        return out

class f2(nn.Module):
    def __init__(self, encoding_size):
        super().__init__()
        nhidden = 512
        self.actv = nn.Tanh()
        self.dense1 = nn.Linear(encoding_size, nhidden)
        self.dense2 = nn.Linear(nhidden, nhidden)
        self.dense3 = nn.Linear(nhidden, 128)


    def forward(self, x):
        out = self.dense1(x)
        out = self.actv(out)
        out = self.dense2(x)
        out = self.actv(out)
        out = self.dense3(out)
        return out

class flow_NODE(nn.Module):
    def __init__(self, encoding_size):
        super().__init__()
        self.encoding_size = encoding_size
        self.flow = NeuralODE(f1(self.encoding_size), sensitivity='adjoint', solver='rk4', solver_adjoint='dopri5', atol_adjoint=1e-4, rtol_adjoint=1e-4).to(device)

    def forward(self,x):
        print("Problem isn't here")
        t, x = self.flow(x)
        print("Or here")
        return x[-1]
        

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

    
    cwd = os.getcwd()
    model_check_point_dir = os.path.join(cwd, check_point_dir)
    os.makedirs(model_check_point_dir, exist_ok=True)
    epoch_end_callback = ModelCheckpointAtEpochEnd(filepath=model_check_point_dir, save_step_frequency=1)
    checkpoint_callback = pl.callbacks.ModelCheckpoint()

    lstm_cell = nn.LSTMCell(input_size=128, hidden_size=512).to(device)

    #model = NeuralODE(f1(encoding_size), sensitivity='adjoint', solver='rk4', solver_adjoint='dopri5', atol_adjoint=1e-4, rtol_adjoint=1e-4).to(device)
    model = HybridNeuralDE(flow = flow_NODE(512), jump = lstm_cell, out= f2(512), last_output=False, reverse=False).to(device)


    learn = Learner(model, epoch_num=max_epoch,
                                 learning_rate=lr,
                                 model_weights_ckpt=weight_ckpt,
                                 batch_size=batch_size)
    trainer = pl.Trainer(logger = WandbLogger(), min_epochs=min_epoch, max_epochs=max_epoch, log_every_n_steps=1)
    trainer.fit(learn)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_epoch', default=200, type=int)
    parser.add_argument('--max_epoch', default=500, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--cp_dir', default='./model_ckpt/sequential_network')
    parser.add_argument('--state_ckpt_dir', default=None, type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--encoding_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--use_layer_norm', action='store_true')
    parser.add_argument('--num_gpu', default=-1, type=int)
    args = parser.parse_args()
    train_model(args)


if __name__ == '__main__':
    main()