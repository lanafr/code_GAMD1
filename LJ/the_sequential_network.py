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
from torchdiffeq import odeint

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

    encoding_size = args.encoding_size

    func = f1(encoding_size)
    
    model = ODEBlock(func)

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

        n=499

        for i in range(9):
            X_train = torch.Tensor(just_a_sequence(i).train_x[0:n]).to(device)
            dataset.append(X_train)

        return DataLoader(dataset, batch_size=1, shuffle = False)

    

    def training_step(self, batch, batch_idx):

        x = batch[0]
        integration_time = torch.arange(x.size()[0]).to(device)
        y_hat = self.model(x[0], integration_time)
        loss = nn.MSELoss()(y_hat, x)

        l1_reg = torch.sum(torch.abs(y_hat))/(y_hat.size()[0]*y_hat.size()[1]*y_hat.size()[2])

        loss = l1_reg +loss

        self.log('training loss', loss, on_step=True, prog_bar=True, logger=True)

        return loss

    def val_dataloader(self):

        dataset = []

        for i in range(1):
            X_val = torch.Tensor(just_a_sequence(9).train_x[0:499]).to(device)
            dataset.append(X_val)

        return DataLoader(dataset, batch_size=1, shuffle = True)

    
    def validation_step(self, batch, batch_idx):

        with torch.no_grad():
            x = batch[0]

            integration_time = torch.arange(x.size()[0]).to("cuda")
            y_hat = self.model(x[0], integration_time)
            loss = nn.MSELoss()(y_hat,x)

            #loss = loss/x.size()[0]

            self.log('val loss', loss, on_step=True, prog_bar=True, logger=True)

    def ode_embed_func(self, embedding_start: torch.Tensor, t):
        x = embedding_start.to(device)
        integration_time = torch.arange(t).to(device)
        y_hat = self.model(x, integration_time)
        return y_hat

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        sched = StepLR(optim, step_size=2, gamma=0.001**(10/500)) ## changed 5 to 2
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

class f1(nn.Module):
    def __init__(self, encoding_size):
        super().__init__()
        nhidden = 512
        self.actv = nn.LeakyReLU(negative_slope=0.01)
        self.dense1 = nn.Linear(encoding_size, 256)
        self.dense2 = nn.Linear(256, nhidden)
        self.dense3 = nn.Linear(nhidden, 1024)
        self.dense4 = nn.Linear(1024, nhidden)
        self.dense5 = nn.Linear(nhidden, 256)
        self.dense6 = nn.Linear(256, encoding_size)
        


    def forward(self, t, x):
        out = self.dense1(x)
        out = self.actv(out)
        out = self.dense2(out)
        out = self.actv(out)
        out = self.dense3(out)
        out = self.actv(out)
        out = self.dense4(out)
        out = self.actv(out)
        out = self.dense5(out)
        out = self.actv(out)
        out = self.dense6(out)
        return out

class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc

    def forward(self, x, integration_time):
        out = odeint(self.odefunc, x, integration_time, solver = 'rk4', atol=1e-6, rtol=1e-6)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
        

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

    #lstm_cell = nn.LSTMCell(input_size=128, hidden_size=512).to(device)

    #model = NeuralODE(f1(encoding_size), sensitivity='adjoint', solver='rk4', solver_adjoint='dopri5', atol_adjoint=1e-4, rtol_adjoint=1e-4).to(device)
    #model = HybridNeuralDE(flow = flow_NODE(512), jump = lstm_cell, out= f2(512), last_output=False, reverse=False).to(device)

    func = f1(encoding_size)
    
    model = ODEBlock(func)
    model.cuda()

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
    parser.add_argument('--min_epoch', default=200, type=int)
    parser.add_argument('--max_epoch', default=300, type=int)
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