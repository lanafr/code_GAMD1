import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import torchdiffeq
import numpy as np
from einops import rearrange, repeat
import time
import torch.optim as optim
import glob
import imageio
from math import pi
from random import random
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal
from torchvision import datasets, transforms
import sys
from matplotlib import pyplot as plt
import pickle
import csv

import torch
from einops import rearrange
from torch import nn
from torchdiffeq import odeint_adjoint

# Format [time, batch, diff, vector]

tol = 1e-3


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def shrink_parameters(model, ratio):
    model_dict = model.state_dict()
    for i in model_dict:
        model_dict[i] *= ratio
    model.load_state_dict(model_dict)
    return model


def gradnorm(model, p=2):
    param_normp = [param.grad.data.norm(p) ** p for param in model.parameters() if param.grad is not None]
    total_normp = sum(param_normp)
    total_norm = total_normp ** (1 / p)
    return total_norm


class ArgumentParser:
    def add_argument(self, str, type, default):
        setattr(self, str[2:], default)

    def parse_args(self):
        return self


def str_rec(names, data, unit=None, sep=', ', presets='{}'):
    if unit is None:
        unit = [''] * len(names)
    out_str = "{{}}: {} {{{{}}}}" + sep
    out_str *= len(names)
    out_str = out_str.format(*data)
    out_str = out_str.format(*names)
    out_str = out_str.format(*unit)
    out_str = presets.format(out_str)
    return out_str


def to_float(arr, truncate=False):
    if isinstance(arr, list):
        return [to_float(i, truncate=truncate) for i in arr]
    if arr is None:
        return None
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().gpu().numpy()
    if isinstance(arr, np.ndarray):
        arr = arr.flatten()[0]
    if truncate:
        arr = int(arr * 10 ** truncate) / 10 ** truncate
    return arr


class EmptyClass:
    pass


class Recorder:
    def __init__(self):
        self.store = []
        self.current = dict()

    def __setitem__(self, key, value):
        for method in ['detach', 'cpu', 'numpy']:
            if hasattr(value, method):
                value = getattr(value, method)()
        if key in self.current:
            self.current[key].append(value)
        else:
            self.current[key] = [value]

    def capture(self, verbose=False):
        for i in self.current:
            self.current[i] = np.mean(self.current[i])
        self.store.append(self.current.copy())
        self.current = dict()
        if verbose:
            for i in self.store[-1]:
                if i[0] != '_':
                    print('{}: {}'.format(i, self.store[-1][i]))
        return self.store[-1]

    def tolist(self):
        labels = set()
        labels = sorted(labels.union(*self.store))
        outlist = []
        for obs in self.store:
            outlist.append([obs.get(i, np.nan) for i in labels])
        return labels, outlist

    def writecsv(self, writer):
        labels, outlist = self.tolist()
        if isinstance(writer, str):
            outfile = open(writer, 'w')
            csvwriter = csv.writer(outfile)
            csvwriter.writerow(labels)
            csvwriter.writerows(outlist)
            outfile.close()
        else:
            csvwriter = writer
            csvwriter.writerow(labels)
            csvwriter.writerows(outlist)


class NLayerNN(nn.Module):
    def __init__(self, *args, actv=nn.ReLU()):
        super().__init__()
        self.linears = nn.ModuleList()
        for i in range(len(args)):
            self.linears.append(nn.Linear(args[i], args[i+1]))
        self.actv = actv

    def forward(self, x):
        for i in range(self.layer_cnt):
            x = self.linears[i](x)
            if i < self.layer_cnt - 1:
                x = self.actv(x)
        return x

    @property
    def layer_cnt(self):
        return len(self.linears)

class Zeronet(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)

zeronet = Zeronet()

class TVnorm(nn.Module):
    def __init__(self):
        super(TVnorm, self).__init__()
        self.osize = 1

    def forward(self, t, x, v):
        return torch.norm(v, 1)


class NormAct(nn.Module):
    def __init__(self, bound):
        super().__init__()
        self.bound = bound
        self.relu = nn.ReLU()
        self.elu = nn.ELU()

    def forward(self, x):
        x = x - self.bound + 1
        x = self.relu(x) * self.elu(-x) + 1
        return x


class Parameter(nn.Module):
    def __init__(self, val, frozen=False):
        super().__init__()
        val = torch.Tensor(val)
        self.val = val
        self.param = nn.Parameter(val)
        self.frozen = frozen

    def forward(self):
        if self.frozen:
            self.val = self.val.to(self.param.device)
            return self.val
        else:
            return self.param

    def freeze(self):
        self.val = self.param.detach().clone()
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def __repr__(self):
        return "val: {}, param: {}".format(self.val.cuda(), self.param.detach().cuda())

class Tinvariant_NLayerNN(NLayerNN):
    def forward(self, t, x):
        return super(Tinvariant_NLayerNN, self).forward(x)


class dfwrapper(nn.Module):
    def __init__(self, df, shape, recf=None):
        super(dfwrapper, self).__init__()
        self.df = df
        self.shape = shape
        self.recf = recf

    def forward(self, t, x):
        bsize = x.shape[0]
        if self.recf:
            x = x[:, :-self.recf.osize].reshape(bsize, *self.shape)
            dx = self.df(t, x)
            dr = self.recf(t, x, dx).reshape(bsize, -1)
            dx = dx.reshape(bsize, -1)
            dx = torch.cat([dx, dr], dim=1)
        else:
            x = x.reshape(bsize, *self.shape)
            dx = self.df(t, x)
            dx = dx.reshape(bsize, -1)
        return dx


class NODEintegrate(nn.Module):

    def __init__(self, df, shape=None, tol=tol, adjoint=True, evaluation_times=None, recf=None):
        """
        Create an OdeRnnBase model
            x' = df(x)
            x(t0) = x0
        :param df: a function that computes derivative. input & output shape [batch, channel, feature]
        :param x0: initial condition.
            - if x0 is set to be nn.parameter then it can be trained.
            - if x0 is set to be nn.Module then it can be computed through some network.
        """
        super().__init__()
        self.df = dfwrapper(df, shape, recf) if shape else df
        self.tol = tol
        self.odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
        self.evaluation_times = evaluation_times if evaluation_times is not None else torch.Tensor([0.0, 1.0])
        self.shape = shape
        self.recf = recf
        if recf:
            assert shape is not None

    def forward(self, x0):
        """
        Evaluate odefunc at given evaluation time
        :param x0: shape [batch, channel, feature]. Set to None while training.
        :param evaluation_times: time stamps where method evaluates, shape [time]
        :param x0stats: statistics to compute x0 when self.x0 is a nn.Module, shape required by self.x0
        :return: prediction by ode at evaluation_times, shape [time, batch, channel, feature]
        """
        bsize = x0.shape[0]
        if self.shape:
            assert x0.shape[1:] == torch.Size(self.shape), \
                'Input shape {} does not match with model shape {}'.format(x0.shape[1:], self.shape)
            x0 = x0.reshape(bsize, -1)
            if self.recf:
                reczeros = torch.zeros_like(x0[:, :1])
                reczeros = repeat(reczeros, 'b 1 -> b c', c=self.recf.osize)
                x0 = torch.cat([x0, reczeros], dim=1)
            out = odeint(self.df, x0, self.evaluation_times, rtol=self.tol, atol=self.tol)
            if self.recf:
                rec = out[-1, :, -self.recf.osize:]
                out = out[:, :, :-self.recf.osize]
                out = out.reshape(-1, bsize, *self.shape)
                return out, rec
            else:
                return out
        else:
            out = odeint(self.df, x0, self.evaluation_times, rtol=self.tol, atol=self.tol)
            return out

    @property
    def nfe(self):
        return self.df.nfe

    def to(self, device, *args, **kwargs):
        super().to(device, *args, **kwargs)
        self.evaluation_times.to(device)


class NODElayer(NODEintegrate):
    def forward(self, x0):
        out = super(NODElayer, self).forward(x0)
        if isinstance(out, tuple):
            out, rec = out
            return out[-1], rec
        else:
            return out[-1]


'''
class ODERNN(nn.Module):
    def __init__(self, node, rnn, evaluation_times, nhidden):
        super(ODERNN, self).__init__()
        self.t = torch.as_tensor(evaluation_times).float()
        self.n_t = len(self.t)
        self.node = node
        self.rnn = rnn
        self.nhidden = (nhidden,) if isinstance(nhidden, int) else nhidden

    def forward(self, x):
        assert len(x) == self.n_t
        batchsize = x.shape[1]
        out = torch.zeros([self.n_t, batchsize, *self.nhidden]).to(x.device)
        for i in range(1, self.n_t):
            odesol = odeint(self.node, out[i - 1], self.t[i - 1:i + 1])
            h_ode = odesol[1]
            out[i] = self.rnn(h_ode, x[i])
        return out
'''


class NODE(nn.Module):
    def __init__(self, df=None, **kwargs):
        super(NODE, self).__init__()
        self.__dict__.update(kwargs)
        self.df = df
        self.nfe = 0
        self.elem_t = None

    def forward(self, t, x):
        self.nfe += 1
        if self.elem_t is None:
            return self.df(t, x)
        else:
            return self.elem_t * self.df(self.elem_t, x)

    def update(self, elem_t):
        self.elem_t = elem_t.view(*elem_t.shape, 1)


class SONODE(NODE):
    def forward(self, t, x):
        """
        Compute [y y']' = [y' y''] = [y' df(t, y, y')]
        :param t: time, shape [1]
        :param x: [y y'], shape [batch, 2, vec]
        :return: [y y']', shape [batch, 2, vec]
        """
        self.nfe += 1
        v = x[:, 1:, :]
        out = self.df(t, x)
        return torch.cat((v, out), dim=1)


class HeavyBallNODE(NODE):
    def __init__(self, df, actv_h=None, gamma_guess=-3.0, gamma_act='sigmoid', corr=-100, corrf=True, sign=1):
        super().__init__(df)
        # Momentum parameter gamma
        self.gamma = Parameter([gamma_guess], frozen=False)
        self.gammaact = nn.Sigmoid() if gamma_act == 'sigmoid' else gamma_act
        self.corr = Parameter([corr], frozen=corrf)
        self.sp = nn.Softplus()
        self.sign = sign # Sign of df
        self.actv_h = nn.Identity() if actv_h is None else actv_h # Activation for dh, GHBNODE only

    def forward(self, t, x):
        """
        Compute [theta' m' v'] with heavy ball parametrization in
        $$ h' = -m $$
        $$ m' = sign * df - gamma * m $$
        based on paper https://www.jmlr.org/papers/volume21/18-808/18-808.pdf
        :param t: time, shape [1]
        :param x: [theta m], shape [batch, 2, dim]
        :return: [theta' m'], shape [batch, 2, dim]
        """
        self.nfe += 1
        h, m = torch.split(x, 1, dim=1)
        dh = self.actv_h(- m)
        dm = self.df(t, h) * self.sign - self.gammaact(self.gamma()) * m
        dm = dm + self.sp(self.corr()) * h
        out = torch.cat((dh, dm), dim=1)
        if self.elem_t is None:
            return out
        else:
            return self.elem_t * out

    def update(self, elem_t):
        self.elem_t = elem_t.view(*elem_t.shape, 1, 1)


HBNODE = HeavyBallNODE # Alias


class ODE_RNN(nn.Module):
    def __init__(self, ode, rnn, nhid, ic, rnn_out=False, both=False, tol=1e-7):
        super().__init__()
        self.ode = ode
        self.t = torch.Tensor([0, 1])
        self.nhid = [nhid] if isinstance(nhid, int) else nhid
        self.rnn = rnn
        self.tol = tol
        self.rnn_out = rnn_out
        self.ic = ic
        self.both = both

    def forward(self, t, x, multiforecast=None):
        """
        --
        :param t: [time, batch]
        :param x: [time, batch, ...]
        :return: [time, batch, *nhid]
        """
        n_t, n_b = t.shape
        h_ode = torch.zeros(n_t + 1, n_b, *self.nhid, device=x.device)
        h_rnn = torch.zeros(n_t + 1, n_b, *self.nhid, device=x.device)
        if self.ic:
            h_ode[0] = h_rnn[0] = self.ic(rearrange(x, 't b c -> b (t c)')).view(h_ode[0].shape)
        if self.rnn_out:
            for i in range(n_t):
                self.ode.update(t[i])
                h_ode[i] = odeint(self.ode, h_rnn[i], self.t, atol=self.tol, rtol=self.tol)[-1]
                h_rnn[i + 1] = self.rnn(h_ode[i], x[i])
            out = (h_rnn,)
        else:
            for i in range(n_t):
                self.ode.update(t[i])
                h_rnn[i] = self.rnn(h_ode[i], x[i])
                h_ode[i + 1] = odeint(self.ode, h_rnn[i], self.t, atol=self.tol, rtol=self.tol)[-1]
            out = (h_ode,)

        if self.both:
            out = (h_rnn, h_ode)

        if multiforecast is not None:
            self.ode.update(torch.ones_like((t[0])))
            forecast = odeint(self.ode, out[-1][-1], multiforecast * 1.0, atol=self.tol, rtol=self.tol)
            out = (*out, forecast)

        return out


class ODE_RNN_with_Grad_Listener(nn.Module):
    def __init__(self, ode, rnn, nhid, ic, rnn_out=False, both=False, tol=1e-7):
        super().__init__()
        self.ode = ode
        self.t = torch.Tensor([0, 1])
        self.nhid = [nhid] if isinstance(nhid, int) else nhid
        self.rnn = rnn
        self.tol = tol
        self.rnn_out = rnn_out
        self.ic = ic
        self.both = both

    def forward(self, t, x, multiforecast=None, retain_grad=True):
        """
        --
        :param t: [time, batch]
        :param x: [time, batch, ...]
        :return: [time, batch, *nhid]
        """
        n_t, n_b = t.shape
        h_ode = [None] * (n_t + 1)
        h_rnn = [None] * (n_t + 1)
        h_ode[-1] = h_rnn[-1] = torch.zeros(n_b, *self.nhid)

        if self.ic:
            h_ode[0] = h_rnn[0] = self.ic(rearrange(x, 't b c -> b (t c)')).view((n_b, *self.nhid))
        else:
            h_ode[0] = h_rnn[0] = torch.zeros(n_b, *self.nhid, device=x.device)
        if self.rnn_out:
            for i in range(n_t):
                self.ode.update(t[i])
                h_ode[i] = odeint(self.ode, h_rnn[i], self.t, atol=self.tol, rtol=self.tol)[-1]
                h_rnn[i + 1] = self.rnn(h_ode[i], x[i])
            out = (h_rnn,)
        else:
            for i in range(n_t):
                self.ode.update(t[i])
                h_rnn[i] = self.rnn(h_ode[i], x[i])
                h_ode[i + 1] = odeint(self.ode, h_rnn[i], self.t, atol=self.tol, rtol=self.tol)[-1]
            out = (h_ode,)

        if self.both:
            out = (h_rnn, h_ode)

        out = [torch.stack(h, dim=0) for h in out]

        if multiforecast is not None:
            self.ode.update(torch.ones_like((t[0])))
            forecast = odeint(self.ode, out[-1][-1], multiforecast * 1.0, atol=self.tol, rtol=self.tol)
            out = (*out, forecast)

        if retain_grad:
            self.h_ode = h_ode
            self.h_rnn = h_rnn
            for i in range(n_t + 1):
                if self.h_ode[i].requires_grad:
                    self.h_ode[i].retain_grad()
                if self.h_rnn[i].requires_grad:
                    self.h_rnn[i].retain_grad()

        return out