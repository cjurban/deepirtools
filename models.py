#!/usr/bin/env python
#
# Purpose: 
#
###############################################################################

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.distributions as dist
from utils import *
from helper_layers import *
from base_class import BaseClass
from read_data import csv_dataset
import timeit

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from factor_analyzer import Rotator
from pylab import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from typing import List

EPS = 1e-7


class CatBiasReshape(nn.Module):
    
    def __init__(self,
                 n_cats,
                 mask=None):
        super(CatBiasReshape, self).__init__()
        self.n_cats = n_cats
        
        # Construct biases.
        bias = torch.empty(sum([n_cat - 1 for n_cat in n_cats]))
        idxs = np.cumsum([n_cat - 1 for n_cat in ([1] + n_cats)])
        bias.data = torch.from_numpy(np.hstack([get_thresholds([mask[idx].item() * -4, 4], n_cat) for
                                                idx, n_cat in zip(idxs[:-1], n_cats)]))
        bias_reshape, sliced_bias = self.reshape(bias, idxs, 9999.)
        self.bias_reshape = nn.Parameter(bias_reshape)
        if mask is not None:
            self.mask = self.reshape(mask, idxs, 0., False)
        
        # Construct drop indices.
        nan_mask = torch.cat([F.pad(_slice, (0, max(self.n_cats) - _slice.size(0) - 1),
                                    value=float("nan")).unsqueeze(0) for
                                    _slice in sliced_bias], axis = 0)
        cum_probs_mask = nan_mask * 0. + 1.
        self.drop_idxs = ~cum_probs_mask.view(-1).isnan().to(mask.device)
        
    def reshape(self, t, idxs, pad_val, return_slices=True):
        sliced_t = [t[idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]
        t_reshape = torch.cat([F.pad(_slice,
                                     (0, max(self.n_cats) - _slice.size(0) - 1),
                                      value=pad_val).unsqueeze(0) for
                                      _slice in sliced_t], axis = 0)
        if return_slices:
            return t_reshape, sliced_t
        return t_reshape

    def forward(self, x):
        if mask is not None:
            return (self.bias_reshape * self.mask) + x
        return self.bias_reshape + x
    
    @property
    def bias(self):
        if self.mask is not None:
            return self.bias_reshape.view(-1)[self.drop_idxs] * self.mask.view(-1)
        return self.bias_reshape.view(-1)[self.drop_idxs]
    
    
class SparseLinear(nn.Module):
    
    def __init__(self,
                 in_features,
                 out_features,
                 Q):
        super(SparseLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.Q = Q
        
        self.free_weight = nn.Parameter(torch.empty(in_features, out_features))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.free_weight, mean=1., std=0.001)
        self.free_weight.data *= self.Q
        
    def forward(self, x):
        return F.linear(x, self.weight, None)
    
    @property
    def weight(self):
        return self.free_weight * self.Q
    
    
# Module for a weight matrix with linear equality constraints.
class LinearConstraints(nn.Module):
    
    def __init__(self,
                 in_features,
                 out_features,
                 A,
                 b = None):
        """
        Args:
            in_features  (int): Size of each input sample.
            out_features (int): Size of each output sample.
            A            (Tensor): Matrix implementing linear constraints.
            b            (Tensor): Vector implementing linear constraints.
        """
        super(LinearConstraints, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.A = A
        self.b = b
        self.theta = nn.Parameter(torch.empty(self.A.shape[0]))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        return nn.init.normal_(self.theta, mean=1., std=0.001)
        
    def forward(self, x):
        return F.linear(x, self.weight, None)
    
    @property
    def weight(self):
        return F.linear(self.theta, self.A, self.b).view(self.in_features, self.out_features).T


class CatProjector(nn.Module):
    def __init__(self, latent_size, n_cats, Q=None, A=None, b=None, ints_mask=None):
        super(CatProjector, self).__init__()
        
        # Loadings.
        if Q is not None:
            self.loadings = SparseLinear(latent_size, len(n_cats), Q)
        elif A is not None:
            self.loadings = LinearConstraints(latent_size, len(n_cats), A, b)
        else:
            self.loadings = nn.Linear(latent_size, len(n_cats), bias = False)
        self.Q = Q
        self.A = A
        
        # Intercepts.
        self.intercepts = CatBiasReshape(n_cats, ints_mask)
        self.n_cats = n_cats
        
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.Q is None and self.A is None:
            nn.init.xavier_uniform_(self.loadings.weight)

    def forward(self, x):
        Bx = self.loadings(x)
        cum_probs = self.intercepts(Bx.unsqueeze(-1).expand(Bx.shape +
                                                                 torch.Size([max(self.n_cats) - 1]))).sigmoid()
        upper_probs = F.pad(cum_probs, (0, 1), value=1.)
        lower_probs = F.pad(cum_probs, (1, 0), value=0.)
        probs = upper_probs - lower_probs
        
        return probs
    
    
# Module for the spherical parameterization of a covariance matrix.
class Spherical(nn.Module):
    
    def __init__(self,
                 dim,
                 correlated_factors,
                 device):
        """
        Args:
            dim                (int): Number of rows and columns.
            correlated_factors (list of int): List of correlated factors.
            device             (str): String specifying whether to run on CPU or GPU.
        """
        super(Spherical, self).__init__()
        
        self.dim = dim
        self.correlated_factors = correlated_factors
        self.device = device
        
        if self.correlated_factors != []:
            n_elts = int((self.dim * (self.dim + 1)) / 2)
            self.theta = nn.Parameter(torch.zeros([n_elts]))
            diag_idxs = torch.from_numpy(np.cumsum(np.arange(1, self.dim + 1)) - 1)
            self.theta.data[diag_idxs] = np.log(np.pi / 2)
            
            # For constraining specific covariances to zero.
            tril_idxs = torch.tril_indices(row = self.dim - 1, col = self.dim - 1, offset = 0)
            uncorrelated_factors = [factor for factor in np.arange(self.dim).tolist() if factor not in self.correlated_factors]
            self.uncorrelated_tril_idxs = tril_idxs[:, sum((tril_idxs[1,:] == factor) + (tril_idxs[0,:] == factor - 1) for
                                                           factor in uncorrelated_factors) > 0]
            
    def cart2spher(self, cart_mat):
        n = cart_mat.size(1)
        spher_mat = torch.zeros_like(cart_mat)
        cos_mat = cart_mat[:, 1:n].cos()
        sin_mat = cart_mat[:, 1:n].sin().cumprod(1)

        spher_mat[:, 0] = cart_mat[:, 0] * cos_mat[:, 0]
        spher_mat[:, 1:(n - 1)] = cart_mat[:, 0].unsqueeze(1) * sin_mat[:, 0:(n - 2)] * cos_mat[:, 1:(n - 1)]
        spher_mat[:, -1] = cart_mat[:, 0] * sin_mat[:, -1]

        return spher_mat
        
    @property
    def weight(self):
        if self.correlated_factors != []:
            tril_idxs = torch.tril_indices(row = self.dim, col = self.dim, offset = 0)
            theta_mat = torch.zeros(self.dim, self.dim, device=self.device)
            theta_mat[tril_idxs[0], tril_idxs[1]] = self.theta

            # Ensure the parameterization is unique.
            exp_theta_mat = torch.zeros(self.dim, self.dim, device=self.device)
            exp_theta_mat[tril_idxs[0], tril_idxs[1]] = self.theta.exp()
            lower_tri_l_mat = (np.pi * exp_theta_mat) / (1 + theta_mat.exp())
            l_mat = exp_theta_mat.diag().diag_embed() + lower_tri_l_mat.tril(diagonal = -1)

            # Constrain variances to one.
            l_mat[:, 0] = torch.ones(l_mat.size(0), device=self.device)
            
            # Constrain specific covariances to zero.
            l_mat[1:, 1:].data[self.uncorrelated_tril_idxs[0], self.uncorrelated_tril_idxs[1]] = np.pi / 2
        
            return self.cart2spher(l_mat)
        else:
            return torch.eye(self.dim, device=self.device)

    def forward(self, x):
        return F.linear(x, self.weight)
    
    @property
    def cov(self):
        if self.correlated_factors != []:
            weight = self.weight

            return torch.matmul(weight, weight.t())
        else:
            return torch.eye(self.dim, device=self.device)
    
    def inv_cov(self):
        if self.correlated_factors != []:
            return torch.cholesky_solve(torch.eye(self.dim, device=self.device),
                                        self.weight)
        else:
            return torch.eye(self.dim, device=self.device)

    
# Variational autoencoder for MIRT module.
class MIRTVAE(nn.Module):
    
    def __init__(self,
         input_size:            int,
         inference_model_sizes: List[int],
         latent_size:           int,
         n_cats:                List[int],
         Q,
         A,
         b,
         correlated_factors,
         device:               torch.device):


        """
        Args:
            input_size            (int): Input vector dimension.
            inference_model_sizes (list of int): Inference model neural network layer dimensions.
            latent_size           (int): Latent vector dimension.
            n_cats                (list of int): List containing number of categories for each observed variable.
            device                (str): String specifying whether to run on CPU or GPU.
        """
        super(MIRTVAE, self).__init__()
        
        # Inference model neural network.
        inf_sizes = [input_size] + inference_model_sizes
        inf_list = sum( ([nn.Linear(size1, size2), nn.ELU()] for size1, size2 in
                          zip(inf_sizes[0:-1], inf_sizes[1:])), [] )
        if inf_list != []:
            self.inf_net = nn.Sequential(*inf_list)
        else:
            self.inf_net = nn.Linear(inf_sizes[0], int(2 * latent_size))
        
        self.projector = CatProjector(latent_size, n_cats, Q, A, b)
        
        self.cholesky = Spherical(latent_size, correlated_factors, device)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.inf_net[-1].weight, mean=0., std=0.001)
        nn.init.normal_(self.inf_net[-1].bias[0:self.latent_size], mean=0., std=0.001)
        nn.init.normal_(self.inf_net[-1].bias[self.latent_size:], mean=math.log(math.exp(1) - 1), std=0.001)

    def encode(self,
               x,
               mc_samples,
               iw_samples):
        hidden = self.inf_net(x)

        # Expand for Monte Carlo samples.
        hidden = hidden.unsqueeze(0).expand(torch.Size([mc_samples]) + hidden.shape)
        
        # Expand for importance-weighted samples.
        hidden = hidden.unsqueeze(0).expand(torch.Size([iw_samples]) + hidden.shape)
        
        mu, std = hidden.chunk(chunks = 2, dim = -1)
        std = F.softplus(std)
            
        return mu, std + EPS

    def forward(self,
                x,
                mc_samples = 1,
                iw_samples = 1):
        mu, std = self.encode(x, mc_samples, iw_samples)
        z = mu + std * torch.randn_like(mu)
        recon_x = self.projector(z)
        
        return recon_x, mu, std, z