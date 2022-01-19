#!/usr/bin/env python
#
# Author: Christopher J. Urban
#
# Purpose: Helpful layers for building models.
#
###############################################################################

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils import *

class CatBiasReshape(nn.Module):
    
    def __init__(self,
                 n_cats,
                 device):
        super(CatBiasReshape, self).__init__()
        self.n_cats = n_cats
        
        # Construct biases.
        bias = torch.empty(sum([n_cat - 1 for n_cat in self.n_cats]))
        bias.data = torch.from_numpy(np.hstack([logistic_thresholds(n_cat) for 
                                                n_cat in self.n_cats]))
        idxs = np.cumsum([n_cat - 1 for n_cat in ([1] + self.n_cats)])
        sliced_bias = [bias[idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]
        self.bias_reshape = nn.Parameter(torch.cat([F.pad(_slice,
                                                          (0, max(n_cats) - _slice.size(0) - 1),
                                                          value=9999.).unsqueeze(0) for
                                                    _slice in sliced_bias], axis = 0))
        
        # Construct drop indices.
        nan_mask = torch.cat([F.pad(_slice, (0, max(self.n_cats) - _slice.size(0) - 1),
                                    value=float("nan")).unsqueeze(0) for
                                    _slice in sliced_bias], axis = 0)
        cum_probs_mask = nan_mask * 0. + 1.
#        probs_mask = F.pad(cum_probs_mask, (1, 0), value=1.)
        self.drop_idxs = ~cum_probs_mask.view(-1).isnan().to(device)

    def forward(self, x):
        return self.bias_reshape + x
    
    @property
    def bias(self):
        return self.bias_reshape.view(-1)[self.drop_idxs]
    
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
        
            return cart2spher(l_mat)
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