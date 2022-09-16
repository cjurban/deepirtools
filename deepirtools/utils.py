import random
import math
import torch
from torch import nn
from torch.utils.data import Dataset
from pyro.distributions import Categorical, Distribution
import numpy as np
from typing import List, Optional
from itertools import chain


def manual_seed(seed: int):
    """Set random seeds to ensure reproducible results."""
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
class ConvergenceChecker():
    
    def __init__(self, n_intervals = 100, log_interval = 1):
        self.n_intervals = n_intervals
        self.log_interval = log_interval
        self.converged = False
        self.best_avg_loss = None
        self.loss_list = []
        self.loss_improvement_counter = 0
    
    def check_convergence(self, epoch, global_step, loss):
        self.loss_list.append(loss)
        if len(self.loss_list) > 100:
            self.loss_list.pop(0)
        if (global_step - 1) % 100 == 0 and global_step != 1:
            cur_mean_loss = np.mean(self.loss_list)
            if self.best_avg_loss is None:
                self.best_avg_loss = cur_mean_loss
            elif cur_mean_loss < self.best_avg_loss:
                self.best_avg_loss = cur_mean_loss
                if self.loss_improvement_counter >= 1:
                    self.loss_improvement_counter = 0
            elif cur_mean_loss >= self.best_avg_loss:
                self.loss_improvement_counter += 1
                if self.loss_improvement_counter >= self.n_intervals:
                    self.converged = True
        if (global_step - 1) % self.log_interval == 0:
            print("\rEpoch = {:8d}".format(epoch),
                  "Iter. = {:8d}".format(global_step),
                  "Cur. loss = {:7.2f}".format(loss),
                  "  Intervals no change = {:4d}".format(self.loss_improvement_counter),
                  end = "")


def get_thresholds(rng, n_cat):
    lower_prob = 1/(1+math.exp(-rng[0]))
    upper_prob = 1/(1+math.exp(-rng[1]))
    probs = torch.linspace(lower_prob, upper_prob, n_cat + 1)[1:-1]
    thresholds = -probs.pow(-1).add(-1).log()
    return thresholds

        
def invert_factors(mat: torch.Tensor):
    """
    For each factor, flip sign if sum of loadings is negative.
    
    Parameters
    __________
    mat : Tensor
        Loadings matrix.
        
    Returns
    -------
    inverted_mat : Tensor
        Inverted loadings matrix.
    """
    
    assert(len(mat.shape) == 2), "Loadings matrix must be 2D."
    mat = mat.clone()
    for col_idx in range(mat.shape[1]): 
        if mat[:, col_idx].sum() < 0: 
            mat[:, col_idx] = -mat[:, col_idx]
    return mat


def invert_cov(cov: torch.Tensor,
               mat: torch.Tensor,
              ):
    """
    Flip factor covariances according to loadings signs.
    
    Parameters
    __________
    cov : Tensor
        Factor covariance matrix.
    mat : Tensor
        Loadings matrix.
        
    Returns
    -------
    inverted_cov : Tensor
        Inverted factor covariance matrix.
    """
    
    assert(len(cov.shape) == 2), "Factor covariance matrix must be 2D."
    assert(len(mat.shape) == 2), "Loadings matrix must be 2D."
    cov = cov.clone()
    for col_idx in range(mat.shape[1]):
        if mat[:, col_idx].sum() < 0:
            inv_col_idxs = np.delete(np.arange(cov.shape[1]), col_idx, 0)
            cov[:, inv_col_idxs] = -cov[:, inv_col_idxs]
            cov[inv_col_idxs, :] = -cov[inv_col_idxs, :]
    return cov



def invert_mean(mean: torch.Tensor,
                mat: torch.Tensor,
               ):
    """
    Flip factor means according to loadings signs.
    
    Parameters
    __________
    mean : Tensor
        Factor mean vector.
    mat : Tensor
        Loadings matrix.
        
    Returns
    -------
    inverted_mean : Tensor
        Inverted factor mean vector.
    """
    
    assert(len(mean.shape) == 1), "Factor mean vector must be 1D."
    assert(len(mat.shape) == 2), "Loadings matrix must be 2D."
    mean = mean.clone()
    for col_idx in range(mat.shape[1]):
        if mat[:, col_idx].sum() < 0:
            mean[col_idx] = -mean[col_idx]
    return mean


def invert_latent_regression_weight(weight: torch.Tensor,
                                    mat: torch.Tensor,
                                   ):
    """
    Flip latent regression weights according to loadings signs.

    Parameters
    __________
    weight : Tensor
        Latent regression weight matrix.
    mat : Tensor
        Loadings matrix.
        
    Returns
    -------
    inverted_weight : Tensor
        Inverted latent regression weight matrix.
    """
    
    assert(len(weight.shape) == 2), "Latent regression weight matrix must be 2D."
    assert(len(mat.shape) == 2), "Loadings matrix must be 2D."
    weight = weight.clone()
    for col_idx in range(mat.shape[1]):
        if mat[:, col_idx].sum() < 0:
            weight[col_idx, :] = -weight[col_idx, :]
    return weight


def normalize_loadings(mat: torch.Tensor):
    """
    Convert loadings to normal ogive metric (only for IRT models).
    
    Parameters
    __________
    mat : Tensor
        Loadings matrix.
        
    Returns
    -------
    normalized_mat : Tensor
        Normalized loadings matrix.
    """
    
    assert(len(mat.shape) == 2), "Loadings matrix must be 2D."
    mat = mat.clone().div(1.702)
    scale_const = mat.pow(2).sum(dim = 1).add(1).sqrt()
    return (mat.T / scale_const).T


def normalize_ints(ints:   torch.Tensor,
                   mat:    torch.Tensor,
                   n_cats: List[int],
                  ):
    """
    Convert intercepts to normal ogive metric (only for IRT models).
    
    Parameters
    __________
    ints : Tensor
        Intercepts vector.
    mat : Tensor
        Loadings matrix.
    n_cats : list of int
        Number of categories for each item.
        
    Returns
    -------
    normalized_ints : Tensor
        Normalized intercepts vector.
    """
    
    assert(len(ints.shape) == 1), "Intercepts vector must be 1D."
    assert(len(mat.shape) == 2), "Loadings matrix must be 2D."
    ints = ints.clone()
    n_cats = [1] + n_cats
    idxs = np.cumsum([n_cat - 1 for n_cat in n_cats])
    sliced_ints = [ints[idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]
    mat = mat.clone().div(1.702)
    scale_const = mat.pow(2).sum(dim = 1).add(1).sqrt()
    return torch.cat([sliced_int / scale_const[i] for i, sliced_int in enumerate(sliced_ints)],
                     dim = 0)
    
    
class MultiCategorical(Distribution):
    """
    Multiple categorical distributions.
    https://github.com/pytorch/pytorch/issues/43250
    """

    def __init__(self, dists: List[Categorical]):
        super().__init__()
        self.dists = dists

    def log_prob(self, value):
        ans = []
        for d, v in zip(self.dists, torch.split(value, 1, dim=-1)):
            ans.append(d.log_prob(v.squeeze(-1)))
        return torch.stack(ans, dim=-1).sum(dim=-1)

    def entropy(self):
        return torch.stack([d.entropy() for d in self.dists], dim=-1).sum(dim=-1)

    def sample(self, sample_shape=torch.Size()):
        return torch.stack([d.sample(sample_shape) for d in self.dists], dim=-1)


def multi_categorical_maker(nvec):
    def get_multi_categorical(probs):
        start = 0
        ans = []
        for n in nvec:
            ans.append(Categorical(probs=probs[..., start: start + n]))
            start += n
        return MultiCategorical(ans)
    return get_multi_categorical
    
    
class tensor_dataset(Dataset):
    
    def __init__(self,
                 data,
                 mask = None,
                 covariates = None,
                ):
        self.data = data
        self.mask = mask
        self.covariates = covariates
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch = {}
        batch["y"] = self.data[idx]
        if self.mask is not None:
            batch["mask"] = self.mask[idx]
        if self.covariates is not None:
            batch["covariates"] = self.covariates[idx]
        return batch