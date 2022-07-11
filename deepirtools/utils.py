import random
import math
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
from typing import List, Optional
from itertools import chain


def manual_seed(seed: int):
    """Set random seed to ensure reproducible results."""
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
            print("\rEpoch = {:7d}".format(epoch),
                  "Iter. = {:6d}".format(global_step),
                  "Cur. loss = {:7.2f}".format(loss),
                  "  Intervals no change = {:3d}".format(self.loss_improvement_counter),
                  end = "")
        if self.converged:
            print("\n")


def get_thresholds(rng, n_cat):
    lower_prob = 1/(1+math.exp(-rng[0]))
    upper_prob = 1/(1+math.exp(-rng[1]))
    probs = torch.linspace(lower_prob, upper_prob, n_cat + 1)[1:-1]
    thresholds = -probs.pow(-1).add(-1).log()
    return thresholds

        
def invert_factors(mat: np.ndarray):
    """
    For each factor, flip sign if sum of loadings is negative.
    
    Args:
        mat (ndarray): Loadings matrix.
    """
    mat = mat.copy()
    for col_idx in range(0, mat.shape[1]): 
        if np.sum(mat[:, col_idx]) < 0: 
            mat[:, col_idx] = -mat[:, col_idx]
    return mat


def invert_cov(cov: np.ndarray,
               mat: np.ndarray):
    """
    Flip covariances according to loadings signs.
    
    Args:
        cov (ndarray): Covariance matrix.
        mat (ndarray): Loadings matrix.
    """
    cov = cov.copy()
    for col_idx in range(0, mat.shape[1]):
        if np.sum(mat[:, col_idx]) < 0:
            # Invert column and row.
            inv_col_idxs = np.delete(np.arange(cov.shape[1]), col_idx, 0)
            cov[:, inv_col_idxs] = -cov[:, inv_col_idxs]
            cov[inv_col_idxs, :] = -cov[inv_col_idxs, :]
    return cov


def normalize_loadings(mat: np.ndarray):
    """
    Convert loadings to normal ogive metric.
    
    Args:
        mat (ndarray): Loadings matrix.
    """
    mat = mat.copy() / 1.702
    scale_const = np.sqrt(1 + np.sum(mat**2, axis = 1))
    return (mat.T / scale_const).T


def normalize_ints(ints:   np.ndarray,
                   mat:    np.ndarray,
                   n_cats: List[int]):
    """
    Convert intercepts to normal ogive metric.
    
    Args:
        ints   (ndarray):     Intercepts vector.
        mat    (ndarray):     Loadings matrix.
        n_cats (List of int): Number of categories for each item.
    """
    n_cats = [1] + n_cats
    idxs = np.cumsum([n_cat - 1 for n_cat in n_cats])
    sliced_ints = [ints[idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]
    mat = mat.copy() / 1.702
    scale_const = np.sqrt(1 + np.sum(mat**2, axis = 1))
    return np.hstack([sliced_int / scale_const[i] for i, sliced_int in enumerate(sliced_ints)])
    
    
class tensor_dataset(Dataset):
    def __init__(self,
                 data,
                 mask = None,
                ):
        self.data = data
        self.mask = mask
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d_samp = self.data[idx]
        if self.mask is not None:
            mask_samp = self.mask[idx]
            return d_samp, mask_samp
        return d_samp