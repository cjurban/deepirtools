import random
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
from scipy.stats import logistic
from sklearn.model_selection import train_test_split
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from pylab import *


def logistic_thresholds(n_cats: List[int]):
    """"""
    thresholds = [logistic.ppf((cat + 1)/ n_cats) for cat in range(n_cats - 1)]
    return np.asarray(thresholds, dtype = np.float32)

        
def invert_factors(mat: np.ndarray):
    """"""
    mat = mat.copy()
    for col_idx in range(0, mat.shape[1]): 
        if np.sum(mat[:, col_idx]) < 0: 
            mat[:, col_idx] = -mat[:, col_idx]
    return mat


def invert_cov(cov: np.ndarray,
               mat: np.ndarray):
    """"""
    cov = cov.copy()
    for col_idx in range(0, mat.shape[1]):
        if np.sum(mat[:, col_idx]) < 0:
            # Invert column and row.
            inv_col_idxs = np.delete(np.arange(cov.shape[1]), col_idx, 0)
            cov[:, inv_col_idxs] = -cov[:, inv_col_idxs]
            cov[inv_col_idxs, :] = -cov[inv_col_idxs, :]
    return cov


def normalize_loadings(mat: np.ndarray):
    """"""
    mat = mat.copy() / 1.702
    scale_const = np.sqrt(1 + np.sum(mat**2, axis = 1))
    return (mat.T / scale_const).T


def normalize_ints(ints:   np.ndarray,
                   mat:    np.ndarray,
                   n_cats: List[int]):
    """"""
    n_cats = [1] + n_cats
    idxs = np.cumsum([n_cat - 1 for n_cat in n_cats])
    sliced_ints = [ints[idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]
    mat = mat.copy() / 1.702
    scale_const = np.sqrt(1 + np.sum(mat**2, axis = 1))
    return np.hstack([sliced_int / scale_const[i] for i, sliced_int in enumerate(sliced_ints)])


def unnormalize_loadings(mat: np.ndarray):
    """"""
    mat = mat.copy()
    ss = np.sum(mat**2, axis = 1)
    scale_const = np.sqrt(1 + (ss / (1 - ss)))
    return 1.702 * (mat.T * scale_const).T
    
    
class tensor_dataset(Dataset):
    def __init__(self,
                 data,
                 mask = None,
                ):
        """
        Args:
        """
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d_samp = self.data[idx]
        if self.mask is not None:
            mask_samp = self.mask[idx]
            return d_samp, mask_samp
        return d_samp
    
                
def screeplot(latent_sizes:             List[int], # list of dimensions in ascending order
              data:                     torch.Tensor,
              n_cats:                   List[int],
              test_size:                float,
              inference_net_sizes_list: List[List[int]],
              learning_rates:           List[float],
              missing_mask:             Optional[torch.Tensor],
              max_epochs:               int = 100000,
              batch_size:               int = 32,
              device:                   torch.device = "cpu",
              log_interval:             int = 100,
              iw_samples_fit:           int = 1,
              iw_samples_ll:            int = 5000,
              random_seed:              int = 1,
              xlabel:                   str = "Number of Factors",
              ylabel:                   str = "Predicted Approximate Negative Log-Likelihood",
              title:                    str = "Approximate Log-Likelihood Scree Plot",
             ):
    """"""
    assert(test_size > 0 and test_size < 1)
    data_train, data_test = train_test_split(data, train_size = 1 - test_size, test_size = test_size)
    n_items = data.size(1)
            
    manual_seed(random_seed)
    ll_list = []
    for idx, latent_size in enumerate(latent_sizes):
        print("\nLatent size = ", latent_size, end="\n")
        model = GRMEstimator(input_size = n_items,
                             inference_net_sizes = inference_net_sizes_list[idx],
                             latent_size = latent_size,
                             n_cats = n_cats,
                             learning_rate = learning_rates[idx],
                             device = device,
                             log_interval = log_interval,
                            )
        ipip_vae.fit(data, batch_size, missing_mask, max_epochs, iw_samples = iw_samples_fit)

        ll = model.log_likelihood(data_test, iw_samples = iw_samples_ll)
        ll_list.append(ll)
        
    fig, ax = plt.subplots(constrained_layout = True)
    fig.set_size_inches(5, 5, forward = True)
    fig.set_size_inches(5, 5, forward = True)
    
    ax.plot(latent_sizes, [ll for ll in ll_ls], "k-o")
    
    ax.set_xticks(np.arange(min(latent_sizes) - 1, max(latent_sizes) + 2).tolist())
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    fig.suptitle(title)
    fig.show()

    pdf = matplotlib.backends.backend_pdf.PdfPages("scree_plot.pdf")
    pdf.savefig(fig, dpi = 300)
    pdf.close()
    
    return ll_list

    
def loadings_heatmap(loadings:     np.ndarray,
                     x_label:      str = "Factor", 
                     y_label:      str = "Item", 
                     title:        str = "Factor Loadings"):
    latent_size = loadings.shape[1]
    
    c = pcolor(invert_factors(loadings))
    set_cmap("gray_r")
    colorbar() 
    c = pcolor(invert_factors(loadings), edgecolors = "w", linewidths = 1, vmin = 0) 
    xlabel(x_label)
    ylabel(y_label)
    xticks(np.arange(latent_size) + 0.5,
           [str(size + 1) for size in range(latent_size)])
    ytick_vals = [int(10 * (size + 1)) for size in range(latent_size)]
    yticks(np.array(ytick_vals) - 0.5, [str(val) for val in ytick_vals])
    plt.gca().invert_yaxis()
    suptitle(title, y = 0.93)
    
    plt.show()
    savefig("loadings_heatmap.pdf")
    
    
def manual_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)