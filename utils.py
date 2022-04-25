#!/usr/bin/env python
#
# Purpose:
#
###############################################################################

import pickle
import math
import torch
from torch import nn
import numpy as np
from scipy.stats import logistic
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset

EPS = 1e-16

# Save an object.
# https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file
def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        

# Load an object.
# https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file
def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)


# Caclucate thresholds with equal area under the logistic distribution.
def logistic_thresholds(n_cats):
    thresholds = [logistic.ppf((cat + 1)/ n_cats) for cat in range(n_cats - 1)]
    return np.asarray(thresholds, dtype = np.float32)
        
# Convert covariance matrix to correlation matrix.
# http://www.statsmodels.org/0.6.1/_modules/statsmodels/stats/moment_helpers.html
def cov2corr(cov,
             return_std = False):
    cov = np.asanyarray(cov)
    std_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_, std_)

    return corr
        
# Change factor loadings signs if sum of loadings is negative.
def invert_factors(mat):
    mat = mat.copy()
    for col_idx in range(0, mat.shape[1]): 
        if np.sum(mat[:, col_idx]) < 0: 
            mat[:, col_idx] = -mat[:, col_idx]
            
    return mat

# Change covariance signs if sum of loadings is negative.
def invert_cov(cov,
               mat):
    cov = cov.copy()
    for col_idx in range(0, mat.shape[1]):
        if np.sum(mat[:, col_idx]) < 0:
            # Invert column and row.
            inv_col_idxs = np.delete(np.arange(cov.shape[1]), col_idx, 0)
            cov[:, inv_col_idxs] = -cov[:, inv_col_idxs]
            cov[inv_col_idxs, :] = -cov[inv_col_idxs, :]
            
    return cov

# Convert factor loadings from raw to normal metric.
def normalize_loadings(mat):
    mat = mat.copy() / 1.702
    scale_const = np.sqrt(1 + np.sum(mat**2, axis = 1))
    
    return (mat.T / scale_const).T

# Convert intercepts from raw to normal metric.
def normalize_ints(ints,
                   mat,
                   n_cats):
    n_cats = [1] + n_cats
    idxs = np.cumsum([n_cat - 1 for n_cat in n_cats])
    sliced_ints = [ints[idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]
    mat = mat.copy() / 1.702
    scale_const = np.sqrt(1 + np.sum(mat**2, axis = 1))
    
    return np.hstack([sliced_int / scale_const[i] for i, sliced_int in enumerate(sliced_ints)])

# Convert factor loadings from normal to raw metric.
def unnormalize_loadings(mat):
    mat = mat.copy()
    ss = np.sum(mat**2, axis = 1)
    scale_const = np.sqrt(1 + (ss / (1 - ss)))
    
    return 1.702 * (mat.T * scale_const).T

# Run Hungarian algorithm.
def hungarian_algorithm(ref_mat,
                        perm_mat,
                        normalize_ldgs = True):
    # Invert factors and normalize loadings.
    ref_mat, perm_mat = invert_factors(ref_mat), invert_factors(perm_mat)
    if normalize_ldgs:
        ref_mat, perm_mat = normalize_loadings(ref_mat), normalize_loadings(perm_mat)
    
    # Get MSE cost matrix between each column in reference and column-permuted matrix.
    cost_mat = np.empty((ref_mat.shape[1], ref_mat.shape[1], ))
    cost_mat[:] = np.nan
    for ref_col in range(ref_mat.shape[1]): 
        for perm_col in range(perm_mat.shape[1]): 
            cost_mat[ref_col, perm_col] = np.sum((ref_mat[:, ref_col] - perm_mat[:, perm_col])**2)
    
    return linear_sum_assignment(cost_mat)[1], ref_mat, perm_mat

# Calculate biases between a reference matrix and column-permutation of another matrix
# that gives the smallest MSE.
def permuted_biases(ref_mat,
                    perm_mat,
                    main_loadings_only = False,
                    cross_loadings_only = False,
                    normalize_ldgs = True):
    # Run Hungarian algorithm.
    col_idxs, ref_mat, perm_mat = hungarian_algorithm(ref_mat, perm_mat, normalize_ldgs)
    
    # Get biases between reference and column-permuted matrix.
    if main_loadings_only:
        biases = perm_mat[:, col_idxs][ref_mat != 0] - ref_mat[ref_mat != 0]
    elif cross_loadings_only:
        biases = perm_mat[:, col_idxs][ref_mat == 0] - ref_mat[ref_mat == 0]
    else:
        biases = perm_mat[:, col_idxs] - ref_mat
    
    return biases

# Calculate biases between a reference covariance matrix and a permuted covariance matrix.
def cov_biases(ref_cov,
               perm_cov,
               ref_mat,
               perm_mat):
    # Invert covariance matrices.
    ref_cov, perm_cov = invert_cov(ref_cov, ref_mat), invert_cov(perm_cov, perm_mat)
    
    # Run Hungarian algorithm.
    col_idxs = hungarian_algorithm(ref_mat, perm_mat)[0]
    
    # Get RMSE between reference and permuted covariance matrix.
    biases = (perm_cov[np.ix_(col_idxs, col_idxs)] - ref_cov)[np.triu_indices(ref_cov.shape[0], k = 1)]
    
    return biases

# Calculate congruence coefficient between a reference matrix and column-permutation of another matrix
# that gives the smallest MSE.
def permuted_congruence(ref_mat,
                        perm_mat,
                        mean = True):
    # Run Hungarian algorithm.
    col_idxs, ref_mat, perm_mat = hungarian_algorithm(ref_mat, perm_mat)
    
    # Get congruence coefficient between reference and column-permuted matrix.
    congruences = [np.dot(ref_mat[:, col], perm_mat[:, col_idxs][:, col]) /
                   np.sqrt(np.clip(np.sum(ref_mat[:, col]**2), a_min = EPS, a_max = None) *
                           np.clip(np.sum(perm_mat[:, col_idxs][:, col]**2), a_min = EPS, a_max = None)) for
                   col in range(ref_mat.shape[1])]

    if mean:
        return np.mean(congruences)
    else:
        return congruences
    
    
class tensor_dataset(Dataset):
    def __init__(self, data):
        """
        Args:
        """
        self.df = data
        
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # data is formatted as an numpy array: 
        # sample = self.df.iloc[idx, :].to_numpy()
        sample = self.df.iloc[idx, :].to_numpy()

        return torch.from_numpy(sample)
    
    def num_columns(self):
        return self.df.shape[1]
    
                
def screeplot(latent_sizes:           List[int], # list of dimensions in ascending order
              csv_data:              pd.DataFrame,
              categories:            List[int],
              n_cats:                List[int], 
              which_split:           str, 
              test_size:             float,
              data_loader_kwargs:    dict = {},
              learning_rate:         float = 5e-3,
              device:                torch.device = "cpu",
              log_interval:          int = 100,
              steps_anneal:          int = 1000,
              iw_samples_training:   int = 5,
              iw_samples_bic:        int = 5000,
              xlabel:                str = "Number of Factors",
              ylabel:                str = "Predicted Approximate Negative Log-Likelihood",
              title:                 str = "Approximate Log-Likelihood Scree Plot"):
    seed = 1
    ll_ls = []
    
    csv_train, csv_test = train_test_split(csv_data, train_size = 1 - test_size, test_size = test_size, random_state = 45)
    
    csv_test = pd.DataFrame(csv_test)
            
    for latent_size in latent_sizes:

        print("\rStarting fitting for P =", latent_size, end="")

        # Set random seeds.
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize model.
        start = timeit.default_timer()

        ipip_vae = MIRTVAEClass(input_size = csv_data.shape[1],
                                inference_model_sizes = [int(np.ceil(csv_data.shape[1]/2))], # adjust NN size for different P
                                latent_size = latent_size,
                                n_cats = n_cats,
                                learning_rate = learning_rate,
                                device = device,
                                log_interval = log_interval,
                                steps_anneal = steps_anneal,
                                Q = None,
                                A = None,
                                b = None)

        # Fit model.
        # run training on training set
        
        ipip_vae.run_training(csv_data, categories, iw_samples = iw_samples_training)
        stop = timeit.default_timer()
        
        
        print("\rModel fitted for P =", latent_size, ", run time =", round(stop - start, 2), "seconds", end = "")

        # Save predicted approximate negative log-likelihood.
        torch.manual_seed(seed)
        np.random.seed(seed)
        start = timeit.default_timer()
        
        bic_result = ipip_vae.bic(csv_test, iw_samples = iw_samples_bic)
        ll_ls.append(bic_result[1]) # I set iw_samples = 5000 in the paper
        stop = timeit.default_timer()
   
        # print("Approx. LL stored, computed in", round(stop - start, 2), "seconds", end="")

    print("\n")
    for idx, dim in enumerate(latent_sizes):
        print("Latent dimension =", dim, "Approx. LL =", round(-ll_ls[idx], 2))

        
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
    return ll_ls, ipip_vae

    
# needs to be generalized to include any # factors
def loadings_heatmap(rot_loadings: [float], 
                     loadings:     [float],
                     x_label:      str = "Factor", 
                     y_label:      str = "Item", 
                     title:        str = "Factor Loadings"):
    norm_loadings = normalize_loadings(rot_loadings)
    c = pcolor(invert_factors(norm_loadings))
    set_cmap("gray_r")
    colorbar() 
    c = pcolor(invert_factors(norm_loadings), edgecolors = "w", linewidths = 1, vmin = 0) 
    xlabel(x_label)
    ylabel(y_label)
    xticks(np.arange(loadings.shape[1]) + 0.5,
           ["1", "2", "3", "4", "5"])
    yticks(np.array([10, 20, 30, 40, 50]) - 0.5, ["10", "20", "30", "40", "50"])
    plt.gca().invert_yaxis()
    suptitle(title, y = 0.93)
    savefig("loadings_heatmap.pdf")