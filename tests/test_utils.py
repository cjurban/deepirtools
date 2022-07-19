import os
from os.path import join
import torch
import torch.nn.functional as F
import pyro.distributions as pydist
import numpy as np
from scipy.optimize import linear_sum_assignment
import subprocess


class BaseFactorModelSimulator():
    
    def __init__(self,
                 loadings:   torch.Tensor,
                 intercepts: torch.Tensor,
                 cov_mat:    torch.Tensor,
                ):
        """Base class for simulating from a latent factor model."""
        super(BaseFactorModelSimulator, self).__init__()

        self.loadings = loadings
        self.intercepts = intercepts
        self.cov_mat = cov_mat
        
    def sample(self):
        raise NotImplementedError

        
class PoissonFactorModelSimulator(BaseFactorModelSimulator):
    
    def __init__(self,
                 loadings:   torch.Tensor,
                 intercepts: torch.Tensor,
                 cov_mat:    torch.Tensor,
                ):
        """ Simulate from a Poisson factor model."""
        super().__init__(loadings = loadings, intercepts = intercepts, cov_mat = cov_mat)
        
    @torch.no_grad()    
    def sample(self,
               sample_size: int):
        latent_size = self.loadings.shape[1]
        x_dist = pydist.MultivariateNormal(loc = torch.zeros([1, latent_size]),
                                           covariance_matrix = self.cov_mat)
        x = x_dist.sample([sample_size]).squeeze(dim = -2)
        rate = F.linear(x, self.loadings, self.intercepts).exp()
        
        y_dist = pydist.Poisson(rate = rate)
        return y_dist.sample()
    
    
class NegativeBinomialFactorModelSimulator(BaseFactorModelSimulator):
    
    def __init__(self,
                 loadings:   torch.Tensor,
                 intercepts: torch.Tensor,
                 cov_mat:    torch.Tensor,
                 probs:      torch.Tensor,
                ):
        """Simulate from a negative binomial factor model."""
        super().__init__(loadings = loadings, intercepts = intercepts, cov_mat = cov_mat)
        
        self.probs = probs
        
    @torch.no_grad()    
    def sample(self,
               sample_size: int):
        latent_size = self.loadings.shape[1]
        x_dist = pydist.MultivariateNormal(loc = torch.zeros([1, latent_size]),
                                           covariance_matrix = self.cov_mat)
        x = x_dist.sample([sample_size]).squeeze(dim = -2)
        total_count = F.linear(x, self.loadings, self.intercepts).exp()
        
        y_dist = pydist.NegativeBinomial(total_count = total_count, probs = self.probs)
        return y_dist.sample()
    
    
class NormalFactorModelSimulator(BaseFactorModelSimulator):
    
    def __init__(self,
                 loadings:     torch.Tensor,
                 intercepts:   torch.Tensor,
                 cov_mat:      torch.Tensor,
                 residual_std: torch.Tensor
                ):
        """Simulate from a normal factor model."""
        super().__init__(loadings = loadings, intercepts = intercepts, cov_mat = cov_mat)
        
        self.residual_std = residual_std
        
    @torch.no_grad()    
    def sample(self,
               sample_size: int):
        latent_size = self.loadings.shape[1]
        x_dist = pydist.MultivariateNormal(loc = torch.zeros([1, latent_size]),
                                           covariance_matrix = self.cov_mat)
        x = x_dist.sample([sample_size]).squeeze(dim = -2)
        loc = F.linear(x, self.loadings, self.intercepts)
        
        y_dist = pydist.Normal(loc = loc, scale = self.residual_std)
        return y_dist.sample()
    
    
class LogNormalFactorModelSimulator(BaseFactorModelSimulator):
    
    def __init__(self,
                 loadings:     torch.Tensor,
                 intercepts:   torch.Tensor,
                 cov_mat:      torch.Tensor,
                 residual_std: torch.Tensor
                ):
        """Simulate from a lognormal factor model."""
        super().__init__(loadings = loadings, intercepts = intercepts, cov_mat = cov_mat)
        
        self.residual_std = residual_std
        
    @torch.no_grad()    
    def sample(self,
               sample_size: int):
        latent_size = self.loadings.shape[1]
        x_dist = pydist.MultivariateNormal(loc = torch.zeros([1, latent_size]),
                                           covariance_matrix = self.cov_mat)
        x = x_dist.sample([sample_size]).squeeze(dim = -2)
        loc = F.linear(x, self.loadings, self.intercepts)
        
        y_dist = pydist.LogNormal(loc = loc, scale = self.residual_std)
        return y_dist.sample()
    
    
SIMULATORS = {"poisson" : PoissonFactorModelSimulator,
              "negative_binomial" : NegativeBinomialFactorModelSimulator,
              "normal" : NormalFactorModelSimulator,
              "lognormal" : LogNormalFactorModelSimulator,
             }
    
    
def simulate_loadings(n_indicators: int,
                      latent_size:  int,
                     ):
    """Simulate a factor loadings matrix."""
    n_items = int(n_indicators * latent_size)
    mask = torch.block_diag(*[torch.ones([n_indicators, 1])] * latent_size)
    ldgs = pydist.Uniform(0.5, 1.7).sample([n_items, latent_size]).mul(mask)
    
    return ldgs


def simulate_categorical_intercepts(n_items: int,
                                    all_same_n_cats: bool = True, 
                                   ):
    """Simulate intercepts for a categorical response model."""
    if all_same_n_cats:
        n_cats = [3] * n_items
    else:
        cats = [2, 3, 4, 5, 6]
        assert(n_items >= len(cats))
        n_cats = cats * (n_items // len(cats)) + cats[:n_items % len(cats)]

    ints = []
    for n_cat in n_cats:
        if n_cat > 2:
            cuts = torch.linspace(-3, 3, n_cat)
            d = 3 / (n_cat - 1)
            tmp = (pydist.Uniform(-d, d).sample([1, n_cat - 1]) +
                   0.5 * (cuts[1:] + cuts[:-1])).flip(-1)
        else:
            tmp = pydist.Uniform(-1.5, 1.5).sample([1, 1]).flip(-1)
        ints.append(F.pad(tmp, (0, max(n_cats) - n_cat), value = float("nan")))

    return torch.cat(ints, dim = 0), n_cats


def simulate_covariance_matrix(latent_size: int,
                               cov_type:    int,
                               ):
    """Simulate a factor covariance matrix."""
    assert(cov_type in (0, 1, 2))
    if cov_type == 0:
        cov_mat = torch.eye(latent_size)
    if cov_type == 1:
        cov_mat = torch.ones([latent_size, latent_size]).mul(0.3)
        cov_mat.fill_diagonal_(1)
    if cov_type == 2:
        L = pydist.Uniform(-0.7, 0.7).sample([latent_size, latent_size]).tril()
        cov_mat = torch.mm(L, L.T)
        
    return cov_mat


def simulate_and_save_data(model_type:      str,
                           n_indicators:    int,
                           latent_size:     int,
                           cov_type:        int,
                           sample_size:     int,
                           expected_dir:    str,
                           data_dir:        str,
                           all_same_n_cats: bool = True,
                          ):
    """Simulate and save parameters and data for several types of latent factor models."""
    params = {}
    n_items = int(n_indicators * latent_size)
    params["cov_mat"] = simulate_covariance_matrix(latent_size, cov_type)
    
    if model_type in ("grm", "gpcm"):
        params["ldgs"] = simulate_loadings(n_indicators, latent_size)
        ints_out = simulate_categorical_intercepts(n_items, all_same_n_cats = all_same_n_cats)
        params["ints"], params["n_cats"] = ints_out[0], ints_out[1]
    else:
        if model_type != "normal":
            params["ldgs"] = simulate_loadings(n_indicators, latent_size).mul_(0.4)
            params["ints"] = pydist.Uniform(0.1, 0.5).sample([n_items])
            if model_type == "negative_binomial":
                params["probs"] = pydist.Uniform(0.5, 0.7).sample([n_items])
            elif model_type == "lognormal":
                params["residual_std"] = pydist.Uniform(1, 1.2).sample([n_items])
        else:
            params["ldgs"] = simulate_loadings(n_indicators, latent_size)
            params["ints"] = torch.randn(n_items).mul(0.1)
            params["residual_std"] = pydist.Uniform(0.6, 0.8).sample([n_items])
    for k, v in params.items():
        np.savetxt(os.path.join(expected_dir, k + ".csv"), np.asarray(v), delimiter = ",")
            
    if model_type in ("grm", "gpcm"):
        subprocess.call(["Rscript", "sim_mirt_data.R", model_type, str(sample_size), expected_dir, data_dir])
    else:
        sim_kwargs = {k : v for k, v in params.items() if k not in ("ldgs", "ints", "cov_mat")}
        Y = SIMULATORS[model_type](loadings = params["ldgs"], intercepts = params["ints"],
                                   cov_mat = params["cov_mat"], **sim_kwargs).sample(sample_size)
        np.savetxt(os.path.join(data_dir, "data.csv"), Y.numpy(), delimiter = ",")
        
        
def match_columns(inp_mat: torch.Tensor,
                  ref_mat: torch.Tensor,
                 ):
    """Permute cols. of input matrix to best match cols. of reference matrix."""
    assert(len(inp_mat.shape) == 2), "Input matrix must be 2D."
    assert(len(ref_mat.shape) == 2), "Reference matrix must be 2D."
    inp_mat = invert_factors(inp_mat.clone()).numpy()
    ref_mat = invert_factors(ref_mat.clone()).numpy()
    
    cost_mat = np.empty((ref_mat.shape[1], ref_mat.shape[1], ))
    cost_mat[:] = np.nan
    for ref_col in range(ref_mat.shape[1]): 
        for inp_col in range(inp_mat.shape[1]): 
            cost_mat[ref_col, inp_col] = np.sum((ref_mat[:, ref_col] - inp_mat[:, inp_col])**2)
    
    return torch.from_numpy(inp_mat[:, linear_sum_assignment(cost_mat)[1]])


def load_torch_from_csv(name, top_dir):
    t = np.loadtxt(os.path.join(top_dir, name), delimiter = ",")
    return torch.from_numpy(t)