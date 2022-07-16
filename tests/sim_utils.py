import os
from os.path import join
import torch
import torch.nn.functional as F
import pyro.distributions as pydist
import numpy as np
import subprocess


class BaseFactorModelSimulator():
    
    def __init__(self,
                 loadings:   torch.Tensor,
                 intercepts: torch.Tensor,
                 cov_mat:    torch.Tensor,
                ):
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
                      shrink:       bool = False,
                     ):
    n_items = int(n_indicators * latent_size)
    mask = torch.block_diag(*[torch.ones([n_indicators, 1])] * latent_size)
    ldgs_dist = pydist.LogNormal(loc = torch.zeros([n_items, latent_size]),
                                 scale = torch.ones([n_items, latent_size]).mul(0.5))
    ldgs = ldgs_dist.sample() * mask
    if shrink:
        ldgs.mul_(0.3).clamp_(max = 0.7)

    return ldgs


def simulate_graded_intercepts(n_items: int,
                               all_same_n_cats: bool = True, 
                              ):
    if all_same_n_cats:
        n_cats = [2] * n_items
    else:
        cats = [2, 3, 4, 5, 6]
        assert(n_items >= len(cats))
        n_cats = cats * (n_items // len(cats)) + cats[:n_items % len(cats)]

    ints = []
    padded_ints = []
    for n_cat in n_cats:
        if n_cat > 2:
            cuts = torch.linspace(-4, 4, n_cat)
            d = 4 / (n_cat - 1)
            tmp = (pydist.Uniform(-d, d).sample([1, n_cat - 1]) +
                   0.5 * (cuts[1:] + cuts[:-1]))
        else:
            tmp = pydist.Uniform(-1.5, 1.5).sample([1, 1])
        ints.append(tmp)
        padded_tmp = F.pad(tmp.flip(-1), (0, max(n_cats) - n_cat), value = float("nan"))
        padded_ints.append(padded_tmp)

    return torch.cat(ints, dim = 0), torch.cat(padded_ints, dim = 0), n_cats


def simulate_non_graded_intercepts(n_items: int,
                                   all_positive: bool = False,
                                  ):
    if all_positive:
        return pydist.Uniform(0.1, 0.5).sample([n_items])
    
    return torch.randn(n_items).mul(0.1)


def simulate_covariance_matrix(latent_size: int,
                               cov_type:    int,
                               ):
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
    n_items = int(n_indicators * latent_size)
    cov_mat = simulate_covariance_matrix(latent_size, cov_type)
    
    if model_type in ("grm", "gpcm"):
        ldgs = simulate_loadings(n_indicators, latent_size)
        ints, ints_reshape, n_cats = simulate_graded_intercepts(n_items, all_same_n_cats = all_same_n_cats)
        iwave_kwargs = {"n_cats" : n_cats}
        
        np.savetxt(os.path.join(expected_dir, "ldgs.csv"), ldgs.numpy(), delimiter = ",")
        np.savetxt(os.path.join(expected_dir, "ints.csv"), ints.numpy(), delimiter = ",")
        ints_reshape = ints_reshape.numpy().astype(str)
        ints_reshape[ints_reshape == "nan"] = "NA"
        np.savetxt(os.path.join(expected_dir, "ints_reshape.csv"), ints_reshape, delimiter = ",", fmt = "%s")
        np.savetxt(os.path.join(expected_dir, "cov_mat.csv"), cov_mat.numpy(), delimiter = ",")
        
        subprocess.call(["Rscript", "sim_mirt_data.R", model_type, str(sample_size), expected_dir, data_dir])
        Y = np.loadtxt(os.path.join(data_dir, "data.csv"), delimiter = ",")
        Y = torch.from_numpy(Y)
    else:
        iwave_kwargs = {"n_items" : n_items}
        
        if model_type != "normal":
            ldgs = simulate_loadings(n_indicators, latent_size, shrink = True)
            ints = simulate_non_graded_intercepts(n_items, all_positive = True)
            if model_type == "negative_binomial":
                sim_kwargs = {"probs" : pydist.Uniform(0.5, 0.7).sample([n_items])}
            elif model_type == "lognormal":
                sim_kwargs = {"residual_std" : pydist.Uniform(1, 1.2).sample([n_items])}
        else:
            ldgs = simulate_loadings(n_indicators, latent_size)
            ints = simulate_non_graded_intercepts(n_items)
            sim_kwargs = {"residual_std" : pydist.Uniform(0.6, 0.8).sample([n_items])}
        Y = SIMULATORS[model_type](loadings = ldgs, intercepts = ints,
                                   cov_mat = cov_mat, **sim_kwargs).sample(sample_size)
        
        np.savetxt(os.path.join(expected_dir, "ldgs.csv"), ldgs.numpy(), delimiter = ",")
        np.savetxt(os.path.join(expected_dir, "ints.csv"), ints.numpy(), delimiter = ",")
        np.savetxt(os.path.join(expected_dir, "cov_mat.csv"), cov_mat.numpy(), delimiter = ",")
        for k, v in sim_kwargs.items():
            np.savetxt(os.path.join(expected_dir, k + ".csv"), v.numpy(), delimiter = ",")
        np.savetxt(os.path.join(data_dir, "data.csv"), Y.numpy(), delimiter = ",")
        
    return Y, iwave_kwargs