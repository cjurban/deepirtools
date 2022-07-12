import torch
from torch import nn
import torch.nn.functional as F
import pyro.distributions as pydist
import deepirtools


class BaseFactorSimulator():
    
    def __init__(self,
                 loadings:   torch.Tensor,
                 intercepts: torch.Tensor,
                 cov_mat:    torch.Tensor,
                 ):
        super(BaseFactorSimulator, self).__init__()

        self.loadings = loadings
        self.intercepts = intercepts
        self.cov_mat = cov_mat
        
    def sample(self):
        raise NotImplementedError

        
class PoissonFactorModelSimulator(BaseFactorSimulator):
    
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
        x = x_dist.sample([sample_size]).squeeze()
        rate = F.linear(x, self.loadings, self.intercepts).exp()
        
        y_dist = pydist.Poisson(rate = rate)
        return y_dist.sample()
    
    
class NegativeBinomialModelSimulator(BaseFactorSimulator):
    
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
        x = x_dist.sample([sample_size]).squeeze()
        total_count = F.linear(x, self.loadings, self.intercepts).exp()
        
        y_dist = pydist.NegativeBinomial(total_count = total_count, probs = self.probs)
        return y_dist.sample()
    
    
class NormalFactorModelSimulator(BaseFactorSimulator):
    
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
        x = x_dist.sample([sample_size]).squeeze()
        loc = F.linear(x, self.loadings, self.intercepts)
        
        y_dist = pydist.Normal(loc = loc, scale = self.residual_std)
        return y_dist.sample()
    
    
class LogNormalFactorModelSimulator(BaseFactorSimulator):
    
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
        x = x_dist.sample([sample_size]).squeeze()
        loc = F.linear(x, self.loadings, self.intercepts)
        
        y_dist = pydist.LogNormal(loc = loc, scale = self.residual_std)
        return y_dist.sample()
    
    
class BaseParamSimulator():
    
    def __init__(self,):
        super(BaseParamSimulator, self).__init__()

        self._param_dict = None
        
    def __sample(self):
        raise NotImplementedError
        
    @property
    def param_dict(self):
        if self._param_dict is None:
            self.__sample()
        return self._param_dict
    
    
class LoadingsSimulator(BaseParamSimulator):
    
    conds = ["cond1", "cond2", "cond3", "cond4"]
    
    def __init__(self,
                 n_indicators:  int,
                 latent_size:   int,
                 seed:          int,
                ):
        super().__init__()
        
        self.n_indicators = n_indicators
        self.latent_size = latent_size
        self.seed = seed
        
    @torch.no_grad()
    def __sample(self):
        n_items = int(n_indicators * latent_size)
        size = torch.Size([n_items, latent_size])
        param_dict = {}
        
        deepirtools.manual_seed(self.seed)
        
        for cond in self.conds:
            if cond == "cond1":
                mask = torch.block_diag(*[torch.ones([n_indicators, 1])] * latent_size)
            elif cond == "cond2":
                mask = torch.block_diag(*[-torch.ones([n_indicators, 1])] * latent_size)
            elif cond == "cond3":
                mask = torch.block_diag(*[torch.ones([n_indicators, 1])] * latent_size)
                mask_mul = torch.bernoulli(torch.ones(size).mul(0.8))
                mask = mask * torch.where(mask_mul == 0, -torch.ones(size), mask_mul)
            elif cond == "cond4":
                mask = torch.block_diag(*[torch.ones([n_indicators, 1])] * latent_size)
                mask[mask == 0] = 0.1
                
            ldgs_dist = pydist.LogNormal(loc = torch.zeros(size),
                                         scale = torch.ones(size).mul(0.5))
            param_dict[cond] = ldgs_dist.sample() * mask
            
        self._param_dict = param_dict
        
        
class GradedInterceptsSimulator(BaseParamSimulator):
    
    conds = ["cond1", "cond2", "cond3"]
    
    def __init__(self,
                 n_items: int,
                 seed:    int,
                ):
        super().__init__()
        
        self.n_items = n_items
        self.seed = seed
        
    @torch.no_grad()
    def __sample(self):
        param_dict = {}
        
        deepirtools.manual_seed(self.seed)
        
        for cond in self.conds:
            if cond == "cond1":
            elif cond == "cond2":
            elif cond == "cond3":
            
        self._param_dict = param_dict 
    
    
class NonGradedInterceptsSimulator(BaseParamSimulator):
    
    conds = ["cond1"]
    
    def __init__(self,
                 n_items: int,
                 seed:    int,
                ):
        super().__init__()
        
        self.n_items = n_items
        self.seed = seed
        
    @torch.no_grad()
    def __sample(self):
        param_dict = {}
        
        deepirtools.manual_seed(self.seed)
        
        for cond in self.conds:
            if cond == "cond1":
                param_dict[cond] = torch.randn(self.n_items)
            
        self._param_dict = param_dict  