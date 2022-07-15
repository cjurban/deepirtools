import torch
import torch.nn.functional as F
import pyro.distributions as pydist


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
    
    
def simulate_loadings(n_indicators: int,
                      latent_size:  int,
                      shrink:       bool = False,
                     ):
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
    for n_cat in n_cats:
        if n_cat > 2:
            cuts = torch.linspace(-4, 4, n_cat)
            d = 4 / (n_cat - 1)
            ints.append(pydist.Uniform(-d, d).sample([n_cat - 1]) +
                        0.5 * (cuts[1:] + cuts[:-1]))
        else:
            ints.append(pydist.Uniform(-1.5, 1.5).sample([1]))

    return torch.cat(ints, dim = 0), n_cats


def simulate_non_graded_intercepts(n_items: int,
                                   all_positive: bool = False,
                                  ):
    if all_positive:
        return pydist.Uniform(0.1, 0.5).sample([n_items])
    
    return torch.randn(n_items).mul(0.1)


def simulate_and_save_data():
    if model_type in ("grm", "gpcm"):
        ldgs = simulate_loadings(n_indicators, latent_size).to(device)
        ints, n_cats = simulate_graded_intercepts(n_items).to(device)
    else:
        if model_type != "normal":
            ldgs = simulate_loadings(n_indicators, latent_size, shrink = True).to(device)
            ints = simulate_non_graded_intercepts(n_items, all_positive = True).to(device)
            if model_type == "negative_binomial":
                sim_kwargs = {"probs" : pydist.Uniform(0.5, 0.7).sample([n_items]).to(device)}
            elif model_type == "lognormal":
                sim_kwargs = {"residual_std" : pydist.Uniform(1, 1.2).sample([n_items]).to(device)}
        else:
            ldgs = simulate_loadings(n_indicators, latent_size).to(device)
            ints = simulate_non_graded_intercepts(n_items).to(device)
            sim_kwargs = {"residual_std" : pydist.Uniform(0.6, 0.8).sample([n_items]).to(device)}
        Y = simulators[model_type](loadings = ldgs, intercepts = ints,
                                   cov_mat = cov_mat, **sim_kwargs).sample(sample_size)
#
#    
#class CovarianceMatrixSimulator(BaseParamSimulator):
#    
#    def __init__(self,
#                 latent_size: int,
#                ):
#        super().__init__()
#        
#        self.latent_size = latent_size
#        
#    @torch.no_grad()
#    def sample(self):
#        cov_mat_list = []
#        for i in range(3):
#            if i == 0:
#                cov_mat = torch.eye(self.latent_size)
#            if i == 1:
#                cov_mat = torch.ones([self.latent_size, self.latent_size]).mul(0.3)
#                cov_mat.fill_diagonal_(1)
#            if i == 2:
#                L = torch.randn([latent_size, latent_size]).tril()
#                cov_mat = torch.mm(L, L.T)
#            cov_mat_list.append(cov_mat)
#            
#        return cov_mat_list