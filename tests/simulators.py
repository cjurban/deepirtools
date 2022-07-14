import torch
import torch.nn.functional as F
import pyro.distributions as pydist


class BaseFactorModelSimulator():
    
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
        x = x_dist.sample([sample_size]).squeeze()
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
        x = x_dist.sample([sample_size]).squeeze()
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
        x = x_dist.sample([sample_size]).squeeze()
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
        x = x_dist.sample([sample_size]).squeeze()
        loc = F.linear(x, self.loadings, self.intercepts)
        
        y_dist = pydist.LogNormal(loc = loc, scale = self.residual_std)
        return y_dist.sample()
    
    
class BaseParamSimulator():
    
    def __init__(self,):
        super(BaseParamSimulator, self).__init__()
        
    def sample(self):
        raise NotImplementedError
    
    
class LoadingsSimulator(BaseParamSimulator):
    
    def __init__(self,
                 n_indicators:  int,
                 latent_size:   int,
                ):
        super().__init__()
        
        self.n_indicators = n_indicators
        self.latent_size = latent_size
        
    @torch.no_grad()
    def sample(self):
        n_items = int(n_indicators * latent_size)
        size = torch.Size([n_items, latent_size])
        
        ldgs_list = []
        for i in range(4):
            if i == 0:
                mask = torch.block_diag(*[torch.ones([n_indicators, 1])] * latent_size)
            elif i == 1:
                mask = torch.block_diag(*[-torch.ones([n_indicators, 1])] * latent_size)
            elif i == 2:
                mask = torch.block_diag(*[torch.ones([n_indicators, 1])] * latent_size)
                mask_mul = torch.bernoulli(torch.ones(size).mul(0.8))
                mask = mask * torch.where(mask_mul == 0, -torch.ones(size), mask_mul)
            elif i == 3:
                mask = torch.block_diag(*[torch.ones([n_indicators, 1])] * latent_size)
                mask[mask == 0] = 0.1
                
            ldgs_dist = pydist.LogNormal(loc = torch.zeros(size),
                                         scale = torch.ones(size).mul(0.5))
            ldgs_list[ldgs_dist.sample() * mask]
            
        return ldgs_list
        
        
class GradedInterceptsSimulator(BaseParamSimulator):
    
    def __init__(self,
                 n_items: int,
                ):
        super().__init__()
        
        self.n_items = n_items
        
    @torch.no_grad()
    def sample(self):
        ints_list = []
        for i in range(3):
            if i == 0:
                n_cats = [2] * self.n_items
            elif i == 1:
                n_cats = [3] * self.n_items
            elif i == 2:
                cats = [2, 3, 4, 5, 6]
                assert(self.n_items >= len(cats))
                n_cats = cats * (self.n_items // len(cats)) + cats[:self.n_items % len(cats)]
            
            ints = []
            for n_cat in n_cats:
                if n_cat > 2:
                    cuts = torch.linspace(-4, 4, n_cat)
                    d = 4 / (n_cat - 1)
                    ints.append(pydist.Uniform(-d, d).sample([n_cat - 1]) +
                                0.5 * (cuts[1:] + cuts[:-1]))
                else:
                    ints.append(pydist.Uniform(-1.5, 1.5).sample([1]))
            ints_list.append((torch.cat(ints, dim = 0), n_cats))
                
        return ints_list
        
    
class NonGradedInterceptsSimulator(BaseParamSimulator):
    
    def __init__(self,
                 n_items: int,
                ):
        super().__init__()
        
        self.n_items = n_items
        
    @torch.no_grad()
    def sample(self):
        return [torch.randn(self.n_items)]
    
    
class CovarianceMatrixSimulator(BaseParamSimulator):
    
    def __init__(self,
                 latent_size: int,
                ):
        super().__init__()
        
        self.latent_size = latent_size
        
    @torch.no_grad()
    def sample(self):
        cov_mat_list = []
        for i in range(3):
            if i == 0:
                cov_mat = torch.eye(self.latent_size)
            if i == 1:
                cov_mat = torch.ones([self.latent_size, self.latent_size]).mul(0.3)
                cov_mat.fill_diagonal_(1)
            if i == 2:
                L = torch.randn([latent_size, latent_size]).tril()
                cov_mat = torch.mm(L, L.T)
            cov_mat_list.append(cov_mat)
            
        return cov_mat_list