import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import pyro.distributions as pydist
import pyro.distributions.transforms as T
from pyro.nn import DenseNN
from deepirtools.utils import get_thresholds
from typing import List, Optional
import itertools
import inspect
import logging


EPS = 1e-7


################################################################################
#
# Helper modules
#
################################################################################


class SparseLinear(nn.Module):
    
    def __init__(self,
                 in_features:  int,
                 out_features: int,
                 Q:            torch.Tensor,
                ):
        """Linear map with binary constraints."""
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.Q = Q
        
        self.free_weight = nn.Parameter(torch.empty(out_features, in_features))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.free_weight, mean=1., std=0.001)
        self.free_weight.data *= self.Q
        
    def forward(self,
                x: torch.Tensor,
               ):
        return F.linear(x, self.weight, None)
    
    @property
    def weight(self):
        return self.free_weight * self.Q
    
    
class LinearConstraints(nn.Module):
    
    def __init__(self,
                 in_features:  int,
                 out_features: int,
                 A:            torch.Tensor,
                 b:            Optional[torch.Tensor] = None,
                ):
        """Linear map with linear constraints."""
        super(LinearConstraints, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.free_weight = nn.Parameter(torch.empty(A.shape[0]))
        self.A = A
        self.b = b
        
        self.reset_parameters()
        
    def reset_parameters(self):
        return nn.init.normal_(self.free_weight, mean=1., std=0.001)
        
    def forward(self,
                x: torch.Tensor,
               ):
        return F.linear(x, self.weight, None)
    
    @property
    def weight(self):
        return F.linear(self.free_weight, self.A, self.b).view(self.in_features, self.out_features).T


class CategoricalBias(nn.Module):
    
    def __init__(self,
                 n_cats: List[int],
                 mask: Optional[torch.Tensor] = None,
                ):
        """Biases (i.e., intercepts) for categorical response models."""
        super(CategoricalBias, self).__init__()
        self.n_cats = n_cats
        M = max(n_cats)
        n_items = len(n_cats)
        
        if mask is None:
            mask = torch.ones([n_items, 1])
        bias_list = []
        for i, n_cat in enumerate(n_cats):
            thresholds = get_thresholds([mask[i].item() * -4, 4], n_cat)
            bias_list.append(F.pad(thresholds, (0, M - n_cat),
                                   value = float("inf"))) # Inf. saturates exponentials.
        self._bias = nn.Parameter(torch.stack(bias_list, dim = 0))
        self.register_buffer("mask", torch.cat([mask, torch.ones([n_items, M - 2])], dim = 1))
        
        nan_mask = torch.where(self._bias.isinf(), torch.ones_like(self._bias) * float("nan"),
                               torch.ones_like(self._bias))
        self.register_buffer("nan_mask", nan_mask)

    def forward(self,
                x: torch.Tensor,
               ):
        return (self._bias * self.mask) + x
    
    @property
    def bias(self):
        bias = (self._bias * self.nan_mask)
        if bias.shape[1] == 1:
            return bias.squeeze()
        return bias

    
################################################################################
#
# Measurement model modules
#
################################################################################
    

class GradedBaseModel(nn.Module):
    
    def __init__(self,
                 latent_size: int,
                 n_cats:      List[int],
                 Q:           Optional[torch.Tensor] = None,
                 A:           Optional[torch.Tensor] = None,
                 b:           Optional[torch.Tensor] = None,
                 ints_mask:   Optional[torch.Tensor] = None,
                ):
        """
        Base model for graded responses.
        
        Args:
            latent_size (int):         Number of latent variables.
            n_cats      (List of int): Number of categories for each item.
            Q           (Tensor):      Binary matrix indicating measurement structure.
            A           (Tensor):      Matrix imposing linear constraints on loadings.
            b           (Tensor):      Vector imposing linear constraints on loadings.
            ints_mask   (Tensor):      Vector constraining first intercepts to zero.
        """
        super(GradedBaseModel, self).__init__()
        
        assert(not (Q is not None and (A is not None or b is not None))), "Q and (A, b) may not be specified at the same time."
        if Q is not None:
            assert(((Q == 0) + (Q == 1)).all()), "Q must only contain ones and zeros."
            self._loadings = SparseLinear(latent_size, len(n_cats), Q)
        elif A is not None:
            self._loadings = LinearConstraints(latent_size, len(n_cats), A, b)
        else:
            self._loadings = nn.Linear(latent_size, len(n_cats), bias = False)
        self.Q = Q
        self.A = A
        
        if ints_mask is not None:
            assert(ints_mask.numel() == len(n_cats)), "ints_mask must be same size as number of items."
            ints_mask = ints_mask.clone().view(-1, 1)
            assert(((ints_mask == 0) + (ints_mask == 1)).all()), "ints_mask must only contain ones and zeros."
        self._intercepts = CategoricalBias(n_cats, ints_mask)
        self.n_cats = n_cats
        
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.Q is None and self.A is None:
            nn.init.xavier_uniform_(self._loadings.weight)

    def forward(self):
        """Compute log p(data | latents)."""
        raise NotImplementedError
        
    @property
    def loadings(self):
        return self._loadings.weight.data
    
    @property
    def intercepts(self):
        return self._intercepts.bias.data
    
    
class GradedResponseModel(GradedBaseModel):
    
    def __init__(self,
                 latent_size: int,
                 n_cats:      List[int],
                 Q:           Optional[torch.Tensor] = None,
                 A:           Optional[torch.Tensor] = None,
                 b:           Optional[torch.Tensor] = None,
                 ints_mask:   Optional[torch.Tensor] = None,
                ):
        """Samejima's graded response model."""
        super().__init__(latent_size = latent_size, n_cats = n_cats, Q = Q, A = A, b = b,
                         ints_mask = ints_mask)

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
               ):
        Bx = self._loadings(x)
        cum_probs = self._intercepts(Bx.unsqueeze(-1).expand(Bx.shape +
                                                             torch.Size([max(self.n_cats) - 1]))).sigmoid()
        upper_probs = F.pad(cum_probs, (0, 1), value = 1.)
        lower_probs = F.pad(cum_probs, (1, 0), value = 0.)
        probs = upper_probs - lower_probs
        
        idxs = y.long().expand(probs[..., -1].shape).unsqueeze(-1)
        log_py_x = -(torch.gather(probs, dim = -1, index = idxs).squeeze(-1)).clamp(min = EPS).log()
        if mask is not None:
            log_py_x = log_py_x.mul(mask)
        return log_py_x.sum(dim = -1, keepdim = True)
    
    
class GeneralizedPartialCreditModel(GradedBaseModel):
    
    def __init__(self,
                 latent_size: int,
                 n_cats:      List[int],
                 Q:           Optional[torch.Tensor] = None,
                 A:           Optional[torch.Tensor] = None,
                 b:           Optional[torch.Tensor] = None,
                 ints_mask:   Optional[torch.Tensor] = None,
                ):
        """Generalized partial credit model."""
        super().__init__(latent_size = latent_size, n_cats = n_cats, Q = Q, A = A, b = b,
                         ints_mask = ints_mask)

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
               ):
        M = max(self.n_cats)
        
        Bx = self._loadings(x)
        shape = Bx.shape + torch.Size([M])
        kBx = Bx.unsqueeze(-1).expand(shape) * torch.linspace(0, M - 1, M)
        
        cum_bias = self._intercepts._bias.cumsum(dim = 1)
        cum_bias = F.pad(cum_bias, (1, 0), value = 0.).expand(shape)
        tmp = kBx - cum_bias
        
        log_py_x = tmp - (tmp).logsumexp(dim = -1, keepdim = True)
        
        idxs = y.long().expand(log_py_x[..., -1].shape).unsqueeze(-1)
        log_py_x = -torch.gather(log_py_x, dim = -1, index = idxs).squeeze(-1)
        if mask is not None:
            log_py_x = log_py_x.mul(mask)
        return log_py_x.sum(dim = -1, keepdim = True)
    
    
class NonGradedBaseModel(nn.Module):
    
    def __init__(self,
                 latent_size: int,
                 n_items:     int,
                 Q:           Optional[torch.Tensor] = None,
                 A:           Optional[torch.Tensor] = None,
                 b:           Optional[torch.Tensor] = None,
                 ints_mask:   Optional[torch.Tensor] = None,
                ):
        """
        Base model for non-graded responses.
        
        Args:
            latent_size (int):         Number of latent variables.
            n_items     (int):         Number of items.
            Q           (Tensor):      Binary matrix indicating measurement structure.
            A           (Tensor):      Matrix imposing linear constraints on loadings.
            b           (Tensor):      Vector imposing linear constraints on loadings.
            ints_mask   (Tensor):      Vector constraining specific intercepts to zero.
        """
        super(NonGradedBaseModel, self).__init__()
        
        assert(not (Q is not None and (A is not None or b is not None))), "Q and (A, b) may not be specified at the same time."
        if Q is not None:
            assert(((Q == 0) + (Q == 1)).all()), "Q must only contain ones and zeros."
            self._loadings = SparseLinear(latent_size, n_items, Q)
        elif A is not None:
            self._loadings = LinearConstraints(latent_size, n_items, A, b)
        else:
            self._loadings = nn.Linear(latent_size, n_items, bias = False)
        self.Q = Q
        self.A = A
        
        if ints_mask is not None:
            assert(((ints_mask == 0) + (ints_mask == 1)).all()), "ints_mask must only contain ones and zeros."
        self._bias = nn.Parameter(torch.empty(n_items))
        self.ints_mask = ints_mask
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        if self.Q is None and self.A is None:
            nn.init.xavier_uniform_(self._loadings.weight)
        nn.init.normal_(self._bias, mean=0., std=0.001)
        
    def forward(self):
        """Compute log p(data | latents)."""
        raise NotImplementedError
        
    @property
    def bias(self):
        if self.ints_mask is not None:
            return self.ints_mask * self._bias
        else:
            return self._bias
        
    @property
    def loadings(self):
        return self._loadings.weight.data
        
    @property
    def intercepts(self):
        return self.bias.data
    
    
class PoissonFactorModel(NonGradedBaseModel):
    
    def __init__(self,
                 latent_size: int,
                 n_items:     int,
                 Q:           Optional[torch.Tensor] = None,
                 A:           Optional[torch.Tensor] = None,
                 b:           Optional[torch.Tensor] = None,
                 ints_mask:   Optional[torch.Tensor] = None,
                ):
        super().__init__(latent_size = latent_size, n_items = n_items, Q = Q, A = A, b = b,
                         ints_mask = ints_mask)
            
    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
               ):
        log_rate = self._loadings(x) + self.bias
        
        py_x = pydist.Poisson(rate = log_rate.exp().clamp(min = EPS, max = 100))
        return -py_x.log_prob(y).sum(-1, keepdim = True)
    
    
class NegativeBinomialFactorModel(NonGradedBaseModel):
    
    def __init__(self,
                 latent_size: int,
                 n_items:     int,
                 Q:           Optional[torch.Tensor] = None,
                 A:           Optional[torch.Tensor] = None,
                 b:           Optional[torch.Tensor] = None,
                 ints_mask:   Optional[torch.Tensor] = None,
                ):
        super().__init__(latent_size = latent_size, n_items = n_items, Q = Q, A = A, b = b,
                         ints_mask = ints_mask)
        
        self.logits = nn.Parameter(torch.empty(n_items))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.logits, mean=0., std=0.001)
            
    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
               ):
        log_total_count = self._loadings(x) + self.bias
        
        py_x = pydist.NegativeBinomial(total_count = log_total_count.exp().clamp(min = EPS, max = 100),
                                       logits = self.logits)
        return -py_x.log_prob(y).sum(-1, keepdim = True)
    
    
class NormalFactorModel(NonGradedBaseModel):
    
    def __init__(self,
                 latent_size: int,
                 n_items:     int,
                 Q:           Optional[torch.Tensor] = None,
                 A:           Optional[torch.Tensor] = None,
                 b:           Optional[torch.Tensor] = None,
                 ints_mask:   Optional[torch.Tensor] = None,
                ):
        super().__init__(latent_size = latent_size, n_items = n_items, Q = Q, A = A, b = b,
                         ints_mask = ints_mask)
        
        self.free_phi = nn.Parameter(torch.empty(n_items))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.free_phi, mean=math.log(math.exp(1) - 1), std=0.001)
            
    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
               ):
        loc = self._loadings(x) + self.bias
        
        py_x = pydist.Normal(loc = loc, scale = self.residual_std)
        return -py_x.log_prob(y).sum(-1, keepdim = True)
    
    @property
    def residual_std(self):
        return F.softplus(self.free_phi) + EPS

    
class LogNormalFactorModel(NonGradedBaseModel):
    
    def __init__(self,
                 latent_size: int,
                 n_items:     int,
                 Q:           Optional[torch.Tensor] = None,
                 A:           Optional[torch.Tensor] = None,
                 b:           Optional[torch.Tensor] = None,
                 ints_mask:   Optional[torch.Tensor] = None,
                ):
        super().__init__(latent_size = latent_size, n_items = n_items, Q = Q, A = A, b = b,
                         ints_mask = ints_mask)
        
        self.free_phi = nn.Parameter(torch.empty(n_items))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.free_phi, mean=math.log(math.exp(1) - 1), std=0.001)
            
    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
               ):
        loc = self._loadings(x) + self.bias
        
        py_x = pydist.LogNormal(loc = loc, scale = self.residual_std)
        return -py_x.log_prob(y).sum(-1, keepdim = True)
    
    @property
    def residual_std(self):
        return F.softplus(self.free_phi) + EPS
            
    
################################################################################
#
# Latent prior modules
#
################################################################################
    

class Spherical(nn.Module):
    
    def __init__(self,
                 size:               int,
                 fixed_variances:    bool,
                 correlated_factors: List[int],
                ):
        """
        Spherical parameterization of a covariance matrix.
        
        Args:
            size               (int):         Number of correlated variables.
            fixed_variances    (bool):        Whether to fix variances to one.
            correlated_factors (List of int): Which variables should be correlated.
        """
        super(Spherical, self).__init__()
        if correlated_factors != []:
            assert(max(correlated_factors) <= size - 1), ("correlated_factors may include no values ",
                                                          "larger than {}.".format(size - 1))
        self.size = size
        self.fixed_variances = fixed_variances
        self.correlated_factors = correlated_factors
        
        if self.correlated_factors != []:
            n_elts = int((size * (size + 1)) / 2)
            self.theta = nn.Parameter(torch.zeros([n_elts]))
            diag_idxs = torch.arange(1, size + 1).cumsum(dim = 0) - 1
            self.theta.data[diag_idxs] = math.log(math.pi / 2)
            
            tril_idxs = torch.tril_indices(row = size - 1, col = size - 1, offset = 0)
            uncorrelated_factors = [factor for factor in [i for i in range(size)] if factor not in correlated_factors]
            self.uncorrelated_tril_idxs = tril_idxs[:, sum((tril_idxs[1,:] == factor) + (tril_idxs[0,:] == factor - 1) for
                                                           factor in uncorrelated_factors) > 0]
            
    def cart2spher(self, cart_mat):
        n = cart_mat.size(1)
        spher_mat = torch.zeros_like(cart_mat)
        
        if n > 1:
            cos_mat = cart_mat[:, 1:n].cos()
            sin_mat = cart_mat[:, 1:n].sin().cumprod(1)
            spher_mat[:, 0] = cart_mat[:, 0] * cos_mat[:, 0]
            spher_mat[:, 1:(n - 1)] = cart_mat[:, 0].unsqueeze(1) * sin_mat[:, 0:(n - 2)] * cos_mat[:, 1:(n - 1)]
            spher_mat[:, -1] = cart_mat[:, 0] * sin_mat[:, -1]
            return spher_mat
        else:
            return cart_mat
        
    @property
    def weight(self):
        if self.correlated_factors != [] or not self.fixed_variances:
            tril_idxs = torch.tril_indices(row = self.size, col = self.size, offset = 0)
            theta_mat = torch.zeros(self.size, self.size, device=self.theta.device)
            theta_mat[tril_idxs[0], tril_idxs[1]] = self.theta

            # Ensure the parameterization is unique.
            exp_theta_mat = torch.zeros(self.size, self.size, device=self.theta.device)
            exp_theta_mat[tril_idxs[0], tril_idxs[1]] = self.theta.exp()
            lower_tri_l_mat = (math.pi * exp_theta_mat) / (1 + theta_mat.exp())
            l_mat = exp_theta_mat.diag().diag_embed() + lower_tri_l_mat.tril(diagonal = -1)

            if self.fixed_variances:
                l_mat[:, 0] = torch.ones(l_mat.size(0), device=self.theta.device)
            
            # Constrain specific correlations to zero.
            l_mat[1:, 1:].data[self.uncorrelated_tril_idxs[0], self.uncorrelated_tril_idxs[1]] = math.pi / 2
        
            return self.cart2spher(l_mat)
        else:
            return torch.eye(self.size, device=self.theta.device)

    def forward(self,
                x: torch.Tensor
               ):
        return F.linear(x, self.weight)
    
    @property
    def cov(self):
        if self.correlated_factors != [] or not self.fixed_variances:
            weight = self.weight
            return torch.matmul(weight, weight.t())
        else:
            return torch.eye(self.size, device=self.theta.device)
    
    @property
    def inv_cov(self):
        if self.correlated_factors != [] or not self.fixed_variances:
            return torch.cholesky_solve(torch.eye(self.size, device=self.theta.device),
                                        self.weight)
        else:
            return torch.eye(self.size, device=self.theta.device)
        
        
def spline_coupling(input_dim, split_dim=None, hidden_dims=None, count_bins=4, bound=3.):
    """Modification of Pyro's spline_coupling() to use ELU activations."""
    if split_dim is None:
        split_dim = input_dim // 2

    if hidden_dims is None:
        hidden_dims = [input_dim * 10, input_dim * 10]

    net = DenseNN(
        split_dim,
        hidden_dims,
        param_dims=[
            (input_dim - split_dim) * count_bins,
            (input_dim - split_dim) * count_bins,
            (input_dim - split_dim) * (count_bins - 1),
            (input_dim - split_dim) * count_bins,
        ],
        nonlinearity=nn.ELU(),
    )

    return T.SplineCoupling(input_dim, split_dim, net, count_bins, bound)

        
################################################################################
#
# Autoencoder modules
#
################################################################################
        
    
class VariationalAutoencoder(nn.Module):
    
    def __init__(self,
                 decoder,
                 input_size:            int,
                 latent_size:           int,
                 inference_net_sizes:   List[int] = [100],
                 fixed_variances:       bool = True,
                 correlated_factors:    List[int] = [],
                 covariate_size:        int = 0,
                 use_spline_prior:      bool = False,
                 **kwargs,
                ):
        """
        Variational autoencoder with an interchangeable measurement model (i.e., decoder).
        
        Args:
            decoder             (nn.Module):   Measurement model whose forward() method returns log p(data | latents).
            input_size          (int):         Neural network input dimension.
            inference_net_sizes (List of int): Neural network hidden layer dimensions.
                                                   E.g., a neural network with two hidden layers of size 100
                                                   has inference_net_sizes = [100, 100]
            latent_size         (int):         Number of latent variables.
            fixed_variances     (bool):        Whether to constrain latent variances to one.
            correlated_factors  (List of int): Which latent variables should be correlated.
            covariate_size      (int):         Number of covariates for latent regression.
            use_spline_prior    (bool):        Whether to use spline/spline coupling prior.
            kwargs              (dict):        Named parameters passed to decoder.__init__() and
                                               rational linear spline parameters passed to spline_coupling().
        """
        super(VariationalAutoencoder, self).__init__()
        
        # Inference model neural network.
        self.inf_net = DenseNN(input_size + covariate_size, inference_net_sizes,
                               [int(2 * latent_size)], nonlinearity = nn.ELU())
                
        # Measurement model.
        decoder_args = list(inspect.signature(decoder).parameters)
        decoder_kwargs = {k: kwargs.pop(k) for k in dict(kwargs) if k in decoder_args}
        self.decoder = decoder(latent_size=latent_size, **decoder_kwargs)
        
        # Latent regression.
        if covariate_size > 0:
            self.lreg_weight = nn.Parameter(torch.empty([latent_size, covariate_size]))
        self.cov_size = covariate_size
        
        # Latent prior.
        assert(not (covariate_size > 0 and use_spline_prior)), ("Latent regression not supported with ",
                                                                "spline/spline coupling prior.")
        assert(not (correlated_factors != [] and use_spline_prior)), ("Cannot constrain factor correlations ",
                                                                      "with spline/spline coupling prior.")
        spline_kwargs = {k: kwargs.pop(k) for k in dict(kwargs) if k in ["count_bins", "bound"]}
        assert(len(kwargs) == 0), "Unused arguments: " + ", ".join([k for k in kwargs])
        if use_spline_prior:
            if latent_size == 1:
                self.flow = T.Spline(1, **spline_kwargs)
            else:
                self.flow1 = T.spline_coupling(latent_size, **spline_kwargs)
                self.flow2 = T.Permute(torch.Tensor(list(reversed(range(latent_size)))).long())
                self.flow3 = T.spline_coupling(latent_size, **spline_kwargs)
        else:
            self.cholesky = Spherical(latent_size, fixed_variances, correlated_factors)
        self.latent_size = latent_size
        self.fixed_variances = fixed_variances
        self.use_spline_prior = use_spline_prior
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.inf_net.layers[-1].weight, mean=0., std=0.001)
        nn.init.normal_(self.inf_net.layers[-1].bias[0:self.latent_size], mean=0., std=0.001)
        nn.init.normal_(self.inf_net.layers[-1].bias[self.latent_size:], mean=math.log(math.exp(1) - 1), std=0.001)
        
        if self.cov_size > 0:
            nn.init.normal_(self.lreg_weight, mean=0., std=0.001)
        
        if self.use_spline_prior:
            device = self.inf_net.layers[0].weight.device
            params = [p.parameters() for p in self.__get_flow() if hasattr(p, "parameters")]
            optimizer = Adam([{"params" : itertools.chain(*params)}], lr = 1e-3, amsgrad = True)
            base_dist = pydist.Normal(torch.zeros([1, self.latent_size], device = device),
                                      torch.ones([1, self.latent_size], device = device))
            px = pydist.TransformedDistribution(base_dist, self.__get_flow())
            for _ in range(1000):
                self.zero_grad()
                x = torch.randn([128, self.latent_size], device = device)
                loss = -px.log_prob(x).mean()
                loss.backward()
                optimizer.step()
                px.clear_cache()
        
    def __get_flow(self):
        if self.use_spline_prior:
            if self.latent_size == 1:
                return [self.flow]
            else:
                return [self.flow1, self.flow2, self.flow3]
        return None

    def encode(self,
               y:          torch.Tensor,
               mc_samples: int,
               iw_samples: int,
              ):
        """Sample approximate latent variable posterior."""
        hidden = self.inf_net(y)

        # Monte Carlo samples.
        hidden = hidden.unsqueeze(0).expand(torch.Size([mc_samples]) + hidden.shape)
        
        # Importance-weighted samples.
        hidden = hidden.unsqueeze(0).expand(torch.Size([iw_samples]) + hidden.shape)
        
        mu, std = hidden.chunk(chunks = 2, dim = -1)
        std = F.softplus(std)
            
        return mu, std + EPS

    def forward(self,
                y:              torch.Tensor,
                grad_estimator: str,
                mask:           Optional[torch.Tensor] = None,
                covariates:     Optional[torch.Tensor] = None,
                mc_samples:     int = 1,
                iw_samples:     int = 1,
               ):
        """
        Compute evidence lower bound (ELBO).
        
        Args:
            y              (Tensor): Observed data.
            grad_estimator (str):    Gradient estimator for inference model parameters:
                                         "dreg" = doubly reparameterized gradient estimator
                                         "iwae" = standard gradient estimator
            mask           (Tensor): Binary mask indicating missing item responses.
            covariates     (Tensor): Matrix of covariates.
            mc_samples     (int):    Number of Monte Carlo samples.
            iw_samples     (int):    Number of importance-weighted samples.
        """
        if self.cov_size > 0:
            try:
                _y = torch.cat((y, covariates), dim = 1)
            except TypeError:
                if covariates is None:
                    logging.exception("Covariates must be passed to fit() when covariate_size > 0.")
        else:
            _y = y
        mu, std = self.encode(_y, mc_samples, iw_samples)
        x = mu + std * torch.randn_like(mu)
        
        # Log p(y | x, covariates).
        log_py_x = self.decoder(x, y, mask)
        
        # Log p(x | covariates).
        if self.use_spline_prior:
            base_dist = pydist.Normal(torch.zeros_like(x, device = x.device), torch.ones_like(x, device = x.device))
            flow = self.__get_flow()
            if self.fixed_variances:
                x_mean = x.mean(dim = -2, keepdim = True)
                x_dispersion = x.var(dim = -2, keepdim = True).add(EPS).sqrt().pow(-1)
                flow.append(T.AffineTransform(loc = -x_mean * x_dispersion, scale = x_dispersion))
            px = pydist.TransformedDistribution(base_dist, flow)
            log_px = px.log_prob(x).unsqueeze(-1)
        else:
            if self.cov_size > 0:
                loc = F.linear(covariates, self.lreg_weight)
            else:
                loc = torch.zeros_like(x, device = x.device)
            if self.cholesky.correlated_factors != []:
                log_px = pydist.MultivariateNormal(loc, scale_tril = self.cholesky.weight).log_prob(x).unsqueeze(-1)
            else:
                log_px = pydist.Normal(loc, torch.ones_like(x, device = x.device)).log_prob(x).sum(dim = -1, keepdim = True)
            
        # Log q(x | y, covariates).
        if iw_samples > 1 and grad_estimator == "dreg":
            qx_y = pydist.Normal(mu.detach(), std.detach())
        else:
            qx_y = pydist.Normal(mu, std)
        log_qx_y = qx_y.log_prob(x).sum(dim = -1, keepdim = True)
        
        elbo = log_py_x + log_qx_y - log_px
        return elbo, x