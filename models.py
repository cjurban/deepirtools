import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist
from utils import *
from typing import List, Optional
from itertools import chain

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


class CatBiasReshape(nn.Module):
    
    def __init__(self,
                 n_cats: List[int],
                 mask: Optional[torch.Tensor] = None,
                ):
        """Biases (i.e., intercepts) for graded response models. Reshaping makes computatiton faster."""
        super(CatBiasReshape, self).__init__()
        self.n_cats = n_cats
        
        # Biases.
        bias = torch.empty(sum([n_cat - 1 for n_cat in n_cats]))
        if mask is None:
            mask = torch.ones_like(bias)
        idxs = np.cumsum([n_cat - 1 for n_cat in ([1] + n_cats)])
        bias.data = torch.from_numpy(np.hstack([get_thresholds([mask[idx].item() * -4, 4], n_cat) for
                                                idx, n_cat in zip(idxs[:-1], n_cats)]))
        
        # Appending large values below numerically saturates exponentials.
        bias_reshape, sliced_bias = self._reshape(bias, idxs, 9999.)
        self.bias_reshape = nn.Parameter(bias_reshape)
        if mask is not None:
            self.mask = self._reshape(mask, idxs, 1., False)
        
        # Drop indices.
        nan_mask = torch.cat([F.pad(_slice, (0, max(self.n_cats) - _slice.size(0) - 1),
                                    value=float("nan")).unsqueeze(0) for
                                    _slice in sliced_bias], axis = 0)
        cum_probs_mask = nan_mask * 0. + 1.
        self.drop_idxs = ~cum_probs_mask.view(-1).isnan().to(mask.device)
        
    def _reshape(self, t, idxs, pad_val, return_slices=True):
        sliced_t = [t[idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]
        t_reshape = torch.cat([F.pad(_slice,
                                     (0, max(self.n_cats) - _slice.size(0) - 1),
                                      value=pad_val).unsqueeze(0) for
                                      _slice in sliced_t], axis = 0)
        if return_slices:
            return t_reshape, sliced_t
        return t_reshape

    def forward(self,
                x: torch.Tensor,
               ):
        return (self.bias_reshape * self.mask) + x
    
    @property
    def bias(self):
        return (self.bias_reshape.view(-1) * self.mask.view(-1))[self.drop_idxs]

    
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
            ints_mask   (Tensor):      Vector constraining specific intercepts to zero.
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
        
        self._intercepts = CatBiasReshape(n_cats, ints_mask)
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
        
        cum_bias = self._intercepts.bias_reshape.cumsum(dim = 1)
        cum_bias = F.pad(cum_bias, (1, 0), value = 0.).expand(shape)
        tmp = kBx - cum_bias
        
        log_py_x = tmp - (tmp).logsumexp(dim = -1, keepdim = True)
        
        idxs = y.long().expand(log_py_x[..., -1].shape).unsqueeze(-1)
        log_py_x = -torch.gather(log_py_x, dim = -1, index = idxs).squeeze(-1)
        if mask is not None:
            log_py_x = log_py_x.mul(mask)
        return log_py_x.sum(dim = -1, keepdim = True)
    
    
class ContinuousBaseModel(nn.Module):
    
    def __init__(self,
                 latent_size: int,
                 n_items:     int,
                 Q:           Optional[torch.Tensor] = None,
                 A:           Optional[torch.Tensor] = None,
                 b:           Optional[torch.Tensor] = None,
                 ints_mask:   Optional[torch.Tensor] = None,
                ):
        """
        Base model for continuous responses.
        
        Args:
            latent_size (int):         Number of latent variables.
            n_items     (int):         Number of items.
            Q           (Tensor):      Binary matrix indicating measurement structure.
            A           (Tensor):      Matrix imposing linear constraints on loadings.
            b           (Tensor):      Vector imposing linear constraints on loadings.
            ints_mask   (Tensor):      Vector constraining specific intercepts to zero.
        """
        super(ContinuousBaseModel, self).__init__()
        
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
        
        self._bias = nn.Parameter(torch.empty(n_items))
        self.ints_mask = ints_mask
        
        self.free_phi = nn.Parameter(torch.empty(n_items))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.Q is None and self.A is None:
            nn.init.xavier_uniform_(self.loadings.weight)
        nn.init.normal_(self._bias, mean=0., std=0.001)
        nn.init.normal_(self.free_phi, mean=math.log(math.exp(1) - 1), std=0.001)
        
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
    def residual_std(self):
        return F.softplus(self.free_phi) + EPS
        
    @property
    def loadings(self):
        return self._loadings.weight.data
        
    @property
    def intercepts(self):
        return self.bias.data
    
    
class NormalFactorModel(ContinuousBaseModel):
    
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
        loc = self._loadings(x) + self.bias
        
        py_x = dist.Normal(loc = loc, scale = self.residual_std)
        return -py_x.log_prob(y).sum(-1, keepdim = True)

    
class LogNormalFactorModel(ContinuousBaseModel):
    
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
        loc = self._loadings(x) + self.bias
        
        py_x = dist.LogNormal(loc = loc, scale = self.residual_std)
        return -py_x.log_prob(y).sum(-1, keepdim = True)
            
    
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
                 device:             str,
                ):
        """
        Spherical parameterization of a covariance matrix.
        
        Args:
            size               (int):         Number of correlated variables.
            fixed_variances    (bool):        Whether to fix variances to one.
            correlated_factors (List of int): Which variables should be correlated.
            device             (str):         Computing device used for fitting.
        """
        super(Spherical, self).__init__()
        self.size = size
        self.fixed_variances = fixed_variances
        self.correlated_factors = correlated_factors
        self.device = device
        
        if self.correlated_factors != []:
            n_elts = int((self.size * (self.size + 1)) / 2)
            self.theta = nn.Parameter(torch.zeros([n_elts]))
            diag_idxs = torch.from_numpy(np.cumsum(np.arange(1, self.size + 1)) - 1)
            self.theta.data[diag_idxs] = np.log(np.pi / 2)
            
            tril_idxs = torch.tril_indices(row = self.size - 1, col = self.size - 1, offset = 0)
            uncorrelated_factors = [factor for factor in np.arange(self.size).tolist() if factor not in self.correlated_factors]
            self.uncorrelated_tril_idxs = tril_idxs[:, sum((tril_idxs[1,:] == factor) + (tril_idxs[0,:] == factor - 1) for
                                                           factor in uncorrelated_factors) > 0]
            
    def _cart2spher(self, cart_mat):
        n = cart_mat.size(1)
        spher_mat = torch.zeros_like(cart_mat)
        cos_mat = cart_mat[:, 1:n].cos()
        sin_mat = cart_mat[:, 1:n].sin().cumprod(1)

        spher_mat[:, 0] = cart_mat[:, 0] * cos_mat[:, 0]
        spher_mat[:, 1:(n - 1)] = cart_mat[:, 0].unsqueeze(1) * sin_mat[:, 0:(n - 2)] * cos_mat[:, 1:(n - 1)]
        spher_mat[:, -1] = cart_mat[:, 0] * sin_mat[:, -1]

        return spher_mat
        
    @property
    def weight(self):
        if self.correlated_factors != [] or not self.fixed_variances:
            tril_idxs = torch.tril_indices(row = self.size, col = self.size, offset = 0)
            theta_mat = torch.zeros(self.size, self.size, device=self.device)
            theta_mat[tril_idxs[0], tril_idxs[1]] = self.theta

            # Ensure the parameterization is unique.
            exp_theta_mat = torch.zeros(self.size, self.size, device=self.device)
            exp_theta_mat[tril_idxs[0], tril_idxs[1]] = self.theta.exp()
            lower_tri_l_mat = (np.pi * exp_theta_mat) / (1 + theta_mat.exp())
            l_mat = exp_theta_mat.diag().diag_embed() + lower_tri_l_mat.tril(diagonal = -1)

            if self.fixed_variances:
                l_mat[:, 0] = torch.ones(l_mat.size(0), device=self.device)
            
            # Constrain specific correlations to zero.
            l_mat[1:, 1:].data[self.uncorrelated_tril_idxs[0], self.uncorrelated_tril_idxs[1]] = np.pi / 2
        
            return self._cart2spher(l_mat)
        else:
            return torch.eye(self.size, device=self.device)

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
            return torch.eye(self.size, device=self.device)
    
    @property
    def inv_cov(self):
        if self.correlated_factors != [] or not self.fixed_variances:
            return torch.cholesky_solve(torch.eye(self.size, device=self.device),
                                        self.weight)
        else:
            return torch.eye(self.size, device=self.device)

        
################################################################################
#
# Autoencoder modules
#
################################################################################
        
    
class VariationalAutoencoder(nn.Module):
    
    def __init__(self,
                 decoder,
                 inference_net_sizes:   List[int],
                 latent_size:           int,
                 device:                str,
                 fixed_variances:       bool = True,
                 correlated_factors:    List[int] = [],
                 **decoder_kwargs,
                ):
        """
        Variational autoencoder with an interchangeable measurement model (i.e., decoder).
        
        Args:
            decoder             (nn.Module):   Measurement model whose forward() method returns log p(data | latents).
            inference_net_sizes (List of int): Neural network input and hidden layer dimensions.
                                                   E.g., a neural network with input size 50 and two hidden layers of
                                                   size 100 has inference_net_sizes = [50, 100, 100]
            latent_size         (int):         Number of latent variables.
            device              (str):         Computing device used for fitting.
            fixed_variances     (bool):        Whether to constrain latent variances to one.
            correlated_factors  (List of int): Which latent variables should be correlated.
            decoder_kwargs      (dict):        Named parameters passed to decoder.__init__().
        """
        super(VariationalAutoencoder, self).__init__()
        
        # Inference model neural network.
        inf_list = list(chain.from_iterable((nn.Linear(size1, size2), nn.ELU()) for size1, size2 in
                        zip(inference_net_sizes[0:-1], inference_net_sizes[1:])))
        if inf_list != []:
            inf_list.append(nn.Linear(inference_net_sizes[-1], int(2 * latent_size)))
            self.inf_net = nn.Sequential(*inf_list)
        else:
            self.inf_net = nn.Linear(inference_net_sizes[0], int(2 * latent_size))
        
        # Measurement model.
        self.decoder = decoder(latent_size=latent_size, **decoder_kwargs)
        
        # Latent prior.
        self.cholesky = Spherical(latent_size, fixed_variances, correlated_factors, device)
        self.latent_size = latent_size
        
        self.reset_parameters()
        
    def reset_parameters(self):
        try:
            nn.init.normal_(self.inf_net[-1].weight, mean=0., std=0.001)
            nn.init.normal_(self.inf_net[-1].bias[0:self.latent_size], mean=0., std=0.001)
            nn.init.normal_(self.inf_net[-1].bias[self.latent_size:], mean=math.log(math.exp(1) - 1), std=0.001)
        except TypeError:
            nn.init.normal_(self.inf_net.weight, mean=0., std=0.001)
            nn.init.normal_(self.inf_net.bias[0:self.latent_size], mean=0., std=0.001)
            nn.init.normal_(self.inf_net.bias[self.latent_size:], mean=math.log(math.exp(1) - 1), std=0.001)

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
            mc_samples     (int):    Number of Monte Carlo samples.
            iw_samples     (int):    Number of importance-weighted samples.
        """
        mu, std = self.encode(y, mc_samples, iw_samples)
        x = mu + std * torch.randn_like(mu)
        
        # Log p(y | x).
        log_py_x = self.decoder(x, y, mask)
        
        # Log p(x).
        if self.cholesky.correlated_factors != []:
            log_px = dist.MultivariateNormal(torch.zeros_like(x, device = x.device),
                                             scale_tril = self.cholesky.weight).log_prob(x).unsqueeze(-1)
        else:
            log_px = dist.Normal(torch.zeros_like(x, device = x.device),
                                 torch.ones_like(x, device = x.device)).log_prob(x).sum(-1, keepdim = True)
            
        # Log q(x | y).
        if iw_samples > 1 and grad_estimator == "dreg":
            qx_y = dist.Normal(mu.detach(), std.detach())
        else:
            qx_y = dist.Normal(mu, std)
        log_qx_y = qx_y.log_prob(x).sum(dim = -1, keepdim = True)
        
        elbo = log_py_x + log_qx_y - log_px
        return elbo, x