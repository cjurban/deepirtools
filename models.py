import math
import torch
from torch import nn
import torch.nn.functional as F
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
        """
        Args:
        """
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
        """
        Args:
        """
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
        """
        Args:
        """
        super(CatBiasReshape, self).__init__()
        self.n_cats = n_cats
        
        # Biases.
        bias = torch.empty(sum([n_cat - 1 for n_cat in n_cats]))
        if mask is None:
            mask = torch.ones_like(bias)
        idxs = np.cumsum([n_cat - 1 for n_cat in ([1] + n_cats)])
        bias.data = torch.from_numpy(np.hstack([get_thresholds([mask[idx].item() * -4, 4], n_cat) for
                                                idx, n_cat in zip(idxs[:-1], n_cats)]))
        bias_reshape, sliced_bias = self._reshape(bias, idxs, 9999.)
        self.bias_reshape = nn.Parameter(bias_reshape)
        if mask is not None:
            self.mask = self._reshape(mask, idxs, 0., False)
        
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
        return self.bias_reshape.view(-1)[self.drop_idxs] * self.mask.view(-1)

    
################################################################################
#
# Measurement model modules
#
################################################################################
    

class GradedResponseModel(nn.Module):
    def __init__(self,
                 latent_size: int,
                 n_cats:      List[int],
                 Q:           Optional[torch.Tensor] = None,
                 A:           Optional[torch.Tensor] = None,
                 b:           Optional[torch.Tensor] = None,
                 ints_mask:   Optional[torch.Tensor] = None,
                ):
        """
        Args:
        """
        super(GradedResponseModel, self).__init__()
        
        # Loadings.
        assert(not (Q is not None and A is not None)) # print errors at asserts
        if Q is not None:
            self.loadings = SparseLinear(latent_size, len(n_cats), Q)
        elif A is not None:
            self.loadings = LinearConstraints(latent_size, len(n_cats), A, b)
        else:
            self.loadings = nn.Linear(latent_size, len(n_cats), bias = False)
        self.Q = Q
        self.A = A
        
        # Intercepts.
        self.intercepts = CatBiasReshape(n_cats, ints_mask)
        self.n_cats = n_cats
        
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.Q is None and self.A is None:
            nn.init.xavier_uniform_(self.loadings.weight)

    def forward(self,
                x: torch.Tensor,
               ):
        Bx = self.loadings(x)
        cum_probs = self.intercepts(Bx.unsqueeze(-1).expand(Bx.shape +
                                                                 torch.Size([max(self.n_cats) - 1]))).sigmoid()
        upper_probs = F.pad(cum_probs, (0, 1), value=1.)
        lower_probs = F.pad(cum_probs, (1, 0), value=0.)
        probs = upper_probs - lower_probs
        
        return probs
    
    
################################################################################
#
# Latent prior modules
#
################################################################################
    

class Spherical(nn.Module):
    
    def __init__(self,
                 size:               int,
                 correlated_factors: List[int],
                 device:             str,
                ):
        """
        Args:
        """
        super(Spherical, self).__init__()
        self.size = size
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
        if self.correlated_factors != []:
            tril_idxs = torch.tril_indices(row = self.size, col = self.size, offset = 0)
            theta_mat = torch.zeros(self.size, self.size, device=self.device)
            theta_mat[tril_idxs[0], tril_idxs[1]] = self.theta

            # Ensure the parameterization is unique.
            exp_theta_mat = torch.zeros(self.size, self.size, device=self.device)
            exp_theta_mat[tril_idxs[0], tril_idxs[1]] = self.theta.exp()
            lower_tri_l_mat = (np.pi * exp_theta_mat) / (1 + theta_mat.exp())
            l_mat = exp_theta_mat.diag().diag_embed() + lower_tri_l_mat.tril(diagonal = -1)

            # Constrain variances to one.
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
        if self.correlated_factors != []:
            weight = self.weight

            return torch.matmul(weight, weight.t())
        else:
            return torch.eye(self.size, device=self.device)
    
    @property
    def inv_cov(self):
        if self.correlated_factors != []:
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
                 input_size:            int,
                 inference_net_sizes:   List[int],
                 latent_size:           int,
                 device:                str,
                 correlated_factors:    List[int] = [],
                 **decoder_kwargs,
                ):
        """
        Args:
        """
        super(VariationalAutoencoder, self).__init__()
        
        # Inference model neural network.
        inf_sizes = [input_size] + inference_net_sizes
        inf_list = list(chain.from_iterable((nn.Linear(size1, size2), nn.ELU()) for size1, size2 in
                        zip(inf_sizes[0:-1], inf_sizes[1:])))
        if inf_list != []:
            inf_list.append(nn.Linear(inf_sizes[-1], int(2 * latent_size)))
            self.inf_net = nn.Sequential(*inf_list)
        else:
            self.inf_net = nn.Linear(inf_sizes[0], int(2 * latent_size))
        
        # Measurement model.
        self.decoder = decoder(latent_size=latent_size, **decoder_kwargs)
        
        # Latent prior.
        self.cholesky = Spherical(latent_size, correlated_factors, device)
        self.latent_size = latent_size
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Check type of inf_net -- if one linear layer, can't index
        nn.init.normal_(self.inf_net[-1].weight, mean=0., std=0.001)
        nn.init.normal_(self.inf_net[-1].bias[0:self.latent_size], mean=0., std=0.001)
        nn.init.normal_(self.inf_net[-1].bias[self.latent_size:], mean=math.log(math.exp(1) - 1), std=0.001)

    def encode(self,
               y,
               mc_samples,
               iw_samples):
        hidden = self.inf_net(y)

        # Monte Carlo samples.
        hidden = hidden.unsqueeze(0).expand(torch.Size([mc_samples]) + hidden.shape)
        
        # Importance-weighted samples.
        hidden = hidden.unsqueeze(0).expand(torch.Size([iw_samples]) + hidden.shape)
        
        mu, std = hidden.chunk(chunks = 2, dim = -1)
        std = F.softplus(std)
            
        return mu, std + EPS

    def forward(self,
                y,
                mc_samples = 1,
                iw_samples = 1):
        mu, std = self.encode(y, mc_samples, iw_samples)
        x = mu + std * torch.randn_like(mu)
        recon_y = self.decoder(x)
        
        return recon_y, mu, std, x