import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import pyro.distributions as pydist
import pyro.distributions.transforms as T
from pyro.nn import DenseNN
from deepirtools.utils import get_thresholds
from deepirtools.settings import EPS
from typing import List, Optional, Union
from operator import itemgetter
import itertools
import inspect
import logging


################################################################################
#
# Helper modules
#
################################################################################


class SparseLinear(nn.Module):
    """Linear map with binary constraints."""
    
    def __init__(self,
                 in_features:  int,
                 out_features: int,
                 Q:            torch.Tensor,
                ):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.free_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("Q", Q)
        
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
    """Linear map with linear constraints."""
    
    def __init__(self,
                 in_features:  int,
                 out_features: int,
                 A:            torch.Tensor,
                 b:            Optional[torch.Tensor] = None,
                ):
        super(LinearConstraints, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.free_weight = nn.Parameter(torch.empty(A.shape[0]))
        self.register_buffer("A", A)
        self.register_buffer("b", b)
        
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
    """Biases (i.e., intercepts) for categorical response models."""
    
    def __init__(self,
                 n_cats: List[int],
                 ints_mask: Optional[torch.Tensor] = None,
                ):
        super(CategoricalBias, self).__init__()
        self.n_cats = n_cats
        M = max(n_cats)
        n_items = len(n_cats)
        
        if ints_mask is None:
            ints_mask = torch.ones([n_items])
            
        bias_list = []
        for i, n_cat in enumerate(n_cats):
            thresholds = get_thresholds([ints_mask[i].item() * -4, 4], n_cat)
            bias_list.append(F.pad(thresholds, (0, M - n_cat),
                                   value = float("inf"))) # Inf. saturates exponentials.
        self._bias = nn.Parameter(torch.stack(bias_list, dim = 0))
        self.register_buffer("ints_mask", torch.cat([ints_mask.unsqueeze(1), torch.ones([n_items, M - 2])], dim = 1))
        
        nan_mask = torch.where(self._bias.isinf(), torch.ones_like(self._bias) * float("nan"),
                               torch.ones_like(self._bias))
        self.register_buffer("nan_mask", nan_mask)

    def forward(self,
                x: torch.Tensor,
               ):
        return (self._bias * self.ints_mask) + x
    
    @property
    def bias(self):
        bias = self._bias * self.nan_mask * self.ints_mask
        if bias.shape[1] == 1:
            return bias.squeeze()
        return bias

    
################################################################################
#
# Measurement model modules
#
################################################################################
    

class GradedBaseModel(nn.Module):
    """Base model for graded responses."""
    
    def __init__(self,
                 latent_size:  int,
                 n_cats:       List[int],
                 Q:            Optional[torch.Tensor] = None,
                 A:            Optional[torch.Tensor] = None,
                 b:            Optional[torch.Tensor] = None,
                 ints_mask:    Optional[torch.Tensor] = None,
                 _no_loadings: bool = False,
                ):
        super(GradedBaseModel, self).__init__()
        
        if not _no_loadings:
            assert(not (Q is not None and (A is not None or b is not None))), "Q and (A, b) may not be specified at the same time."
            if Q is not None:
                assert(((Q == 0) + (Q == 1)).all()), "Q must only contain ones and zeros."
                assert(len(Q.shape) == 2), "Q must be 2D."
                self._loadings = SparseLinear(latent_size, len(n_cats), Q)
            elif A is not None:
                assert(len(A.shape) == 2), "A must be 2D."
                if b is not None:
                    assert(len(b.shape) == 1), "b must be 1D."
                self._loadings = LinearConstraints(latent_size, len(n_cats), A, b)
            else:
                self._loadings = nn.Linear(latent_size, len(n_cats), bias = False)
            self.Q = Q
            self.A = A
        self._no_loadings = _no_loadings
        
        if ints_mask is not None:
            assert(len(ints_mask.shape) == 1), "ints_mask must be 1D."
            assert(((ints_mask == 0) + (ints_mask == 1)).all()), "ints_mask must only contain ones and zeros."
        self._intercepts = CategoricalBias(n_cats, ints_mask)
        self.n_cats = n_cats
        
        self.reset_parameters()
        
    def reset_parameters(self):
        if not self._no_loadings:
            if self.Q is None and self.A is None:
                nn.init.xavier_uniform_(self._loadings.weight)

    def forward(self):
        """Compute log p(data | latents)."""
        raise NotImplementedError
        
    @property
    def loadings(self):
        try:
            return self._loadings.weight.data
        except AttributeError:
            return None
    
    @property
    def intercepts(self):
        return self._intercepts.bias.data
    
    
class GradedResponseModel(GradedBaseModel):
    """Samejima's graded response model."""
    
    def __init__(self,
                 latent_size:  int,
                 n_cats:       List[int],
                 Q:            Optional[torch.Tensor] = None,
                 A:            Optional[torch.Tensor] = None,
                 b:            Optional[torch.Tensor] = None,
                 ints_mask:    Optional[torch.Tensor] = None,
                 _no_loadings: bool = False,
                ):
        super().__init__(latent_size = latent_size, n_cats = n_cats, Q = Q, A = A, b = b,
                         ints_mask = ints_mask, _no_loadings = _no_loadings)

    def forward(self,
                x:        torch.Tensor,
                y:        torch.Tensor,
                mask:     Optional[torch.Tensor] = None,
                loadings: Optional[torch.Tensor] = None,
               ):
        if loadings is None:
            Bx = self._loadings(x)
        else:
            Bx = F.linear(x, loadings)
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
    """Generalized partial credit model."""
    
    def __init__(self,
                 latent_size:  int,
                 n_cats:       List[int],
                 Q:            Optional[torch.Tensor] = None,
                 A:            Optional[torch.Tensor] = None,
                 b:            Optional[torch.Tensor] = None,
                 ints_mask:    Optional[torch.Tensor] = None,
                 _no_loadings: bool = False,
                ):
        super().__init__(latent_size = latent_size, n_cats = n_cats, Q = Q, A = A, b = b,
                         ints_mask = ints_mask, _no_loadings = _no_loadings)

    def forward(self,
                x:        torch.Tensor,
                y:        torch.Tensor,
                mask:     Optional[torch.Tensor] = None,
                loadings: Optional[torch.Tensor] = None,
               ):
        M = max(self.n_cats)
        
        if loadings is None:
            Bx = self._loadings(x)
        else:
            Bx = F.linear(x, loadings)
        shape = Bx.shape + torch.Size([M])
        kBx = Bx.unsqueeze(-1).expand(shape) * torch.linspace(0, M - 1, M)
        
        cum_bias = self._intercepts._bias.mul(self._intercepts.ints_mask).cumsum(dim = 1)
        cum_bias = F.pad(cum_bias, (1, 0), value = 0.).expand(shape)
        tmp = kBx - cum_bias
        
        log_py_x = tmp - (tmp).logsumexp(dim = -1, keepdim = True)
        
        idxs = y.long().expand(log_py_x[..., -1].shape).unsqueeze(-1)
        log_py_x = -torch.gather(log_py_x, dim = -1, index = idxs).squeeze(-1)
        if mask is not None:
            log_py_x = log_py_x.mul(mask)
        return log_py_x.sum(dim = -1, keepdim = True)
    
    
class NonGradedBaseModel(nn.Module):
    """Base model for non-graded responses."""
    
    def __init__(self,
                 latent_size:  int,
                 n_items:      int,
                 Q:            Optional[torch.Tensor] = None,
                 A:            Optional[torch.Tensor] = None,
                 b:            Optional[torch.Tensor] = None,
                 ints_mask:    Optional[torch.Tensor] = None,
                 _no_loadings: bool = False,
                ):
        super(NonGradedBaseModel, self).__init__()
        
        if not _no_loadings:
            assert(not (Q is not None and (A is not None or b is not None))), "Q and (A, b) may not be specified at the same time."
            if Q is not None:
                assert(((Q == 0) + (Q == 1)).all()), "Q must only contain ones and zeros."
                assert(len(Q.shape) == 2), "Q must be 2D."
                self._loadings = SparseLinear(latent_size, n_items, Q)
            elif A is not None:
                assert(len(A.shape) == 2), "A must be 2D."
                if b is not None:
                    assert(len(b.shape) == 1), "b must be 1D."
                self._loadings = LinearConstraints(latent_size, n_items, A, b)
            else:
                self._loadings = nn.Linear(latent_size, n_items, bias = False)
            self.Q = Q
            self.A = A
        self._no_loadings = _no_loadings
        
        if ints_mask is not None:
            assert(len(ints_mask.shape) == 1), "ints_mask must be 1D."
            assert(((ints_mask == 0) + (ints_mask == 1)).all()), "ints_mask must only contain ones and zeros."
        self._bias = nn.Parameter(torch.empty(n_items))
        self.register_buffer("ints_mask", ints_mask)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        if not self._no_loadings:
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
        try:
            return self._loadings.weight.data
        except AttributeError:
            return None
        
    @property
    def intercepts(self):
        return self.bias.data
    
    
class PoissonFactorModel(NonGradedBaseModel):
    """Poisson factor model."""
    
    def __init__(self,
                 latent_size:  int,
                 n_items:      int,
                 Q:            Optional[torch.Tensor] = None,
                 A:            Optional[torch.Tensor] = None,
                 b:            Optional[torch.Tensor] = None,
                 ints_mask:    Optional[torch.Tensor] = None,
                 _no_loadings: bool = False,
                ):
        super().__init__(latent_size = latent_size, n_items = n_items, Q = Q, A = A, b = b,
                         ints_mask = ints_mask, _no_loadings = _no_loadings)
            
    def forward(self,
                x:        torch.Tensor,
                y:        torch.Tensor,
                mask:     Optional[torch.Tensor] = None,
                loadings: Optional[torch.Tensor] = None,
               ):
        if loadings is None:
            log_rate = self._loadings(x) + self.bias
        else:
            log_rate = F.linear(x, loadings, self.bias)
        
        py_x = pydist.Poisson( # Clamp for numerical stability.
            rate = log_rate.clamp(min = math.log(EPS), max = -math.log(EPS)).exp(),
        )
        return -py_x.log_prob(y).sum(-1, keepdim = True)
    
    
class NegativeBinomialFactorModel(NonGradedBaseModel):
    """Negative binomial factor model."""
    
    def __init__(self,
                 latent_size:  int,
                 n_items:      int,
                 Q:            Optional[torch.Tensor] = None,
                 A:            Optional[torch.Tensor] = None,
                 b:            Optional[torch.Tensor] = None,
                 ints_mask:    Optional[torch.Tensor] = None,
                 _no_loadings: bool = False,
                ):
        super().__init__(latent_size = latent_size, n_items = n_items, Q = Q, A = A, b = b,
                         ints_mask = ints_mask, _no_loadings = _no_loadings)
        
        self.logits = nn.Parameter(torch.empty(n_items))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.logits, mean=0., std=0.001)
            
    def forward(self,
                x:        torch.Tensor,
                y:        torch.Tensor,
                mask:     Optional[torch.Tensor] = None,
                loadings: Optional[torch.Tensor] = None,
               ):
        if loadings is None:
            log_total_count = self._loadings(x) + self.bias
        else:
            log_total_count = F.linear(x, loadings, self.bias)
        
        py_x = pydist.NegativeBinomial( # Clamp for numerical stability.
            total_count = log_total_count.clamp(min = math.log(EPS), max = -math.log(EPS)).exp(),
            logits = self.logits,
        )
        return -py_x.log_prob(y).sum(-1, keepdim = True)
    
    
class NormalFactorModel(NonGradedBaseModel):
    """Normal (linear) factor model."""
    
    def __init__(self,
                 latent_size:  int,
                 n_items:      int,
                 Q:            Optional[torch.Tensor] = None,
                 A:            Optional[torch.Tensor] = None,
                 b:            Optional[torch.Tensor] = None,
                 ints_mask:    Optional[torch.Tensor] = None,
                 _no_loadings: bool = False,
                ):
        super().__init__(latent_size = latent_size, n_items = n_items, Q = Q, A = A, b = b,
                         ints_mask = ints_mask, _no_loadings = _no_loadings)
        
        self.free_phi = nn.Parameter(torch.empty(n_items))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.free_phi, mean=math.log(math.exp(1) - 1), std=0.001)
            
    def forward(self,
                x:        torch.Tensor,
                y:        torch.Tensor,
                mask:     Optional[torch.Tensor] = None,
                loadings: Optional[torch.Tensor] = None,
               ):
        if loadings is None:
            loc = self._loadings(x) + self.bias
        else:
            loc = F.linear(x, loadings, self.bias)
        
        py_x = pydist.Normal(loc = loc, scale = self.residual_std)
        return -py_x.log_prob(y).sum(-1, keepdim = True)
    
    @property
    def residual_std(self):
        return F.softplus(self.free_phi) + EPS

    
class LogNormalFactorModel(NonGradedBaseModel):
    """Lognormal factor model."""
    
    def __init__(self,
                 latent_size:  int,
                 n_items:      int,
                 Q:            Optional[torch.Tensor] = None,
                 A:            Optional[torch.Tensor] = None,
                 b:            Optional[torch.Tensor] = None,
                 ints_mask:    Optional[torch.Tensor] = None,
                 _no_loadings: bool = False,
                ):
        super().__init__(latent_size = latent_size, n_items = n_items, Q = Q, A = A, b = b,
                         ints_mask = ints_mask, _no_loadings = _no_loadings)
        
        self.free_phi = nn.Parameter(torch.empty(n_items))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.free_phi, mean=math.log(math.exp(1) - 1), std=0.001)
            
    def forward(self,
                x:        torch.Tensor,
                y:        torch.Tensor,
                mask:     Optional[torch.Tensor] = None,
                loadings: Optional[torch.Tensor] = None,
               ):
        if loadings is None:
            loc = self._loadings(x) + self.bias
        else:
            loc = F.linear(x, loadings, self.bias)
        
        py_x = pydist.LogNormal(loc = loc, scale = self.residual_std)
        return -py_x.log_prob(y).sum(-1, keepdim = True)
    
    @property
    def residual_std(self):
        return F.softplus(self.free_phi) + EPS
    
    
class ModelTypes():
    
    MODEL_TYPES = {"grm" : GradedResponseModel,
                   "gpcm" : GeneralizedPartialCreditModel,
                   "poisson" : PoissonFactorModel,
                   "negative_binomial" : NegativeBinomialFactorModel,
                   "normal" : NormalFactorModel,
                   "lognormal" : LogNormalFactorModel,
                  }
    
    
class MixedFactorModel(nn.Module):
    """Factor model with mixed item types."""
    
    def __init__(self,
                 latent_size: int,
                 model_types: List[str],
                 n_cats:      Optional[List[Union[int, None]]] = None,
                 n_items:     Optional[int] = None,
                 Q:           Optional[torch.Tensor] = None,
                 A:           Optional[torch.Tensor] = None,
                 b:           Optional[torch.Tensor] = None,
                 ints_mask:   Optional[torch.Tensor] = None,
                ):
        super(MixedFactorModel, self).__init__()
        assert(not (n_items is None and n_cats is None)), "Must define either n_items or n_cats."
        if n_items is None:
            n_items = len(n_cats)
            
        sorted_idxs, sorted_model_types = zip(*sorted(enumerate(model_types), key = itemgetter(1)))
        n_items_per_model = [len(list(g)) for k, g in itertools.groupby(sorted_model_types)]
        self.register_buffer("cum_idxs", torch.Tensor([0] + n_items_per_model).cumsum(dim  = 0).long())
        self.register_buffer("sorted_idxs", torch.Tensor(sorted_idxs).long())
        
        unsorted_idxs = torch.zeros(n_items).long()
        unsorted_idxs[self.sorted_idxs] = torch.arange(n_items)
        self.register_buffer("unsorted_idxs", unsorted_idxs)
            
        assert(not (Q is not None and (A is not None or b is not None))), "Q and (A, b) may not be specified at the same time."
        if Q is not None:
            assert(((Q == 0) + (Q == 1)).all()), "Q must only contain ones and zeros."
            assert(len(Q.shape) == 2), "Q must be 2D."
            self._loadings = SparseLinear(latent_size, n_items, Q)
            check_mat = Q[self.sorted_idxs]
        elif A is not None:
            assert(len(A.shape) == 2), "A must be 2D."
            if b is not None:
                assert(len(b.shape) == 1), "b must be 1D."
            self._loadings = LinearConstraints(latent_size, n_items, A, b)
            check_mat = F.linear(torch.ones([A.shape[1]]), A, b).view(latent_size, n_items).T[self.sorted_idxs]
        else:
            self._loadings = nn.Linear(latent_size, n_items, bias = False)
            check_mat = torch.ones([n_items, latent_size])
        self.Q = Q
        self.A = A
        
        unique_model_types = list(dict.fromkeys(sorted_model_types))
        _models = tuple(ModelTypes().MODEL_TYPES[m] for m in unique_model_types)
        
        keep_idxs = []
        models = []
        for model_idx, (idx1, idx2) in enumerate(zip(self.cum_idxs[:-1], self.cum_idxs[1:])):
            keep_idx = check_mat[idx1:idx2].sum(dim = 0).gt(0)
            keep_idxs.append(keep_idx)
            if ints_mask is not None:
                assert(len(ints_mask.shape) == 1), "ints_mask must be 1D."
                assert(((ints_mask == 0) + (ints_mask == 1)).all()), "ints_mask must only contain ones and zeros."
                _ints_mask = ints_mask[self.sorted_idxs][idx1:idx2]
            else:
                _ints_mask = None
            if unique_model_types[model_idx] in ("grm", "gpcm"):
                model_kwargs = {"n_cats" : [n_cats[i] for i in sorted_idxs][idx1:idx2]}
            else:
                model_kwargs = {"n_items" : (idx2 - idx1).item()}
            models.append(_models[model_idx](latent_size = max(1, keep_idx.sum().item()),
                                             ints_mask = _ints_mask, _no_loadings = True, **model_kwargs))
            self.register_buffer("keep_idxs", torch.stack(keep_idxs, dim = 0))
            self.models = nn.ModuleList(models)
            
            self.reset_parameters()
            
    def reset_parameters(self):
        if self.Q is None and self.A is None:
            nn.init.xavier_uniform_(self._loadings.weight)

    def forward(self,
                x:        torch.Tensor,
                y:        torch.Tensor,
                mask:     Optional[torch.Tensor] = None,
               ):
        ldgs_sorted = self._loadings.weight[self.sorted_idxs]
        y_sorted = y[:, self.sorted_idxs]
        if mask is not None:
            mask_sorted = mask[:, self.sorted_idxs]
            
        out = []
        for model_idx, (idx1, idx2) in enumerate(zip(self.cum_idxs[:-1], self.cum_idxs[1:])):
            keep_idx = self.keep_idxs[model_idx]
            if keep_idx.sum().gt(0):
                _ldgs_sorted = ldgs_sorted[idx1:idx2, keep_idx]
                _x = x[..., keep_idx]
            else: # Intercepts-only model.
                _ldgs_sorted = torch.zeros([idx2 - idx1, 1])
                _x = torch.zeros(x.size()[:-1] + torch.Size([1]))
            _y_sorted = y_sorted[:, idx1:idx2]
            if mask is not None:
                _mask_sorted = mask_sorted[:, idx1:idx2]
            else:
                _mask_sorted = None
            out.append(self.models[model_idx](x = _x, y = _y_sorted, mask = _mask_sorted,
                                              loadings = _ldgs_sorted))
        
        return torch.cat(out, dim = -1).sum(-1, keepdim = True)
    
    @property
    def loadings(self):
        return self._loadings.weight.data
        
    @property
    def intercepts(self):
        ints = [m.intercepts.unsqueeze(1) if len(m.intercepts.shape) == 1 else
                m.intercepts for m in self.models]
        M = max([i.shape[-1]for i in ints])
        ints = torch.cat([F.pad(i, (0, M - i.shape[1]), value = float("nan")) for
                          i in ints], dim = 0)[self.unsorted_idxs]
        return (ints.squeeze() if ints.shape[1] == 1 else ints)
    
    @property
    def residual_std(self):
        try:
            residual_stds = []
            for m in self.models:
                try:
                    residual_stds.append(m.residual_std)
                except AttributeError:
                    residual_stds.append(torch.zeros(m.intercepts.shape[0]) * float("nan"))
            residual_std = torch.cat(residual_stds, dim = 0)[self.unsorted_idxs]
            assert(~residual_std.isnan().all())
            return residual_std
        except AssertionError:
            return None
    
    @property
    def logits(self):
        try:
            logits_list = []
            for m in self.models:
                try:
                    logits_list.append(m.logits)
                except AttributeError:
                    logits_list.append(torch.zeros(m.intercepts.shape[0]) * float("nan"))
            logits = torch.cat(logits_list, dim = 0)[self.unsorted_idxs]
            assert(~logits.isnan().all())
            return logits
        except AssertionError:
            return None
    
    
################################################################################
#
# Latent prior modules
#
################################################################################
    

class Spherical(nn.Module):
    """Spherical parameterization of a covariance matrix."""
    
    def __init__(self,
                 size:               int,
                 fixed_variances:    bool,
                 correlated_factors: List[int],
                ):
        super(Spherical, self).__init__()
        if correlated_factors != []:
            assert(max(correlated_factors) <= size - 1), ("correlated_factors may include no values ",
                                                          "larger than {}.".format(size - 1))

        self.size = size
        self.fixed_variances = fixed_variances
        self.correlated_factors = correlated_factors
        self.model_correlations = (len(correlated_factors) > 1)
        
        n_elts = int((size * (size + 1)) / 2)
        self.theta = nn.Parameter(torch.zeros([n_elts]))
        diag_idxs = torch.arange(1, size + 1).cumsum(dim = 0) - 1
        self.theta.data[diag_idxs] = math.log(math.pi / 2) # TODO: Change init s.t. cov is identity.

        if self.model_correlations:
            tril_idxs = torch.tril_indices(row = size - 1, col = size - 1, offset = 0)
            uncorrelated_factors = [factor for factor in [i for i in range(size)] if factor not in correlated_factors]
            self.register_buffer("uncorrelated_tril_idxs", tril_idxs[:, sum((tril_idxs[1,:] == factor) + (tril_idxs[0,:] == factor - 1) for
                                                                     factor in uncorrelated_factors) > 0])
            
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
        if self.model_correlations or not self.fixed_variances:
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
            if self.model_correlations:
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
        if self.model_correlations or not self.fixed_variances:
            weight = self.weight
            return torch.matmul(weight, weight.t())
        else:
            return torch.eye(self.size, device=self.theta.device)
    
    @property
    def inv_cov(self):
        if self.model_correlations or not self.fixed_variances:
            return torch.cholesky_solve(torch.eye(self.size, device=self.theta.device),
                                        self.weight)
        else:
            return torch.eye(self.size, device=self.theta.device)
        
        
def spline_coupling(input_dim, count_bins=32, bound=5.):
    """Modification of Pyro's spline_coupling()."""
    
    split_dim = input_dim // 2
    hidden_dims = [min(100, input_dim * 10)]

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
    """Variational autoencoder with an interchangeable measurement model (i.e., decoder)."""
    
    def __init__(self,
                 decoder,
                 latent_size:           int,
                 inference_net_sizes:   List[int] = [100],
                 fixed_variances:       bool = True,
                 fixed_means:           bool = True,
                 correlated_factors:    List[int] = [],
                 covariate_size:        int = 0,
                 use_spline_prior:      bool = False,
                 **kwargs,
                ):
        super(VariationalAutoencoder, self).__init__()
                
        # Measurement model.
        decoder_args = list(inspect.signature(decoder).parameters)
        decoder_kwargs = {k: kwargs.pop(k) for k in dict(kwargs) if k in decoder_args}
        self.decoder = decoder(latent_size=latent_size, **decoder_kwargs)
        
        # Inference model neural network.
        try:
            input_size = len(decoder_kwargs["n_cats"])
        except KeyError:
            input_size = decoder_kwargs["n_items"]
        self.inf_net = DenseNN(input_size + covariate_size, inference_net_sizes,
                               [int(2 * latent_size)], nonlinearity = nn.ELU())
        
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
                self.flow1 = spline_coupling(latent_size, **spline_kwargs)
                self.flow2 = T.Permute(torch.Tensor(list(reversed(range(latent_size)))).long())
                self.flow3 = spline_coupling(latent_size, **spline_kwargs)
        else:
            self.cholesky = Spherical(latent_size, fixed_variances, correlated_factors)
            if not fixed_means:
                self.mean = nn.Parameter(torch.empty([1, latent_size]))
            else:
                self.register_buffer("mean", torch.zeros([1, latent_size]))
        self.latent_size = latent_size
        self.fixed_variances = fixed_variances
        self.fixed_means = fixed_means
        self.use_spline_prior = use_spline_prior
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.inf_net.layers[-1].weight, mean=0., std=0.001)
        nn.init.normal_(self.inf_net.layers[-1].bias[0:self.latent_size], mean=0., std=0.001)
        nn.init.normal_(self.inf_net.layers[-1].bias[self.latent_size:], mean=math.log(math.exp(1) - 1), std=0.001)
        
        if not self.fixed_means:
            nn.init.normal_(self.mean, mean=0., std=0.001)
        
        if self.cov_size > 0:
            nn.init.normal_(self.lreg_weight, mean=0., std=0.001)
        
        if self.use_spline_prior:
            device = self.inf_net.layers[0].weight.device
            params = [p.parameters() for p in self.__get_flow() if hasattr(p, "parameters")]
            lr = (0.1 / (self.latent_size + 1))*5**-1
            optimizer = Adam([{"params" : itertools.chain(*params)}], lr = lr, amsgrad = True)
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
        """Compute evidence lower bound (ELBO)."""
        
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
                scale = x.var(dim = -2, keepdim = True).add(EPS).sqrt().pow(-1)
            else:
                scale = torch.ones_like(x, device = x.device)
            if self.fixed_means:
                loc = -x.mean(dim = -2, keepdim = True) * scale
            else:
                loc = torch.zeros_like(x, device = x.device)
            flow.append(T.AffineTransform(loc = loc, scale = scale))
            px = pydist.TransformedDistribution(base_dist, flow)
            log_px = px.log_prob(x).unsqueeze(-1)
        else:
            if self.cov_size > 0:
                loc = F.linear(covariates, self.lreg_weight)
            else:
                loc = torch.zeros_like(x, device = x.device)
            if not self.fixed_means:
                loc.add_(self.mean)
            if self.cholesky.model_correlations:
                log_px = pydist.MultivariateNormal(loc, scale_tril = self.cholesky.weight).log_prob(x).unsqueeze(-1)
            else:
                log_px = pydist.Normal(loc, scale = self.cholesky.cov.diag().sqrt().view(1, -1)).log_prob(x).sum(dim = -1, keepdim = True)
            
        # Log q(x | y, covariates).
        if iw_samples > 1 and grad_estimator == "dreg":
            qx_y = pydist.Normal(mu.detach(), std.detach())
        else:
            qx_y = pydist.Normal(mu, std)
        log_qx_y = qx_y.log_prob(x).sum(dim = -1, keepdim = True)
        
        elbo = log_py_x + log_qx_y - log_px
        return elbo, x