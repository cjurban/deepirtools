import random
import torch
import torch.nn.functional as F
import pyro.distributions as pydist
import numpy as np
from scipy.optimize import linear_sum_assignment
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, packages
from typing import List, Optional
import itertools
from operator import itemgetter
from deepirtools.utils import invert_factors


numpy2ri.activate()


class BaseFactorModelSimulator():
    
    def __init__(self,
                 cov_mat:    torch.Tensor,
                 mean:       torch.Tensor,
                ):
        """Base class for simulating from a latent factor model."""
        super(BaseFactorModelSimulator, self).__init__()

        self.cov_mat = cov_mat
        self.mean = mean
    
    @torch.no_grad()
    def _scores(self,
                sample_size: Optional[int] = None,
               ):
        x_dist = pydist.MultivariateNormal(loc = self.mean,
                                           covariance_matrix = self.cov_mat)
        if self.mean.shape[0] > 1:
            return x_dist.sample()
        else:
            return x_dist.sample([sample_size]).squeeze(dim = -2)
        
    def sample(self):
        raise NotImplementedError
        
        
class GradedResponseModelSimulator(BaseFactorModelSimulator):
    
    def __init__(self,
                 loadings:   torch.Tensor,
                 intercepts: torch.Tensor,
                 cov_mat:    torch.Tensor,
                 mean:       torch.Tensor,
                ):
        """ Simulate from Samejima's graded response model."""
        super().__init__(cov_mat = cov_mat, mean = mean)
        
        self.loadings = -loadings # TODO: Why are there GRM sign reversals?
        self.intercepts = intercepts
        
    @torch.no_grad()    
    def sample(self,
               sample_size: Optional[int] = None,
               x:           Optional[torch.Tensor] = None,
              ):
        assert(not ((sample_size is None) and (x is None))), "Must specify either sample_size or x."
        
        ro.r("rm(list = ls())")
        packages.importr("mirt")

        ldgs_R = ro.r.matrix(self.loadings.numpy(), nrow = self.loadings.shape[0],
                             ncol = self.loadings.shape[1])
        ints_R = ro.r.matrix(self.intercepts.numpy(), nrow = self.intercepts.shape[0],
                             ncol = self.intercepts.shape[1])
        if x is None:
            x = self._scores(sample_size)
        Theta_R = ro.r.matrix(x.numpy(), nrow = x.shape[0], ncol = x.shape[1])
        ro.r.assign("ldgs", ldgs_R); ro.r.assign("ints", ints_R); ro.r.assign("Theta", Theta_R)

        ro.r("""
                if (dim(ints)[2] > 1) {
                  itemtype = ifelse(is.na(ints[, 2]), "2PL", "graded")
                } else if (dim(ints)[2] == 1) {
                  itemtype = rep("2PL", dim(ints)[1])
                }

                ldgs = matrix(as.vector(t(ldgs)), nrow = dim(ldgs)[1], byrow = TRUE)
                ints = matrix(as.vector(t(ints)), nrow = dim(ints)[1], byrow = TRUE)
                Theta = matrix(as.vector(t(Theta)), nrow = dim(Theta)[1], byrow = TRUE)

                Y = simdata(a = ldgs,
                            d = ints,
                            itemtype = itemtype,
                            Theta = Theta
                           )
             """)

        return torch.from_numpy(ro.r["Y"])


class GeneralizedPartialCreditModelSimulator(BaseFactorModelSimulator):
    
    def __init__(self,
                 loadings:   torch.Tensor,
                 intercepts: torch.Tensor,
                 cov_mat:    torch.Tensor,
                 mean:       torch.Tensor,
                ):
        """ Simulate from the generalized partial credit model."""
        super().__init__(cov_mat = cov_mat, mean = mean)
        
        self.loadings = loadings
        self.intercepts = intercepts
        
    @torch.no_grad()    
    def sample(self,
               sample_size: Optional[int] = None,
               x:           Optional[torch.Tensor] = None,
              ):
        assert(not ((sample_size is None) and (x is None))), "Must specify either sample_size or x."
        
        ro.r("rm(list = ls())")
        packages.importr("mirt")

        ldgs_R = ro.r.matrix(self.loadings.numpy(), nrow = self.loadings.shape[0],
                             ncol = self.loadings.shape[1])
        ints_R = ro.r.matrix(self.intercepts.numpy(), nrow = self.intercepts.shape[0],
                             ncol = self.intercepts.shape[1])
        if x is None:
            x = self._scores(sample_size)
        Theta_R = ro.r.matrix(x.numpy(), nrow = x.shape[0], ncol = x.shape[1])
        ro.r.assign("ldgs", ldgs_R); ro.r.assign("ints", ints_R); ro.r.assign("Theta", Theta_R)

        ro.r("""
                ldgs = matrix(as.vector(t(ldgs)), nrow = dim(ldgs)[1], byrow = TRUE)
                ints = cbind(rep(0, dim(ints)[1]), ints)
                ints = matrix(as.vector(t(ints)), nrow = dim(ints)[1], byrow = TRUE)
                Theta = matrix(as.vector(t(Theta)), nrow = dim(Theta)[1], byrow = TRUE)

                Y = simdata(a = ldgs,
                            d = ints,
                            itemtype = rep("gpcm", dim(ints)[1]),
                            Theta = Theta
                           )
             """)
    
        return torch.from_numpy(ro.r["Y"])

        
class PoissonFactorModelSimulator(BaseFactorModelSimulator):
    
    def __init__(self,
                 loadings:   torch.Tensor,
                 intercepts: torch.Tensor,
                 cov_mat:    torch.Tensor,
                 mean:       torch.Tensor,
                ):
        """ Simulate from the Poisson factor model."""
        super().__init__(cov_mat = cov_mat, mean = mean)
        
        self.loadings = loadings
        self.intercepts = intercepts
        
    @torch.no_grad()    
    def sample(self,
               sample_size: Optional[int] = None,
               x:           Optional[torch.Tensor] = None,
              ):
        assert(not ((sample_size is None) and (x is None))), "Must specify either sample_size or x."
        
        if x is None:
            x = self._scores(sample_size)
        rate = F.linear(x, self.loadings, self.intercepts).exp()
        
        y_dist = pydist.Poisson(rate = rate)
        return y_dist.sample()
    
    
class NegativeBinomialFactorModelSimulator(BaseFactorModelSimulator):
    
    def __init__(self,
                 loadings:   torch.Tensor,
                 intercepts: torch.Tensor,
                 cov_mat:    torch.Tensor,
                 mean:       torch.Tensor,
                 probs:      torch.Tensor,
                ):
        """Simulate from the negative binomial factor model."""
        super().__init__(cov_mat = cov_mat, mean = mean)
        
        self.loadings = loadings
        self.intercepts = intercepts
        self.probs = probs
        
    @torch.no_grad()    
    def sample(self,
               sample_size: Optional[int] = None,
               x:           Optional[torch.Tensor] = None,
              ):
        assert(not ((sample_size is None) and (x is None))), "Must specify either sample_size or x."
        
        if x is None:
            x = self._scores(sample_size)
        total_count = F.linear(x, self.loadings, self.intercepts).exp()
        
        y_dist = pydist.NegativeBinomial(total_count = total_count, probs = self.probs)
        return y_dist.sample()
    
    
class NormalFactorModelSimulator(BaseFactorModelSimulator):
    
    def __init__(self,
                 loadings:     torch.Tensor,
                 intercepts:   torch.Tensor,
                 cov_mat:      torch.Tensor,
                 mean:         torch.Tensor,
                 residual_std: torch.Tensor,
                ):
        """Simulate from the normal factor model."""
        super().__init__(cov_mat = cov_mat, mean = mean)
        
        self.loadings = loadings
        self.intercepts = intercepts
        self.residual_std = residual_std
        
    @torch.no_grad()    
    def sample(self,
               sample_size: Optional[int] = None,
               x:           Optional[torch.Tensor] = None,
              ):
        assert(not ((sample_size is None) and (x is None))), "Must specify either sample_size or x."
        
        if x is None:
            x = self._scores(sample_size)
        loc = F.linear(x, self.loadings, self.intercepts)
        
        y_dist = pydist.Normal(loc = loc, scale = self.residual_std)
        return y_dist.sample()
    
    
class LogNormalFactorModelSimulator(BaseFactorModelSimulator):
    
    def __init__(self,
                 loadings:     torch.Tensor,
                 intercepts:   torch.Tensor,
                 cov_mat:      torch.Tensor,
                 mean:         torch.Tensor,
                 residual_std: torch.Tensor,
                ):
        """Simulate from the lognormal factor model."""
        super().__init__(cov_mat = cov_mat, mean = mean)
        
        self.loadings = loadings
        self.intercepts = intercepts
        self.residual_std = residual_std
        
    @torch.no_grad()    
    def sample(self,
               sample_size: Optional[int] = None,
               x:           Optional[torch.Tensor] = None,
              ):
        assert(not ((sample_size is None) and (x is None))), "Must specify either sample_size or x."
        
        if x is None:
            x = self._scores(sample_size)
        loc = F.linear(x, self.loadings, self.intercepts)
        
        y_dist = pydist.LogNormal(loc = loc, scale = self.residual_std)
        return y_dist.sample()
    
    
class Simulators():
    
    SIMULATORS = {"grm" : GradedResponseModelSimulator,
                  "gpcm" : GeneralizedPartialCreditModelSimulator,
                  "poisson" : PoissonFactorModelSimulator,
                  "negative_binomial" : NegativeBinomialFactorModelSimulator,
                  "normal" : NormalFactorModelSimulator,
                  "lognormal" : LogNormalFactorModelSimulator,
                 }
    
    
class MixedFactorModelSimulator(BaseFactorModelSimulator):
    
    def __init__(self,
                 model_types:  List[str],
                 simulators:   List,
                 cov_mat:      torch.Tensor,
                 mean:         torch.Tensor,
                ):
        """Simulate from a factor model with mixed item types."""
        super().__init__(cov_mat = cov_mat, mean = mean)
        self.cov_mat = cov_mat
        self.mean = mean
        self.sims = simulators
        
        n_items = len(model_types)
        sorted_idxs, sorted_model_types = zip(*sorted(enumerate(model_types), key = itemgetter(1)))
        unsorted_idxs = torch.zeros(n_items).long()
        unsorted_idxs[torch.Tensor(sorted_idxs).long()] = torch.arange(n_items)
        self.model_types = model_types
        self.unique_model_types = list(dict.fromkeys(sorted_model_types))
        self.unsorted_idxs = unsorted_idxs
        
    @torch.no_grad()
    def sample(self,
               sample_size: int,
              ):
        x = self._scores(sample_size)
        
        out = []
        for m in self.unique_model_types:
            out.append(self.sims[m].sample(x = x))
        return torch.cat(out, dim = 1)[:, self.unsorted_idxs]
    
    @property
    def loadings(self):
        ldgs = []
        for m in self.unique_model_types:
            ldgs.append((-self.sims[m].loadings if m == "grm" else self.sims[m].loadings))
        return torch.cat(ldgs, dim = 0)[self.unsorted_idxs]
    
    @property
    def intercepts(self):
        ints = [self.sims[m].intercepts.unsqueeze(1) if
                len(self.sims[m].intercepts.shape) == 1 else
                self.sims[m].intercepts for m in self.unique_model_types]
        M = max([i.shape[-1]for i in ints])
        ints = torch.cat([F.pad(i, (0, M - i.shape[1]), value = float("nan")) for
                          i in ints], dim = 0)[self.unsorted_idxs]
        ints = torch.stack([-i if m in ("grm", "gpcm") else i for
                            m, i in zip(self.model_types, ints)], dim=0)
        return (ints.squeeze() if ints.shape[1] == 1 else ints)
    
    @property
    def residual_std(self):
        residual_stds = []
        for m in self.unique_model_types:
            try:
                residual_stds.append(self.sims[m].residual_std)
            except AttributeError:
                residual_stds.append(torch.zeros(self.sims[m].intercepts.shape[0]) * float("nan"))
        return torch.cat(residual_stds, dim = 0)[self.unsorted_idxs]
    
    @property
    def probs(self):
        probs_list = []
        for m in self.unique_model_types:
            try:
                probs_list.append(self.sims[m].probs)
            except AttributeError:
                probs_list.append(torch.zeros(self.sims[m].intercepts.shape[0]) * float("nan"))
        return torch.cat(probs_list, dim = 0)[self.unsorted_idxs]
    
    @property
    def n_cats(self):
        return [sum(~i.isnan()).item() + 1 if m in ("grm", "gpcm") else None for
                i, m in zip(self.intercepts, self.model_types)]
        
    
def get_loadings(n_indicators:         int,
                 latent_size:          int,
                 reference_indicators: bool = False,
                ):
    """Simulate a factor loadings matrix."""
    n_items = int(n_indicators * latent_size)
    mask = torch.block_diag(*[torch.ones([n_indicators, 1])] * latent_size)
    ldgs = pydist.Uniform(0.5, 1.7).sample([n_items, latent_size]).mul(mask)
    if reference_indicators:
        for col in range(ldgs.shape[1]):
            ldgs[n_indicators * col, col] = 1
    
    return ldgs


def get_categorical_intercepts(n_items:         int,
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

    return torch.cat(ints, dim = 0)


def get_covariance_matrix(latent_size: int,
                          cov_type:    str,
                         ):
    """Simulate a factor covariance matrix."""
    cov_types = ("fixed_variances_no_covariances", "fixed_variances", "free")
    assert(cov_type in cov_types)
    if cov_type == cov_types[0]:
        cov_mat = torch.eye(latent_size)
    if cov_type == cov_types[1]:
        cov_mat = torch.ones([latent_size, latent_size]).mul(0.3)
        cov_mat.fill_diagonal_(1)
    if cov_type == cov_types[2]: # Sample from Wishart((0.4)_{latent_size X 1}, 5)
        cov_mat = torch.zeros([latent_size, latent_size])
        for i in range(5):
            G = torch.randn([latent_size, 1]).mul(0.4)
            cov_mat.add_(torch.mm(G, G.T))
        
    return cov_mat


def get_mean(latent_size: int,
             mean_type:   str,
             sample_size: Optional[int] = None,
            ):
    """Simulate a factor mean vector."""
    mean_types = ("fixed_means", "latent_regression", "free")
    assert(mean_type in mean_types)
    covariates = None; lreg_weight = None
    if mean_type == mean_types[0]:
        mean = torch.zeros([1, latent_size])
    elif mean_type == mean_types[1]:
        covariates = torch.cat((torch.randn(sample_size, 1),
                                torch.bernoulli(torch.ones(sample_size, 1) * 0.5)), dim = 1)
        lreg_weight = pydist.Uniform(-0.5, 0.5).sample([latent_size, 2])
        mean = F.linear(covariates, lreg_weight)
    elif mean_type == mean_types[2]:
        mean = pydist.Uniform(-0.5, 0.5).sample([1, latent_size])
        
    return mean, covariates, lreg_weight


def get_constraints(latent_size:     int,
                    n_indicators:    int,
                    constraint_type: str,
                   ):
    """Get loadings matrix constraints."""
    n_items = int(n_indicators * latent_size)
    constraint_types = ("none", "binary", "linear")
    assert(constraint_type in constraint_types)
    Q = None; A = None; b = None
    if constraint_type == "binary":
        Q = torch.block_diag(*[torch.ones([n_indicators, 1])] * latent_size)
    elif constraint_type == "linear":
        constraints = ([torch.zeros(1), torch.eye(n_indicators - 1),
                        torch.zeros([n_items, n_items])] * (latent_size - 1) +
                       [torch.zeros(1), torch.eye(n_indicators - 1)])
        A = torch.block_diag(*constraints)
        b = torch.cat([torch.ones(1), torch.zeros([n_indicators - 1 + n_items])] * (latent_size - 1) +
                      [torch.ones(1), torch.zeros([n_indicators - 1])], dim = 0)
    
    return {"Q" : Q, "A" : A, "b" : b}


def get_ints_mask(n_indicators:         int,
                  latent_size:          int,
                  reference_indicators: bool = False,
                 ):
    """Get intercepts mask."""
    n_items = int(n_indicators * latent_size)
    ints_mask = torch.ones(n_items)
    if reference_indicators:
        for i in range(0, n_items, n_indicators):
            ints_mask[i] = 0
            
    return ints_mask
        

def get_params_and_data(model_type:      str,
                        n_indicators:    int,
                        latent_size:     int,
                        cov_type:        str,
                        mean_type:       str,
                        sample_size:     int,
                        all_same_n_cats: bool = True,
                       ):
    """Simulate parameters and data for several types of latent factor models."""
    n_items = int(n_indicators * latent_size)
    
    cov_mat = get_covariance_matrix(latent_size, cov_type)
    mean, covariates, lreg_weight = get_mean(latent_size, mean_type, sample_size)
    ldgs = get_loadings(n_indicators, latent_size, cov_type == "free")
    ints_mask = get_ints_mask(n_indicators, latent_size, mean_type == "free")
    
    if model_type == "mixed":
        model_names = [k for k, _ in Simulators().SIMULATORS.items()]
        model_types = random.choices(model_names, k = n_items)
        unique_model_types = list(dict.fromkeys(model_types))
        item_idxs = [torch.Tensor([i for i, m in enumerate(model_types) if m == u]).long() for
                     u in unique_model_types]
    else:
        model_types = [model_type] * n_items
        unique_model_types = [model_type]
        item_idxs = [torch.arange(n_items)]
    
    sims = {}
    for i, u in enumerate(unique_model_types):
        _item_idxs = item_idxs[i]
        _n_items = len(_item_idxs)
        _ints_mask = ints_mask[_item_idxs]
        
        sim_kwargs = {"loadings" : ldgs[_item_idxs], "cov_mat" : cov_mat, "mean" : mean}
        if u in ("grm", "gpcm"):
            sim_kwargs["intercepts"] = get_categorical_intercepts(_n_items, all_same_n_cats)
        else:
            if u != "normal":
                sim_kwargs["loadings"].mul_(0.4)
                sim_kwargs["intercepts"] = pydist.Uniform(0.1, 0.5).sample([_n_items])
                if u == "negative_binomial":
                    sim_kwargs["probs"] = pydist.Uniform(0.5, 0.7).sample([_n_items])
                elif u == "lognormal":
                    sim_kwargs["residual_std"] = pydist.Uniform(1, 1.2).sample([_n_items])
            else:
                sim_kwargs["intercepts"] = torch.randn(_n_items).mul(0.1)
                sim_kwargs["residual_std"] = pydist.Uniform(0.6, 0.8).sample([_n_items])
                
        if len(sim_kwargs["intercepts"].shape) > 1:
            sim_kwargs["intercepts"].mul_(torch.cat((_ints_mask.unsqueeze(1),
                                                     torch.ones([_n_items, 
                                                                 sim_kwargs["intercepts"].shape[1] - 1])), dim = 1))
        else:
            sim_kwargs["intercepts"].mul_(_ints_mask)
        sims[u] = Simulators().SIMULATORS[u](**sim_kwargs)
    sim = MixedFactorModelSimulator(model_types, sims, cov_mat, mean)
        
    return {"Y" : sim.sample(sample_size),
            "loadings" : sim.loadings,
            "intercepts" : sim.intercepts,
            "cov_mat" : cov_mat,
            "mean" : mean,
            "covariates" : covariates,
            "latent_regression_weight" : lreg_weight,
            "model_type" : (model_type if model_type != "mixed" else model_types),
            "model_types" : model_types,
            "residual_std" : sim.residual_std,
            "probs" : sim.probs,
            "n_cats" : sim.n_cats,
            "ints_mask" : ints_mask,
           }
        
        
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