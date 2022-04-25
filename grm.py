import torch
from torch import optim
import torch.distributions as dist
import math
import timeit
from factor_analyzer import Rotator
from typing import List, Optional

from base import BaseEstimator
from models import GRMVAE
from utils import tensor_dataset

EPS = 1e-7

  
class GRMEstimator(BaseEstimator):
    
    def __init__(self,
                 input_size:          int,
                 inference_net_sizes: List[int],
                 latent_size:         int,
                 n_cats:              List[int],
                 learning_rate:       float,
                 device:              str,
                 gradient_estimator:  str = "dreg",
                 Q:                   Optional[torch.Tensor] = None,
                 A:                   Optional[torch.Tensor] = None,
                 b:                   Optional[torch.Tensor] = None,
                 correlated_factors:  List[int] = [],
                 log_interval:        int = 100,
                 verbose:             bool = True,
                ):
        """
        Args:
        """
        super().__init__(input_size, inference_net_sizes, latent_size, learning_rate,
                         device, log_interval, verbose)
        assert(gradient_estimator == "iwae" or gradient_estimator == "dreg")
        self.grad_estimator = gradient_estimator
        
        self.model = GRMVAE(input_size = input_size,
                            inference_net_sizes = inference_net_sizes,
                            latent_size = latent_size,
                            n_cats = n_cats,
                            Q = Q,
                            A = A,
                            b = b,
                            device = device,
                            correlated_factors = correlated_factors,
                           ).to(device)
        self.n_cats = n_cats
        self.Q = Q
        self.A = A
        self.b = b
        self.correlated_factors = correlated_factors
        
        self.optimizer = optim.Adam([{"params" : self.model.parameters()}],
                                    lr = learning_rate,
                                    amsgrad = True)
        self.timerecords = {}
                        
    def loss_function(self,
                      y:                 torch.Tensor,
                      recon_y:           torch.Tensor,
                      mu:                torch.Tensor,
                      std:               torch.Tensor,
                      x:                 torch.Tensor,
                      mc_samples:        int,
                      iw_samples:        int,
                      return_components: bool = False,
                     ):
        """Loss for one batch."""
        # Log p(y | x).        
        idxs = y.long().expand(recon_y[..., -1].shape).unsqueeze(-1)
        log_py_x = -(torch.gather(recon_y, dim = -1, index = idxs).squeeze(-1)).clamp(min = EPS).log().sum(dim = -1, keepdim = True)
        
        # Log p(x).
        if self.correlated_factors != []:
            log_pz = dist.MultivariateNormal(torch.zeros_like(x).to(self.device),
                                             scale_tril = self.model.cholesky.weight).log_prob(x).unsqueeze(-1)
        else:
            log_px = dist.Normal(torch.zeros_like(x).to(self.device),
                                 torch.ones_like(x).to(self.device)).log_prob(x).sum(-1, keepdim = True)
            
        # Log q(x | y).
        if self.model.training and iw_samples > 1 and self.grad_estimator == "dreg":
            mu_, std_ = mu.detach(), std.detach()
            qx_y = dist.Normal(mu_, std_)
        else:
            qx_y = dist.Normal(mu, std)
        log_qx_y = qx_y.log_prob(x).sum(dim = -1, keepdim = True)
        
        if return_components:
            return log_py_x, log_qx_y, log_px
        
        elbo = log_py_x + log_qx_y - log_px
        
        # ELBO over batch.
        if iw_samples == 1:
            elbo = elbo.squeeze(0).mean(0)
            if self.model.training:
                return elbo.mean()
            else:
                return elbo.sum()

        # IW-ELBO over batch.
        elif self.grad_estimator == "iwae":
            elbo *= -1
            iw_elbo = math.log(elbo.size(0)) - elbo.logsumexp(dim = 0)
                    
            if self.model.training:
                return iw_elbo.mean()
            else:
                return iw_elbo.mean(0).sum()

        # IW-ELBO with DReG estimator over batch.
        elif self.grad_estimator == "dreg":
            elbo *= -1
            with torch.no_grad():
                w_tilda = (elbo - elbo.logsumexp(dim = 0)).exp()
                
                if x.requires_grad:
                    x.register_hook(lambda grad: w_tilda * grad)
            
            if self.model.training:
                return (-w_tilda * elbo).sum(0).mean()
            else:
                return (-w_tilda * elbo).sum()
       
    @torch.no_grad()
    def log_likelihood(self,
                       data:         torch.Tensor,
                       missing_mask: Optional[torch.Tensor] = None,
                       mc_samples:   int = 1,
                       iw_samples:   int = 5000,
                      ):
        loader =  torch.utils.data.DataLoader(
                    tensor_dataset(data=data, mask=missing_mask),
                    batch_size = 32, shuffle = True
                  )
        
        old_estimator = self.grad_estimator
        self.grad_estimator = "iwae"
        
        print("\nComputing approx. LL", end="")
        
        start = timeit.default_timer()
        ll = self.test(loader, mc_samples =  mc_samples, iw_samples = iw_samples)
        stop = timeit.default_timer()
        self.timerecords["log_likelihood"] = stop - start
        print("\nApprox. LL computed in", round(stop - start, 2), "seconds\n", end = "")
        
        self.grad_estimator = old_estimator

        return ll
    
    @torch.no_grad()
    def scores(self,
               data:         torch.Tensor,
               missing_mask: Optional[torch.Tensor] = None,
               mc_samples:   int = 1,
               iw_samples:   int = 1,
              ):
        
        loader =  torch.utils.data.DataLoader(
                    tensor_dataset(data=data, mask=missing_mask),
                    batch_size = 32, shuffle = True
                  )
        
        scores_ls = []
        for batch in loader:
            batch = batch.to(self.device).float()
            recon_y, mu, logstd, x = self.model(batch, mc_samples, iw_samples)

            if iw_samples == 1:
                scores_ls.append(mu.mean(1).squeeze())
            else:
                log_py_x, log_qx_y, log_px = self.loss_function(
                                                   batch, recon_y, mu, logstd, x, mc_samples,
                                                   iw_samples, return_components=True
                                             )
                elbo = -log_py_x - log_qx_y + log_px
                reweight = (elbo - elbo.logsumexp(dim = 0)).exp()

                iw_idxs = dist.Categorical(probs = reweight.T).sample().reshape(-1)
                mc_idxs = torch.arange(mc_samples).repeat(data.size(0))
                batch_idxs = torch.arange(data.size(0)).repeat_interleave(mc_samples)
                scores_ls.append(x[iw_idxs, mc_idxs, batch_idxs, ...].reshape(data.size(0), mc_samples, self.latent_size).mean(-2))                  
        return torch.cat(scores_ls, dim = 0)
    
    def rotate_loadings(self,
                        method:        str, 
                        loadings_only: bool = True):
        loadings = self.loadings.numpy()
        rotator = Rotator(method = method)
        
        start = timeit.default_timer()
        rot_loadings = rotator.fit_transform(loadings)
        stop = timeit.default_timer()
        self.timerecords["rotation"] = stop - start

        if loadings_only: 
            return rot_loadings
        else:
            return rot_loadings, rotator.phi_, rotator.rotation_
        
    @property
    def loadings(self):
        return self.model.projector.loadings.weight.data # need to define this property in projector?
    
    @property
    def intercepts(self):
        return self.model.projector.intercepts.bias.data