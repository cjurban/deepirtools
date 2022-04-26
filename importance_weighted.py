import torch
from torch import optim
import math
import timeit
from factor_analyzer import Rotator
from typing import List, Optional

from base import BaseEstimator
from models import *
from utils import tensor_dataset

EPS = 1e-7

  
class ImportanceWeightedEstimator(BaseEstimator):
    
    def __init__(self,
                 learning_rate:       float,
                 device:              str,
                 mirt_model:          str = "grm",
                 gradient_estimator:  str = "dreg",
                 log_interval:        int = 100,
                 verbose:             bool = True,
                 **model_kwargs,
                ):
        """
        Args:
        """
        super().__init__(device, log_interval, verbose)
        assert(gradient_estimator in ("iwae", "dreg"))
        self.grad_estimator = gradient_estimator
        
        self.runtime_kwargs["grad_estimator"] = self.grad_estimator
        
        assert(mirt_model in ("grm")) # TODO: Implement more mirt models
        if mirt_model == "grm":
            decoder = GradedResponseModel
        self.model = VariationalAutoencoder(decoder=decoder, device=device, **model_kwargs)
        self.optimizer = optim.Adam([{"params" : self.model.parameters()}],
                                    lr = learning_rate,
                                    amsgrad = True)
        self.timerecords = {}
                        
    def loss_function(self,
                      elbo: torch.Tensor,
                      x:    Optional[torch.Tensor] = None,
                     ):
        """Loss for one batch."""
        # ELBO over batch.
        if elbo.size(0) == 1:
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
                    x.register_hook(lambda grad: (w_tilda * grad).float())
            
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
    
#    @torch.no_grad()
#    def scores(self, # need to check this works, maybe make more efficient
#               data:         torch.Tensor,
#               missing_mask: Optional[torch.Tensor] = None,
#               mc_samples:   int = 1,
#               iw_samples:   int = 1,
#              ):
#        
#        loader =  torch.utils.data.DataLoader(
#                    tensor_dataset(data=data, mask=missing_mask),
#                    batch_size = 32, shuffle = True
#                  )
#        
#        scores_ls = []
#        for batch in loader:
#            batch = batch.to(self.device).float()
#            recon_y, mu, logstd, x = self.model(batch, mc_samples, iw_samples)
#
#            if iw_samples == 1:
#                scores_ls.append(mu.mean(1).squeeze())
#            else:
#                log_py_x, log_qx_y, log_px = self.loss_function(
#                                                   batch, recon_y, mu, logstd, x, mc_samples,
#                                                   iw_samples, return_components=True
#                                             )
#                elbo = -log_py_x - log_qx_y + log_px
#                reweight = (elbo - elbo.logsumexp(dim = 0)).exp()
#
#                iw_idxs = dist.Categorical(probs = reweight.T).sample().reshape(-1)
#                mc_idxs = torch.arange(mc_samples).repeat(data.size(0))
#                batch_idxs = torch.arange(data.size(0)).repeat_interleave(mc_samples)
#                scores_ls.append(x[iw_idxs, mc_idxs, batch_idxs, ...].reshape(data.size(0), mc_samples, self.latent_size).mean(-2))                  
#        return torch.cat(scores_ls, dim = 0)
        
    @property
    def loadings(self): # need to check this exists
        return self.model.decoder.loadings.weight.data # need to define this property in decoder?
    
    @property
    def intercepts(self): # need to check this exists
        return self.model.decoder.intercepts.bias.data