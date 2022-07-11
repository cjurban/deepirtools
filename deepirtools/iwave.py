import torch
from torch.optim import Adam
import math
import timeit
from typing import List, Optional
from deepirtools.base import BaseEstimator
from deepirtools.models import *
from deepirtools.utils import tensor_dataset

  
class IWAVE(BaseEstimator):
    
    def __init__(self,
                 learning_rate:       float,
                 device:              str,
                 model_type:          str,
                 gradient_estimator:  str = "dreg",
                 log_interval:        int = 100,
                 verbose:             bool = True,
                 **model_kwargs,
                ):
        """
        Importance-weighted amortized variational estimator (I-WAVE).
        
        Args:
            learning_rate      (float): Step size for stochastic gradient optimizer.
            device             (str):   Computing device used for fitting.
            model_type         (str):   Measurement model type. Current options are:
                                            "grm"       = graded response model
                                            "gpcm"      = generalized partial credit model
                                            "normal"    = normal factor model
                                            "lognormal" = log-normal factor model
            gradient_estimator (str):   Gradient estimator for inference model parameters:
                                            "dreg" = doubly reparameterized gradient estimator
                                            "iwae" = standard gradient estimator
            log_interval       (str):   Frequency of updates printed during fitting.
            verbose            (bool):  Whether to print updates during fitting.
            model_kwargs       (dict):  Named parameters passed to VariationalAutoencoder.__init__().
        """
        super().__init__(device, log_interval, verbose)
        grad_estimators = ("iwae", "dreg")
        assert(gradient_estimator in grad_estimators), "gradient_estimator must be one of {}".format(grad_estimators)
        self.grad_estimator = gradient_estimator
        
        self.runtime_kwargs["grad_estimator"] = self.grad_estimator
        
        model_types = ("grm", "gpcm", "poisson", "negative_binomial", "normal", "lognormal")
        assert(model_type in model_types), "model_type must be one of {}".format(model_types)
        if model_type == "grm":
            decoder = GradedResponseModel
        elif model_type == "gpcm":
            decoder = GeneralizedPartialCreditModel
        elif model_type == "poisson":
            decoder = PoissonFactorModel
        elif model_type == "negative_binomial":
            decoder = NegativeBinomialFactorModel
        elif model_type == "normal":
            decoder = NormalFactorModel
        elif model_type == "lognormal":
            decoder = LogNormalFactorModel
        self.model = VariationalAutoencoder(decoder=decoder, **model_kwargs).to(device)
        self.optimizer = Adam([{"params" : self.model.parameters()}],
                                lr = learning_rate, amsgrad = True)
        self.timerecords = {}
                        
    def loss_function(self,
                      elbo: torch.Tensor,
                      x:    torch.Tensor,
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
        """Log-likelihood for a data set."""
        loader =  torch.utils.data.DataLoader(
                    tensor_dataset(data = data, mask = missing_mask),
                    batch_size = 32, shuffle = True,
                    pin_memory = self.device == "cuda",
                  )
        
        old_estimator = self.grad_estimator
        self.grad_estimator = "iwae"
        
        print("\nComputing approx. LL", end="")
        
        start = timeit.default_timer()
        ll = -self.test(loader, mc_samples = mc_samples, iw_samples = iw_samples)
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
               iw_samples:   int = 5000,
              ):
        
        loader = torch.utils.data.DataLoader(
                    tensor_dataset(data = data, mask = missing_mask),
                    batch_size = 32, shuffle = True,
                    pin_memory = self.device == "cuda",
                  )
        
        scores = []
        for batch in loader:
            if isinstance(batch, list):
                batch, mask = batch[0], batch[1]
                mask = mask.to(self.device).float()
            else:
                mask = None
            batch =  batch.to(self.device).float()
            batch_size = batch.size(0)
            
            elbo, x = self.model(batch, mask = mask, mc_samples = mc_samples,
                                 iw_samples = iw_samples, **self.runtime_kwargs)
            w_tilda = (elbo - elbo.logsumexp(dim = 0)).exp()
            latent_size = x.size(-1)

            idxs = torch.distributions.Categorical(probs = w_tilda.permute([1, 2, 3, 0])).sample()
            idxs = idxs.expand(x[-1, ...].shape).unsqueeze(0).long()
            scores.append(torch.gather(x, axis = 0, index = idxs).squeeze(0).mean(dim = 0))                  
        return torch.cat(scores, dim = 0).cpu()
        
    @property
    def loadings(self):
        try:
            return self.model.decoder.loadings.cpu()
        except AttributeError:
            return None
    
    @property
    def intercepts(self):
        try:
            return self.model.decoder.intercepts.cpu()
        except AttributeError:
            return None
        
    @property
    def residual_std(self):
        try:
            return self.model.decoder.residual_std.data.cpu()
        except AttributeError:
            return None
    
    @property
    def cov(self):
        try:
            return self.model.cholesky.cov.data.cpu()
        except AttributeError:
            return None