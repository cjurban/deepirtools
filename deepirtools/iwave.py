import torch
from torch.optim import Adam
from torch.distributions.utils import logits_to_probs
import math
import timeit
from typing import List, Optional, Union
from deepirtools.base import BaseEstimator
from deepirtools.models import (ModelTypes,
                                MixedFactorModel,
                                VariationalAutoencoder,
                               )
from deepirtools.utils import tensor_dataset
from deepirtools.settings import GRAD_ESTIMATORS

  
class IWAVE(BaseEstimator):
    
    def __init__(self,
                 model_type:          Union[str, List[str]],
                 learning_rate:       float = 1e-3,
                 device:              str = "cpu",
                 gradient_estimator:  str = "dreg",
                 log_interval:        int = 100,
                 verbose:             bool = True,
                 **model_kwargs,
                ):
        """
        Importance-weighted amortized variational estimator (I-WAVE).
        
        Args:
            model_type         (str/List of str): Measurement model type. Can either be a string if all items
                                                  have same type or a list of strings specifying each item type.
                                                  Current options are:
                                                      "grm"               = graded response model
                                                      "gpcm"              = generalized partial credit model
                                                      "poisson"           = poisson factor model
                                                      "negative_binomial" = negative binomial factor model
                                                      "normal"            = normal factor model
                                                      "lognormal"         = lognormal factor model
            learning_rate      (float):           Step size for stochastic gradient optimizer.
            device             (str):             Computing device used for fitting.
            gradient_estimator (str):             Gradient estimator for inference model parameters:
                                                      "dreg" = doubly reparameterized gradient estimator
                                                      "iwae" = standard gradient estimator
            log_interval       (str):             Frequency of updates printed during fitting.
            verbose            (bool):            Whether to print updates during fitting.
            model_kwargs       (dict):            Named parameters passed to VariationalAutoencoder.__init__().
        """
        super().__init__(device, log_interval, verbose)
        assert(gradient_estimator in GRAD_ESTIMATORS), "gradient_estimator must be one of {}".format(GRAD_ESTIMATORS)
        self.grad_estimator = gradient_estimator
        self.runtime_kwargs["grad_estimator"] = gradient_estimator
        
        model_names = [k for k, _ in ModelTypes().MODEL_TYPES.items()]
        if isinstance(model_type, list):
            assert(all(m in model_names for m in model_type)), "All elements of model_type must be one of {}".format(model_names)
            model_kwargs["model_types"] = model_type
            decoder = MixedFactorModel
        else:
            assert(model_type in model_names), "model_type must be one of {}".format(model_names)
            decoder = ModelTypes().MODEL_TYPES[model_type]
        
        if verbose:
            print("\nInitializing model parameters", end = "")
        start = timeit.default_timer()
        self.model = VariationalAutoencoder(decoder=decoder, **model_kwargs).to(device)
        stop = timeit.default_timer()
        if verbose:
            print("\nInitialization ended in ", round(stop - start, 2), " seconds", end = "\n")
        
        self.optimizer = Adam([{"params" : self.model.parameters()}],
                                lr = learning_rate, amsgrad = True)
        
        self.timerecords = {}
        self.timerecords["init"] = stop - start
                        
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
                       covariates:   Optional[torch.Tensor] = None,
                       mc_samples:   int = 1,
                       iw_samples:   int = 5000,
                      ):
        """Log-likelihood for a data set."""
        loader =  torch.utils.data.DataLoader(
                    tensor_dataset(data = data, mask = missing_mask,
                                   covariates = covariates),
                    batch_size = 32, shuffle = True,
                    pin_memory = self.device == "cuda",
                  )
        
        old_estimator = self.grad_estimator
        self.grad_estimator = "iwae"
        
        if self.verbose:
            print("\nComputing approx. LL", end="")
        
        start = timeit.default_timer()
        ll = -self.test(loader, mc_samples = mc_samples, iw_samples = iw_samples)
        stop = timeit.default_timer()
        self.timerecords["log_likelihood"] = stop - start
        
        if self.verbose:
            print("\nApprox. LL computed in", round(stop - start, 2), "seconds\n", end = "")
        
        self.grad_estimator = old_estimator

        return ll
    
    @torch.no_grad()
    def scores(self,
               data:         torch.Tensor,
               missing_mask: Optional[torch.Tensor] = None,
               covariates:   Optional[torch.Tensor] = None,
               mc_samples:   int = 1,
               iw_samples:   int = 5000,
              ):
        
        loader = torch.utils.data.DataLoader(
                    tensor_dataset(data = data, mask = missing_mask,
                                   covariates = covariates),
                    batch_size = 32, shuffle = True,
                    pin_memory = self.device == "cuda",
                  )
        
        scores = []
        for batch in loader:
            batch = {k : v.to(self.device).float() if v is not None else v for k, v in batch.items()}
            
            elbo, x = self.model(**batch, mc_samples = mc_samples, iw_samples = iw_samples,
                                 **self.runtime_kwargs)
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
        except TypeError:
            return None
    
    @property
    def probs(self):
        try:
            logits_to_probs(self.model.decoder.logits.data.cpu(), is_binary = True)
        except AttributeError:
            return None
        except TypeError:
            return None
    
    @property
    def cov(self):
        try:
            return self.model.cholesky.cov.data.cpu()
        except AttributeError:
            return None
        
    @property
    def mean(self):
        try:
            return self.model.mean.data.cpu()
        except AttributeError:
            return None
        
    @property
    def latent_regression_weight(self):
        try:
            return self.model.lreg_weight.data.cpu()
        except AttributeError:
            return None