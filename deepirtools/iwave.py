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
    r"""Importance-weighted amortized variational estimator (I-WAVE).

    Attributes
    __________
    loadings : Tensor
        Factor loadings matrix.

        A :math:`J \times D` matrix where :math:`J` is the number of items and :math:`D`
        is the latent dimension.
    intercepts : Tensor
        Intercepts.

        When all items are continuous, a length :math:`J` vector where :math:`J`
        is the number of items. When some items are graded (i.e., ordinal), a :math:`J \times M`
        matrix where :math:`M` is the maximum number of response categories across all items.
    residual_std : Tensor or None
        Residual standard deviations.

        A length :math:`J` vector where :math:`J` is the number of items. Only applicable to
        normal and lognormal factor models.
    probs : Tensor or None
        Success probabilities for Bernoulli trials.

        A length :math:`J` vector where :math:`J` is the number of items. Only applicable to
        negative binomial factor models.
    cov : Tensor
        Factor covariance matrix.

        A :math:`D \times D` matrix where :math:`D` is the latent dimension.
    mean : Tensor
        Factor mean vector.

        A length :math:`D` vector where :math:`D` is the latent dimension.
    latent_regression_weight : Tensor or None
        Latent regression weight matrix.

        A :math:`D \times C` matrix where :math:`D` is the latent dimension and :math:`C`
        is the number of covariates. Only applicable to latent regression models.
    grad_estimator : str
        Gradient estimator for inference model parameters.
    device : str
        Computing device used for fitting.
    verbose : bool
        Whether to print updates during fitting.
    global_iter : int
        Number of mini-batches processed during fitting.
    timerecords : dict
        Stores run times for various processes (e.g., fitting).
    """
    
    def __init__(self,
                 model_type:          Union[str, List[str]],
                 learning_rate:       float = 0.001,
                 device:              str = "cpu",
                 gradient_estimator:  str = "dreg",
                 log_interval:        int = 100,
                 verbose:             bool = True,
                 n_intervals:         int = 100,
                 **model_kwargs,
                ):
        r"""Initialize I-WAVE.
        
        Parameters
        __________
        model_type : str or list of str)
            Measurement model type.

            Can either be a string if all items have same type or a list of strings specifying each
            item type. Current options are:

            * "grm", graded response model;
            * "gpcm", generalized partial credit model;
            * "poisson", poisson factor model;
            * "negative_binomial", negative binomial factor model;
            * "normal", normal factor model; and
            * "lognormal", lognormal factor model.
        latent_size : int
            Number of latent factors.
        n_cats : list of int and None, optional
            Number of response categories for each item.

            Only needed if some items are categorical. Any continuous items or counts are indicated
            with None.

            For example, setting ``n_cats = [3, 3, None, 2]`` indicates that items 1--2 are categorical
            with 3 categories, item 3 is continuous, and item 4 is categorical with 2 categories.
        n_items : int, optional
            Number of items.

            Only specified if all items are continuous. Not needed if n_cats is specified instead.
        inference_net_sizes : list of int, default = [100]
            Neural network inference model hidden layer dimensions.

            For example, setting ``inference_net_sizes = [100, 100]`` creates a neural network
            inference model with two hidden layers of size 100.
        fixed_variances : bool, default = True
            Whether to constrain variances of latent factors to one.
        fixed_means : bool, default = True
            Whether to constrain means of latent factors to zero.
        correlated_factors : list of int, default = []
            Which latent factors should be correlated.

            For example, setting ``correlated_factors = [0, 3, 4]`` in a model with 5 latent
            factors models the correlations between the first, fourth, and fifth factors
            while constraining the other correlations to zero.
        covariate_size : int, default = None
            Number of covariates for latent regression.
        Q : Tensor, default = None
            Binary matrix indicating measurement structure.

            A :math:`J \times D` matrix where :math:`J` is the number of items and :math:`D`
            is the latent dimension. Elements of :math:`\mathbf{Q}` are zero if the corresponding
            loading is set to zero and one otherwise:
            
            .. math::
               \beta_{j,d} = q_{j,d} \beta_{j,d}',
            
            where :math:`\beta_{j,d}` is the loading for item :math:`j` on factor :math:`d`,
            :math:`q_{j,d} \in \{0, 1\}` is an element of :math:`\mathbf{Q}`, and :math:`\beta_{j,d}'`
            is an unconstrained loading.
        A : Tensor, default = None
            Matrix imposing linear constraints on loadings.

            Linear constraints are imposed as follows:

            .. math::
               \boldsymbol{\beta} = \boldsymbol{b} + \boldsymbol{A} \boldsymbol{\beta}',

            where :math:`\boldsymbol{\beta} = (\beta_{1, 1}, \ldots, \beta_{J, 1}, \ldots,
            \beta_{1, D}, \ldots, \beta_{J, D})^\top` is a :math:`DJ \times 1` vector of
            constrained loadings values, :math:`\boldsymbol{b}` is a :math:`DJ \times 1`
            vector of constants, :math:`\boldsymbol{A}` is a :math:`DJ \times DJ` matrix
            of constants, and :math:`\boldsymbol{\beta}' = (\beta_{1, 1}', \ldots, \beta_{J, 1}',
            \ldots, \beta_{1, D}', \ldots, \beta_{J, D}')^\top` is a :math:`DJ \times ` vector of
            unconstrained loadings.
        b : Tensor, default = None
            Vector imposing linear constraints on loadings.

            See above for elaboration on linear constraints.
        ints_mask : Tensor, default = None
            Vector constraining specific intercepts to zero.

            A length :math:`J` vector where :math:`J` is the number of items. For categorical
            items, only the smallest category intercept is constrained to zero.
        learning_rate : float, default = 0.001
            Step size for stochastic gradient optimizer.

            This is the main hyperparameter that may require tuning. Decreasing it typically
            improves optimization stability at the cost of increased fitting time.
        device : str, default = "cpu"
            Computing device used for fitting.

            Current options are:

            * "cpu", central processing unit; and
            * "cuda", graphics processing unit.
        gradient_estimator : str, default = "dreg"
            Gradient estimator for inference model parameters.

            Current options are:

            * "dreg", doubly reparameterized gradient estimator; and
            * "iwae", standard gradient estimator.

            "dreg" is the recommended option due to its bounded variance as the number of
            importance-weighted samples tends to infinity.
        log_interval : str, default = 100
            Number of mini-batches between printed updates during fitting.
        verbose : bool, default = True
            Whether to print updates during fitting.
        n_intervals : str, default = 100
            Number of 100-mini-batch intervals after which fitting is terminated if best average
            loss does not improve.
        use_spline_prior : bool, default = False
            Whether to use spline/spline coupling prior.
        count_bins : int, optional
            Number of segments for each spline transformation.
        bound : float, optional
            Quantity determining the bounding box of each spline transformation.
        """
        
        super().__init__(device, log_interval, verbose, n_intervals)
        assert(gradient_estimator in GRAD_ESTIMATORS), ("gradient_estimator ",
                                                        "must be one of {}".format(GRAD_ESTIMATORS))
        self.grad_estimator = gradient_estimator
        self.runtime_kwargs["grad_estimator"] = gradient_estimator
        
        model_names = [k for k, _ in ModelTypes().MODEL_TYPES.items()]
        if isinstance(model_type, list):
            assert(all(m in model_names for m in model_type)), ("All elements ",
                                                                "of model_type must be one of {}".format(model_names))
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
        r"""Approximate log-likelihood of a data set.
        
        Parameters
        __________
        data : Tensor
            Data set.

            An :math:`N \times J` matrix where :math:`N` is the number of people and :math:`J`
            is the number of items.
        missing_mask : Tensor, default = None
            Binary mask indicating missing item responses.

            An :math:`N \times J` matrix where :math:`N` is the number of people and :math:`J`
            is the number of items.
        covariates : Tensor, default = None
            Matrix of covariates.

            An :math:`N \times C` matrix where :math:`N` is the number of people and :math:`C`
            is the number of covariates.
        mc_samples : int, default = 1
            Number of Monte Carlo samples.

            Increasing this decreases the log-likelihood estimator's variance.
        iw_samples : int, default = 5000
            Number of importance-weighted samples.

            Increasing this decreases the log-likelihood estimator's bias.
        
        Returns
        _______
        ll : float
            Approximate log-likelihood of the data set.
        """
        
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
        r"""Approximate expected a posteriori (EAP) factor scores given a data set.
        
        Parameters
        __________
        data : Tensor
            Data set.

            An :math:`N \times J` matrix where :math:`N` is the number of people and :math:`J`
            is the number of items.
        missing_mask : Tensor, default = None
            Binary mask indicating missing item responses.

            An :math:`N \times J` matrix where :math:`N` is the number of people and :math:`J`
            is the number of items.
        covariates : Tensor, default = None
            Matrix of covariates.

            An :math:`N \times C` matrix where :math:`N` is the number of people and :math:`C`
            is the number of covariates.
        mc_samples : int, default = 1
            Number of Monte Carlo samples.

            Increasing this decreases the EAP estimator's variance.
        iw_samples : int, default = 5000
            Number of importance-weighted samples.

            Increasing this decreases the EAP estimator's bias. When ``iw_samples > 1``, samples
            are drawn from the expected importance-weighted distribution using sampling-
            importance-resampling. # TODO : Include reference.
        
        Returns
        _______
        factor_scores : Tensor
            Approximate EAP factor scores given the data set.
            
            An :math:`N \times D` matrix where :math:`N` is the number of people and :math:`D`
            is the latent dimension.
        """
        
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
            return self.model.mean[0, :].data.cpu()
        except AttributeError:
            return None
        
    @property
    def latent_regression_weight(self):
        try:
            return self.model.lreg_weight.data.cpu()
        except AttributeError:
            return None