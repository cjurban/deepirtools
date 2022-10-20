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
    r"""Importance-weighted amortized variational estimator (I-WAVE). [1]_ [2]_
    
    Parameters
    __________
    model_type : str or list of str)
        Measurement model type.

        Can either be a string if all items have same type or a list of strings specifying each
        item type. 
        
        Let :math:`y_j` be the response to item :math:`j`, :math:`\boldsymbol{x}` be a
        :math:`\text{latent_size} \times 1` vector of latent factors, and
        :math:`\boldsymbol{\beta}_j` be a :math:`\text{latent_size} \times 1`
        vector of factor loadings for item :math:`j`. Current measurement model
        options are:

        * \"grm\", graded response model:
        
          .. math::
              \begin{split}
                  &\text{Pr}(y_j = k \mid \mathbf{x}) \\
                  &=
                  \begin{cases}
                      1 - \sigma(\alpha_{j,k + 1} + \boldsymbol{\beta}_j^\top \mathbf{x}), & \text{if $k = 0$} \\
                      \sigma(\alpha_{j,k} + \boldsymbol{\beta}_j^\top \mathbf{x}) - \sigma(\alpha_{j,k+1} +
                      \boldsymbol{\beta}_j^\top \mathbf{x}), & \text{if $k \in \{1, \ldots, K_j - 2\}$},\\
                      \sigma(\alpha_{j,k} + \boldsymbol{\beta}_j^\top \mathbf{x}), & \text{if $k = K_j - 1$},
                  \end{cases}
              \end{split}
            
          where :math:`\alpha_{j, k}` is the :math:`k^\text{th}` category intercept for item
          :math:`j`, :math:`K_j` is the number of response categories for item :math:`j`,
          and :math:`\sigma(z) = 1 / (1 + \exp[-z])` is the inverse logistic link function.
        
        * \"gpcm\", generalized partial credit model:
        
          .. math::
              \text{Pr}(y_j = k - 1 \mid \boldsymbol{x}) = \frac{\exp\big[(k - 1)\boldsymbol{\beta}_j^\top
              \boldsymbol{x} + \sum_{\ell = 1}^k \alpha_{j, \ell} \big]}{\sum_{m = 1}^{K_j} \exp \big[ (m - 1)
              \boldsymbol{\beta}_j^\top\boldsymbol{x} + \sum_{\ell = 1}^m \alpha_{j, \ell} \big]},
            
          where :math:`k = 1, \ldots, K_j` and :math:`\alpha_{j, k}` is the :math:`k^\text{th}`
          category intercept for item :math:`j`.
          
        * \"nominal\", nominal response model:
        
          .. math::
              \text{Pr}(y_j = k \mid \boldsymbol{x}) = \frac{\exp[\mathbb{1}(k \neq 0) (\alpha_{j, k} + \boldsymbol{\beta}_j^\top
              \boldsymbol{x}) ]}{1 + \sum_{\ell = 1}^{K_j} \exp [ \alpha_{j, \ell} + \boldsymbol{\beta}_j^\top
              \boldsymbol{x} ]},
            
          where :math:`k = 0, \ldots, K_j - 1`, :math:`\mathbb{1}(\cdot)` is the indicator function, and :math:`\alpha_{j, k}` is the :math:`k^\text{th}`
          category intercept for item :math:`j`.
        
        * \"poisson\", poisson factor model:
        
          .. math::
              y_j \mid \boldsymbol{x} \sim \text{Pois}(\exp[\boldsymbol{\beta}_j^\top\boldsymbol{x}
              + \alpha_j]),
            
          where :math:`y_j \in \{ 0, 1, 2, \ldots \}` and :math:`\alpha_j` is the intercept
          for item :math:`j`.
        
        * \"negative_binomial\", negative binomial factor model:
        
          .. math::
              y_j \mid \boldsymbol{x} \sim \text{NB}(\exp[\boldsymbol{\beta}_j^\top\boldsymbol{x}
              + \alpha_j], p),
            
          where :math:`y_j \in \{ 0, 1, 2, \ldots \}`,  :math:`\alpha_j` is the intercept
          for item :math:`j`, and :math:`p` is a success probability.
        
        * \"normal\", normal factor model:
        
          .. math::
              y_j \mid \boldsymbol{x} \sim \mathcal{N}(\boldsymbol{\beta}_j^\top\boldsymbol{x}
              + \alpha_j, \sigma_j^2),
            
          where :math:`y_j \in (-\infty, \infty)`,  :math:`\alpha_j` is the intercept
          for item :math:`j`, and :math:`\sigma_j^2` is the residual variance for item :math:`j`.
        
        * \"lognormal\", lognormal factor model:
        
          .. math::
              \ln y_j \mid \boldsymbol{x} \sim \mathcal{N}(\boldsymbol{\beta}_j^\top\boldsymbol{x}
              + \alpha_j, \sigma_j^2),
            
          where :math:`y_j > 0` and :math:`\alpha_j` is the intercept for item :math:`j`.
    learning_rate : float, default = 0.001
        Step size for stochastic gradient optimizer.

        This is the main hyperparameter that may require tuning. Decreasing it typically
        improves optimization stability at the cost of increased fitting time.
    device : str, default = "cpu"
        Computing device used for fitting.

        Current options are:

        * \"cpu\", central processing unit; and
        * \"cuda\", graphics processing unit.
        
    gradient_estimator : str, default = "dreg"
        Gradient estimator for inference model parameters.

        Current options are:

        * \"dreg\", doubly reparameterized gradient estimator; and
        * \"iwae\", standard gradient estimator.

        \"dreg\" is the recommended option due to its bounded variance as the number of
        importance-weighted samples tends to infinity.
    log_interval : str, default = 100
        Number of mini-batches between printed updates during fitting.
    verbose : bool, default = True
        Whether to print updates during fitting.
    n_intervals : str, default = 100
        Number of 100-mini-batch intervals after which fitting is terminated if best average
        loss does not improve.
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
        
        Only applicable when ``use_spline_prior = False``.
    fixed_means : bool, default = True
        Whether to constrain means of latent factors to zero.
        
        Only applicable when ``use_spline_prior = False``.
    correlated_factors : list of int, default = []
        Which latent factors should be correlated.
        
        Only applicable when ``use_spline_prior = False``.

        For example, setting ``correlated_factors = [0, 3, 4]`` in a model with 5 latent
        factors models the correlations between the first, fourth, and fifth factors
        while constraining the other correlations to zero.
    covariate_size : int, default = None
        Number of covariates for latent regression.
        
        Setting ``covariate_size > 0`` when ``use_spline_prior = False`` models the
        distribution of the latent factors as:[3]_ [4]_
        
        .. math::
            \boldsymbol{x} \mid \boldsymbol{z} \sim \mathcal{N}(\boldsymbol{\Gamma}^\top
            \boldsymbol{z}, \boldsymbol{\Sigma}),

        where :math:`\boldsymbol{\Gamma}` is a :math:`\text{latent_size}
        \times \text{covariate_size}` matrix of regression weights,
        :math:`\boldsymbol{z}` is a :math:`\text{covariate_size} \times
        1` vector of covariates, and :math:`\boldsymbol{\Sigma}` is a
        :math:`\text{latent_size} \times \text{latent_size}` factor covariance matrix.
    Q : Tensor, default = None
        Binary matrix indicating measurement structure.

        A :math:`\text{n_items} \times \text{latent_size}` matrix.
        Elements of :math:`\mathbf{Q}` are zero if the corresponding
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

        where :math:`\boldsymbol{\beta} = (\beta_{1, 1}, \ldots, \beta_{\text{n_items}, 1},
        \ldots, \beta_{1, \text{latent_size}}, \ldots, \beta_{\text{n_items},
        \text{latent_size}})^\top` is a :math:`(\text{latent_size} \cdot \text{n_items})
        \times 1` vector of constrained loadings values, :math:`\boldsymbol{b}` is a
        :math:`(\text{latent_size} \cdot \text{n_items}) \times 1`
        vector of constants, :math:`\boldsymbol{A}` is a :math:`(\text{latent_size} \cdot
        \text{n_items}) \times (\text{latent_size} \cdot \text{n_items})` matrix
        of constants, and :math:`\boldsymbol{\beta}' = (\beta_{1, 1}', \ldots,
        \beta_{\text{n_items}, 1}', \ldots, \beta_{1, \text{latent_size}}',
        \ldots, \beta_{\text{n_items}, \text{latent_size}}')^\top` is a
        :math:`(\text{latent_size} \cdot \text{n_items}) \times 1` vector of
        unconstrained loadings.
    b : Tensor, default = None
        Vector imposing linear constraints on loadings.

        See above for elaboration on linear constraints.
    ints_mask : Tensor, default = None
        Binary vector constraining specific intercepts to zero.

        A length :math:`\text{n_items}` vector. For categorical items, only
        the smallest category intercept is constrained to zero.
    use_spline_prior : bool, default = False
        Whether to use spline/spline coupling prior distribution for the
        latent factors. [5]_ [6]_
    spline_init_normal : bool, default = True
        Whether to initialize spline/spline coupling prior close to a standard normal
        distribution.
    flow_length : int, default = 2
        Number of spline coupling flow layers.
        
        Only applicable when ``latent_size > 1``.
    count_bins : int, optional
        Number of segments for each spline transformation.
    bound : float, optional
        Quantity determining the bounding box of each spline transformation.
    spline_net_sizes : list of int, optional
        Hidden layer dimensions for neural network mapping from covariates to base density
        for spline/spline coupling prior.
        
        Corresponds to a more flexible version of the latent regression model.

    Attributes
    __________
    loadings : Tensor
        Factor loadings matrix.

        A :math:`\text{n_items} \times \text{latent_size}` matrix.
    intercepts : Tensor
        Intercepts.

        When all items are continuous, a length :math:`\text{n_items}` vector. When some
        items are graded (i.e., ordinal), a :math:`\text{n_items} \times M`
        matrix where :math:`M` is the maximum number of response categories across all items.
    residual_std : Tensor or None
        Residual standard deviations.

        A length :math:`\text{n_items}` vector. Only applicable to normal and
        lognormal factor models.
    probs : Tensor or None
        Success probabilities for Bernoulli trials.

        A length :math:`\text{n_items}` vector. Only applicable to negative
        binomial factor models.
    cov : Tensor or None
        Factor covariance matrix.

        A :math:`\text{latent_size} \times \text{latent_size}` matrix. Only
        applicable when ``use_spline_prior = False``.
    mean : Tensor or None
        Factor mean vector.

        A length :math:`\text{latent_size}` vector. Only applicable when
        ``use_spline_prior = False``.
    latent_regression_weight : Tensor or None
        Latent regression weight matrix.

        A :math:`\text{latent_size} \times \text{covariate_size}`. Only
        applicable to latent regression models.
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
        
    References
    ----------
    .. [1] Urban, C. J., & Bauer, D. J. (2021). A deep learning algorithm for high-dimensional
       exploratory item factor analysis. *Psychometrika*, *86* (1), 1--29.
       `https://link.springer.com/article/10.1007/s11336-021-09748-3
       <https://link.springer.com/article/10.1007/s11336-021-09748-3>`_
    
    .. [2] Urban, C. J. (2021). *Machine learning-based estimation and goodness-of-fit for
       large-scale confirmatory item factor analysis* (Publication No. 28772217)
       [Master's thesis, University of North Carolina at Chapel Hill].
       ProQuest Dissertations Publishing.
       `https://www.proquest.com/docview/2618877227/21C6C467D6194C1DPQ/
       <https://www.proquest.com/docview/2618877227/21C6C467D6194C1DPQ/>`_
    
    .. [3] Camilli, G., & Fox, J.-P. (2015). An aggregate IRT procedure for exploratory
       factor analysis. *Journal of Educational and Behavioral Statistics*, *40*, 377--401.
    
    .. [4] von Davier, M., & Sinharay, S. (2010). Stochastic approximation methods for
       latent regression item response models. *Journal of Educational and Behavioral
       Statistics*, *35* (2), 174--193.
    
    .. [5] Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019).
       Neural spline flows. *Advances in Neural Information Processing Systems*, *32*.
       `https://papers.nips.cc/paper/2019/hash/7ac71d433f282034e088473244df8c02-Abstract.html
       <https://papers.nips.cc/paper/2019/hash/7ac71d433f282034e088473244df8c02-Abstract.html>`_
       
    .. [6] Dolatabadi, H. M., Erfani, S., & Leckie, C. (2020). Invertible generative modeling
       using linear rational splines. *Proceedings of the 23rd International Conference on
       Artificial Intelligence and Statistics*, *108*, 4236--4246.
       `http://proceedings.mlr.press/v108/dolatabadi20a
       <http://proceedings.mlr.press/v108/dolatabadi20a>`_
       
    .. [7] Cremer, C., Morris, Q., & Duvenaud, D. (2017). Reinterpreting importance-weighted
       autoencoders. In 5th International Conference on Learning Representations. ICLR.
       `https://arxiv.org/abs/1704.02916 <https://arxiv.org/abs/1704.02916>`_.
       
    Examples
    --------
    .. code-block::
    
        >>> import deepirtools
        >>> from deepirtools import IWAVE
        >>> import torch
        >>> deepirtools.manual_seed(123)
        >>> data = deepirtools.load_grm()["data"]
        >>> n_items = data.shape[1]
        >>> model = IWAVE(
        ...     model_type = "grm",
        ...     latent_size = 4,
        ...     n_cats = [3] * n_items,
        ...     Q = torch.block_diag(*[torch.ones([3, 1])] * 4),
        ...     correlated_factors = [0, 1, 2, 3],
        ... )
        Initializing model parameters
        Initialization ended in  0.0  seconds
        >>> model.fit(data, iw_samples = 10)
        Fitting started
        Epoch =      100 Iter. =    25201 Cur. loss =   10.68   Intervals no change =  100
        Fitting ended in  109.23  seconds
        >>> model.loadings
        tensor([[0.8295, 0.0000, 0.0000, 0.0000],
                [0.5793, 0.0000, 0.0000, 0.0000],
                [0.7116, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.7005, 0.0000, 0.0000],
                [0.0000, 1.1687, 0.0000, 0.0000],
                [0.0000, 1.2890, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.9268, 0.0000],
                [0.0000, 0.0000, 1.2653, 0.0000],
                [0.0000, 0.0000, 1.5622, 0.0000],
                [0.0000, 0.0000, 0.0000, 1.0346],
                [0.0000, 0.0000, 0.0000, 1.3641],
                [0.0000, 0.0000, 0.0000, 1.1348]])
        >>> model.intercepts
        tensor([[ 2.4245, -0.1637],
                [ 1.8219, -1.0013],
                [ 2.0811, -1.1320],
                [ 0.0948, -1.7253],
                [ 2.6597, -2.3412],
                [ 0.2610, -1.4938],
                [ 2.8196, -1.3281],
                [ 0.4833, -2.8053],
                [ 1.6395, -2.2220],
                [ 1.3482, -1.8870],
                [ 2.1606, -2.8600],
                [ 2.5318, -0.1333]])
        >>> model.cov
        tensor([[ 1.0000, -0.0737,  0.2130,  0.2993],
                [-0.0737,  1.0000, -0.1206, -0.3031],
                [ 0.2130, -0.1206,  1.0000,  0.1190],
                [ 0.2993, -0.3031,  0.1190,  1.0000]])
        >>>  model.log_likelihood(data)
        Computing approx. LL
        Approx. LL computed in 33.27 seconds
        Out[6]: -85961.69088745117
        >>> model.scores(data)
        tensor([[-0.6211,  0.1301, -0.7207, -0.7485],
                [ 0.2189, -0.2649,  0.0109, -0.2363],
                [ 0.0544,  0.9308,  0.7940, -0.8851],
                ...,
                [-0.2964, -0.9597, -0.8885, -0.0057],
                [-1.6015,  0.9812,  0.0486,  0.1773],
                [ 2.0448,  0.0583,  1.2005, -0.9317]])
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

            A :math:`\text{sample_size} \times \text{n_items}` matrix.
        missing_mask : Tensor, default = None
            Binary mask indicating missing item responses.

            A :math:`\text{sample_size} \times \text{n_items}` matrix.
        covariates : Tensor, default = None
            Matrix of covariates.

            A :math:`\text{sample_size} \times \text{covariate_size}` matrix.
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
                    batch_size = 32, shuffle = False,
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
               return_mu:    bool = True,
              ):
        r"""Approximate expected a posteriori (EAP) factor scores given a data set.
        
        Parameters
        __________
        data : Tensor
            Data set.

            A :math:`\text{sample_size} \times \text{n_items}` matrix.
        missing_mask : Tensor, default = None
            Binary mask indicating missing item responses.

            A :math:`\text{sample_size} \times \text{n_items}` matrix.
        covariates : Tensor, default = None
            Matrix of covariates.

            A :math:`\text{sample_size} \times \text{covariate_size}` matrix.
        mc_samples : int, default = 1
            Number of Monte Carlo samples.

            Increasing this decreases the EAP estimator's variance.
        iw_samples : int, default = 5000
            Number of importance-weighted samples.

            Increasing this decreases the EAP estimator's bias. When ``iw_samples > 1``, samples
            are drawn from the expected importance-weighted distribution using sampling-
            importance-resampling. [7]_
            
        return_mu : bool, default = True
            Whether to return exact approximate EAP scores (i.e., approximate EAP scores with no sampling error).
            
            Equivalent to using an infinite number of Monte Carlo samples. Only applicable when ``iw_samples = 1``.
        Returns
        _______
        factor_scores : Tensor
            Approximate EAP factor scores given the data set.
            
            A :math:`\text{sample_size} \times \text{latent_size}` matrix.
        """
        
        loader = torch.utils.data.DataLoader(
                    tensor_dataset(data = data, mask = missing_mask,
                                   covariates = covariates),
                    batch_size = 32, shuffle = False,
                    pin_memory = self.device == "cuda",
                  )
        
        scores = []
        for batch in loader:
            batch = {k : v.to(self.device).float() if v is not None else v for k, v in batch.items()}
            
            if return_mu and iw_samples == 1:
                if batch["covariates"] is not None:
                    try:
                        y = torch.cat((batch["y"], batch["covariates"]), dim = 1)
                    except TypeError:
                        if covariates is None:
                            logging.exception("Covariates must be passed to scores() when covariate_size > 0.")
                else:
                    y = batch["y"]
                scores.append(self.model.encode(y, mc_samples = 1, iw_samples = 1).squeeze((0, 1)))
            else:
                elbo, x = self.model(**batch, mc_samples = mc_samples, iw_samples = iw_samples,
                                     **self.runtime_kwargs)
                w_tilda = (elbo - elbo.logsumexp(dim = 0)).exp()
                latent_size = x.size(-1)

                idxs = torch.distributions.Categorical(probs = w_tilda.permute([1, 2, 3, 0])).sample()
                idxs = idxs.expand(x[-1, ...].shape).unsqueeze(0).long()
                scores.append(torch.gather(x, axis = 0, index = idxs).squeeze(0).mean(dim = 0))                  
        return torch.cat(scores, dim = 0).cpu()
    
    @torch.no_grad()
    def sample(self,
               sample_size:   int,
               covariates:    Optional[torch.Tensor] = None,
               return_scores: bool = False,
              ):
        r"""Sample from model.
        
        Parameters
        __________
        sample_size : int
            Number of observations to sample.
        covariates : Tensor, default = None
            Matrix of covariates.

            A :math:`\text{sample_size} \times \text{covariate_size}` matrix.
        return_scores : bool, default = False
            Whether to reture sampled factor scores in addition to item responses.
        
        Returns
        _______
        samples : dict
            Sampled observations and factor scores.
        """
        
        y, x = self.model.sample(sample_size = sample_size, covariates = covariates,
                                 return_scores = return_scores)
        return {"obs" : y, "scores" : x}
        
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
            return logits_to_probs(self.model.decoder.logits.data.cpu(), is_binary = True)
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