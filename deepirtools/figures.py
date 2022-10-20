import torch
import numpy as np
from typing import List, Optional, Union
import matplotlib.pyplot as plt
from pylab import *
from deepirtools.iwave import IWAVE
from deepirtools.utils import manual_seed, invert_factors


def screeplot(latent_sizes:             List[int],
              data:                     torch.Tensor,
              model_type:               Union[str, List[str]],
              test_size:                float,
              inference_net_sizes_list: Optional[List[List[int]]] = None,
              learning_rates:           Optional[List[float]] = None,
              missing_mask:             Optional[torch.Tensor] = None,
              max_epochs:               int = 100000,
              batch_size:               int = 32,
              gradient_estimator:       str = "dreg",
              device:                   str = "cpu",
              log_interval:             int = 100,
              mc_samples_fit:           int = 1,
              iw_samples_fit:           int = 1,
              mc_samples_ll:            int = 1,
              iw_samples_ll:            int = 5000,
              random_seed:              int = 1,
              xlab:                     str = "Number of Factors",
              ylab:                     str = "Predicted Approximate Negative Log-Likelihood",
              title:                    str = "Approximate Log-Likelihood Scree Plot",
              **model_kwargs,          
             ):
    r"""Make a log-likelihood screeplot. [1]_
    
    Useful in the exploratory setting to detect the number of latent factors. Result is saved
    as a PDF in the working directory.
    
    Parameters
    __________
    latent_sizes : list of int
        Latent dimensions to plot.
    data : Tensor
        Data set.

        A :math:`\text{sample_size} \times \text{n_items}` matrix.
    model_type : str or list of str
        Measurement model type.
        
        Can either be a string if all items have same type or a list of strings
        specifying each item type.
        
        Current options are:
        
        * \"grm\", graded response model;

        * \"gpcm\", generalized partial credit model;
        
        * \"nominal\", nominal response model;

        * \"poisson\", poisson factor model;

        * \"negative_binomial\", negative binomial factor model;

        * \"normal\", normal factor model; and

        * \"lognormal\", lognormal factor model.
        
        See :obj:`~deepirtools.iwave.IWAVE` class documentation for further details
        regarding each model type.
    test_size : float
        Proportion of data used for calculating LL. Range of values is :math:`(0, 1)`.
    inference_net_sizes_list : list of list of int, default = None
        Neural net hidden layer sizes for each latent dimension in the screeplot.
        
        For example, when making a screeplot for three latent dimensions, setting
        ``inference_net_sizes_list = [[150, 75], [150, 75], [150, 75]]`` creates three neural nets
        each with two hidden layers of size 150 and 75, respectively.
        
        The default ``inference_net_sizes_list = None`` sets each neural net to have a single
        hidden layer of size 100.
    learning_rates : list of float, default = None
        Step sizes for stochastic gradient optimizers.
        
        The default ``learning_rates = None`` sets each optimizer's step size to 0.001.
    missing_mask : Tensor, default = None
        Binary mask indicating missing item responses.

        A :math:`\text{sample_size} \times \text{n_items}` matrix.
    max_epochs : int, default = None
        Number of passes through the full data set after which fitting should be terminated if
        convergence not achieved.
    batch_size : int, default = 32
        Mini-batch size for stochastic gradient optimizer.
    gradient_estimator : str, default = "dreg"
        Gradient estimator for inference model parameters.
        
        Current options are:
        
        * \"dreg\", doubly reparameterized gradient estimator; and
        * \"iwae\", standard gradient estimator.
        
        \"dreg\" is the recommended option due to its bounded variance as the number of importance-weighted
        samples increases.
    device : int, default = "cpu"
        Computing device used for fitting.
    log_interval : str, default = 100
        Frequency of updates printed during fitting.
    mc_samples_fit : int, default = 1
        Number of Monte Carlo samples for fitting.
    iw_samples_fit : int, default = 1
        Number of importance-weighted samples for fitting.
    mc_samples_ll : int, default = 1
        Number of Monte Carlo samples for calculating approximate log-likelihoods.
    iw_samples_ll : int, default = 5000
        Number of importance-weighted samples for calculating approximate log-likelihoods.
    random_seed : int, default = 1
        Seed for reproducibility.
    xlab : str, default = "Number of Factors"
        Screeplot x-axis label.
    ylab : str, default = "Predicted Approximate Negative Log-Likelihood"
        Screeplot y-axis label.
    title : str, default = "Approximate Log-Likelihood Scree Plot"
        Screeplot title.
    **model_kwargs
        Additional keyword arguments passed to ``IWAVE.__init__()``.
        
    Returns
    _______
    ll_list : list of float
        List of approximate hold-out set log-likelihoods for each latent dimension.
        
    Examples
    --------
    .. code-block::
    
        >>> import deepirtools
        >>> from deepirtools import screeplot
        >>> deepirtools.manual_seed(123)
        >>> data = deepirtools.load_grm()["data"]
        >>> n_items = data.shape[1]
        >>> screeplot(
        ...     latent_sizes = [2, 3, 4, 5, 6],
        ...     data = data,
        ...     model_type = "grm",
        ...     test_size = 0.1,
        ...     iw_samples_fit = 10,
        ...     n_cats = [3] * n_items,
        ... )
        Latent size =  2

        Initializing model parameters
        Initialization ended in  0.0  seconds

        Fitting started
        Epoch =       79 Iter. =    17801 Cur. loss =   11.78   Intervals no change =  100
        Fitting ended in  43.68  seconds

        Computing approx. LL
        Approx. LL computed in 2.02 seconds

        Latent size =  3

        Initializing model parameters
        Initialization ended in  0.0  seconds

        Fitting started
        Epoch =      150 Iter. =    33901 Cur. loss =   10.66   Intervals no change =  100
        Fitting ended in  73.85  seconds

        Computing approx. LL
        Approx. LL computed in 2.48 seconds

        Latent size =  4

        Initializing model parameters
        Initialization ended in  0.0  seconds

        Fitting started
        Epoch =       97 Iter. =    22001 Cur. loss =   10.69   Intervals no change =  100
        Fitting ended in  48.14  seconds

        Computing approx. LL
        Approx. LL computed in 2.54 seconds

        Latent size =  5

        Initializing model parameters
        Initialization ended in  0.0  seconds

        Fitting started
        Epoch =       65 Iter. =    14801 Cur. loss =   10.23   Intervals no change =  100
        Fitting ended in  32.25  seconds

        Computing approx. LL
        Approx. LL computed in 2.53 seconds

        Latent size =  6

        Initializing model parameters
        Initialization ended in  0.0  seconds

        Fitting started
        Epoch =       87 Iter. =    19701 Cur. loss =   10.85   Intervals no change =  100
        Fitting ended in  47.94  seconds

        Computing approx. LL
        Approx. LL computed in 2.37 seconds
        
        [-8626.66552734375,
         -8601.913513183594,
         -8603.8798828125,
         -8604.689666748047,
         -8603.455841064453]
    """
    
    assert(0 < test_size < 1), "Test size must be between 0 and 1."
    sample_size = data.size(0)
    n_items = data.size(1)
    
    train_idxs = torch.multinomial(torch.ones(sample_size), int(ceil((1 - test_size) * sample_size)))
    test_idxs = np.setdiff1d(range(sample_size), train_idxs)
    data_train = data[train_idxs]; data_test = data[test_idxs]
    if missing_mask is not None:
        mask_train = missing_mask[train_idxs]; mask_test = missing_mask[test_idxs]
    else:
        mask_train = mask_test = None
    
    if inference_net_sizes_list is None:
        inference_net_sizes_list = [[100]] * len(latent_sizes)
    if learning_rates is None:
        learning_rates = [0.001] * len(latent_sizes)
    latent_sizes, learning_rates, inference_net_sizes_list = zip(*sorted(zip(latent_sizes,
                                                                             learning_rates,
                                                                             inference_net_sizes_list)))
            
    manual_seed(random_seed)
    ll_list = []
    for idx, latent_size in enumerate(latent_sizes):
        print("\nLatent size = ", latent_size, end="\n")
        model = IWAVE(model_type = model_type,
                      learning_rate = learning_rates[idx],
                      device = device,
                      gradient_estimator = gradient_estimator,
                      log_interval = log_interval,
                      latent_size = latent_size,
                      inference_net_sizes = inference_net_sizes_list[idx],
                      **model_kwargs,
                     )
        model.fit(data_train, batch_size = batch_size, missing_mask = mask_train,
                  max_epochs = max_epochs, mc_samples = mc_samples_fit, iw_samples = iw_samples_fit)

        ll = model.log_likelihood(data_test, missing_mask = mask_test,
                                  mc_samples = mc_samples_ll, iw_samples = iw_samples_ll)
        ll_list.append(ll)
        
    fig, ax = plt.subplots(constrained_layout = True)
    fig.set_size_inches(5, 5, forward = True)
    fig.set_size_inches(5, 5, forward = True)
    
    ax.plot(latent_sizes, [-ll for ll in ll_list], "k-o")
    
    ax.set_xticks(np.arange(min(latent_sizes) - 1, max(latent_sizes) + 2).tolist())
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    fig.suptitle(title)

    fig.savefig("screeplot.pdf")
    fig.show()
    
    return ll_list

    
def loadings_heatmap(loadings: torch.Tensor,
                     xlab:     str = "Factor", 
                     ylab:     str = "Item", 
                     title:    str = "Factor Loadings"):
    """Make heatmap of factor loadings.
    
    Result is saved as a PDF in the working directory.
    
    Parameters
    __________
    loadings : Tensor
        Factor loadings matrix.
    xlab : str, default = "Factor"
        Heatmap x-axis label.
    ylab : str, default = "Item"
        Heatmap y-axis label.
    title : str, default = "Factor Loadings"
        Heatmap title.
    """
    
    latent_size = loadings.shape[1]
    
    c = pcolor(invert_factors(loadings))
    set_cmap("gray_r")
    colorbar() 
    c = pcolor(invert_factors(loadings), edgecolors = "w", linewidths = 1, vmin = 0) 
    xlabel(xlab)
    ylabel(ylab)
    xticks(np.arange(latent_size) + 0.5,
           [str(size + 1) for size in range(latent_size)])
    ytick_vals = [int(10 * (size + 1)) for size in range(latent_size)]
    yticks(np.array(ytick_vals) - 0.5, [str(val) for val in ytick_vals])
    plt.gca().invert_yaxis()
    suptitle(title, y = 0.93)
    
    plt.savefig("loadings_heatmap.pdf")
    plt.show()