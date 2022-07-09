import torch
import numpy as np
from typing import List, Optional
import matplotlib.pyplot as plt
from pylab import *
from importance_weighted import ImportanceWeightedEstimator
from utils import manual_seed, invert_factors


def screeplot(latent_sizes:             List[int], # need to sort these if they're not
              data:                     torch.Tensor,
              model_type:               str,
              test_size:                float,
              inference_net_sizes_list: List[List[int]],
              learning_rates:           List[float],
              missing_mask:             Optional[torch.Tensor] = None,
              max_epochs:               int = 100000,
              batch_size:               int = 32,
              gradient_estimator:       str = "dreg",
              device:                   str = "cpu",
              log_interval:             int = 100,
              iw_samples_fit:           int = 1,
              iw_samples_ll:            int = 5000,
              random_seed:              int = 1,
              xlabel:                   str = "Number of Factors",
              ylabel:                   str = "Predicted Approximate Negative Log-Likelihood",
              title:                    str = "Approximate Log-Likelihood Scree Plot",
              **model_kwargs,          
             ):
    """
    In exploratory setting, make log-likelihood screeplot to detect number of latent factors.
    
    Args:
        latent_sizes             (List of int):         Latent dimensions to plot.
        data                     (Tensor):              Data set containing item responses.
        model_type               (str):                 Measurement model type.
        test_size                (float):               Proportion of data used for calculating LL.
        inference_net_sizes_list (List of List of int): Neural net input and hidden layer sizes for each latent dimension.
        learning_rates           (List of float):       Step sizes for stochastic gradient optimizers.
        missing_mask             (Tensor):              Binary mask indicating missing item responses.
        max_epochs               (int):                 Number of passes through the full data set after which
                                                        fitting should be terminated if convergence not achieved.
        batch_size               (int):                 Mini-batch size for stochastic gradient optimizer.
        gradient_estimator       (str):                 Gradient estimator for inference model parameters:
        device                   (str):                 Computing device used for fitting.
        log_interval             (str):                 Frequency of updates printed during fitting.
        iw_samples_fit           (int):                 Number of importance-weight samples for fitting.
        iw_samples_ll            (int):                 Number of importance-weight samples for calculating LL.
        random_seed              (int):                 Seed for reproducibility.
        model_kwargs             (dict):                Named parameters passed to VariationalAutoencoder.__init__().
    """
    assert(test_size > 0 and test_size < 1), "Test size must be between 0 and 1."
    sample_size = data.size(0)
    n_items = data.size(1)
    
    train_idxs = torch.multinomial(torch.ones(sample_size), int(ceil((1 - test_size) * sample_size)))
    test_idxs = np.setdiff1d(range(sample_size), train_idxs)
    data_train = data[train_idxs]; mask_train = missing_mask[train_idxs]
    data_test = data[test_idxs]; mask_test = missing_mask[test_idxs]
    
    latent_sizes.sort()
            
    manual_seed(random_seed)
    ll_list = []
    for idx, latent_size in enumerate(latent_sizes):
        print("\nLatent size = ", latent_size, end="\n")
        model = ImportanceWeightedEstimator(learning_rate = learning_rates[idx],
                                            device = device,
                                            model_type = model_type,
                                            gradient_estimator = gradient_estimator,
                                            log_interval = log_interval,
                                            input_size = n_items,
                                            inference_net_sizes = inference_net_sizes_list[idx],
                                            latent_size = latent_size,
                                            **model_kwargs,
                                           )
        model.fit(data_train, batch_size, mask_train, max_epochs, iw_samples = iw_samples_fit)

        ll = model.log_likelihood(data_test, missing_mask=mask_test, iw_samples = iw_samples_ll)
        ll_list.append(ll)
        
    fig, ax = plt.subplots(constrained_layout = True)
    fig.set_size_inches(5, 5, forward = True)
    fig.set_size_inches(5, 5, forward = True)
    
    ax.plot(latent_sizes, [-ll for ll in ll_list], "k-o")
    
    ax.set_xticks(np.arange(min(latent_sizes) - 1, max(latent_sizes) + 2).tolist())
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    fig.suptitle(title)

    fig.savefig("screeplot.pdf")
    fig.show()
    
    return ll_list

    
def loadings_heatmap(loadings:     np.ndarray,
                     x_label:      str = "Factor", 
                     y_label:      str = "Item", 
                     title:        str = "Factor Loadings"):
    """Make heatmap of factor loadings."""
    latent_size = loadings.shape[1]
    
    c = pcolor(invert_factors(loadings))
    set_cmap("gray_r")
    colorbar() 
    c = pcolor(invert_factors(loadings), edgecolors = "w", linewidths = 1, vmin = 0) 
    xlabel(x_label)
    ylabel(y_label)
    xticks(np.arange(latent_size) + 0.5,
           [str(size + 1) for size in range(latent_size)])
    ytick_vals = [int(10 * (size + 1)) for size in range(latent_size)]
    yticks(np.array(ytick_vals) - 0.5, [str(val) for val in ytick_vals])
    plt.gca().invert_yaxis()
    suptitle(title, y = 0.93)
    
    plt.savefig("loadings_heatmap.pdf")
    plt.show()