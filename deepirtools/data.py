import pkg_resources
import pandas as pd
import numpy as np
import torch

def load_grm():
    """Return data-generating parameters and sampled data for a graded response model.
    
    The generating model has four correlated latent factors and twelve items with
    three categories each.
    
    Returns
    _______
        A dictionary containing sampled data and data-generating parameters.
        
        The returned dictionary includes the following key-value pairs:
        
            "data" : The sampled data set of item responses.
            "loadings" : The data-generating factor loadings.
            "intercepts" : The data-generating category intercepts.
            "cov_mat" : The data-generating factor covariance matrix.
            "factor_scores" : The sampled factor scores.
    """
    
    keys = ["data", "loadings", "intercepts", "cov_mat", "factor_scores"]
    res = {}
    for k in keys:
        stream = pkg_resources.resource_stream(__name__, "data/" + k + ".csv")
        res[k] = torch.from_numpy(pd.read_csv(stream, sep = ",", header = None).to_numpy())
    return res