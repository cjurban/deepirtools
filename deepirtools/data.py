import pkg_resources
import pandas as pd
import numpy as np
import torch

def load_grm():
    r"""Return data-generating parameters and sampled data for a graded response model.
    
    The generating model has four correlated latent factors and twelve items with
    three categories each.
    
    Returns
    _______
    res : dict
        A dictionary containing sampled data and data-generating parameters. 
        The returned dictionary includes the following key-value pairs:

        * \"data\", the sampled data set of item responses;
        * \"loadings\", the data-generating factor loadings;
        * \"intercepts\", the data-generating category intercepts;
        * \"cov_mat\", the data-generating factor covariance matrix; and
        * \"factor_scores\", the sampled factor scores.
    """
    
    keys = ["data", "loadings", "intercepts", "cov_mat", "factor_scores"]
    res = {}
    for k in keys:
        stream = pkg_resources.resource_stream(__name__, "data/" + k + ".csv")
        res[k] = torch.from_numpy(pd.read_csv(stream, sep = ",", header = None).to_numpy())
    return res