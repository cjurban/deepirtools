import pkg_resources
import pandas as pd
import numpy as np
import torch

def load_grm():
    """Return data-generating parameters and sampled data for a graded response model."""
    keys = ["data", "loadings", "intercepts", "cov_mat", "factor_scores"]
    res = {}
    for k in keys:
        stream = pkg_resources.resource_stream(__name__, "data/" + k + ".csv")
        res[k] = torch.from_numpy(pd.read_csv(stream, sep = ",", header = None).to_numpy())
    return res