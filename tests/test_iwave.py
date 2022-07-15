import os
from os.path import join
import pytest
import subprocess
import torch
from deepirtools import IWAVE
from deepirtools.utils import *
from simulators import *
from utils import *


ABS_TOL = 0.1
DATA_DIR = os.path.join("tests", "data")
EXPECTED_DIR = os.path.join("tests", "expected")
SIMULATORS = {"poisson" : PoissonFactorModelSimulator,
              "negative_binomial" : NegativeBinomialFactorModelSimulator,
              "normal" : NormalFactorModelSimulator,
              "lognormal" : LogNormalFactorModelSimulator,
             }


deepirtools.manual_seed(1234)
devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")

sample_size = 1000


@pytest.mark.parametrize("latent_size", [1, 5])
@pytest.mark.parametrize("model_type", ["grm", "gpcm", "poisson", "negative_binomial",
                                        "normal", "lognormal"])
@pytest.mark.parametrize("device", devices)
def test_latent_sizes(latent_size, model_type, device):
    n_indicators = 5
    n_items = int(n_indicators * latent_size)
    cov_mat = torch.eye(latent_size, device=device)
    if model_type in ("grm", "gpcm"):
        ldgs = generate_loadings(n_indicators, latent_size).to(device)
        ints, n_cats = generate_graded_intercepts(n_items).to(device)
        iwave_kwargs = {"n_cats" : n_cats}
    else:
        iwave_kwargs = {"n_items" : n_items}
        
        if model_type != "normal":
            ldgs = generate_loadings(n_indicators, latent_size, shrink = True).to(device)
            ints = generate_non_graded_intercepts(n_items, all_positive = True).to(device)
            if model_type == "negative_binomial":
                sim_kwargs = {"probs" : pydist.Uniform(0.5, 0.7).sample([n_items]).to(device)}
            elif model_type == "lognormal":
                sim_kwargs = {"residual_std" : pydist.Uniform(1, 1.2).sample([n_items]).to(device)}
        else:
            ldgs = generate_loadings(n_indicators, latent_size).to(device)
            ints = generate_non_graded_intercepts(n_items).to(device)
            sim_kwargs = {"residual_std" : pydist.Uniform(0.6, 0.8).sample([n_items]).to(device)}
        Y = SIMULATORS[model_type](loadings = ldgs, intercepts = ints,
                                   cov_mat = cov_mat, **sim_kwargs).sample(sample_size)
        
    model = IWAVE(learning_rate = 1e-3,
                  device = device,
                  model_type = model_type,
                  input_size = n_items,
                  inference_net_sizes = [100],
                  latent_size = latent_size,
                  **iwave_kwargs,
                  )
    model.fit(Y, batch_size = 128, iw_samples = 5)
    
    assert(invert_factors(model.loadings).add(-ldgs).abs().le(ABS_TOL).all())
    assert(model.intercepts.add(-ints).abs().le(ABS_TOL).all())
    