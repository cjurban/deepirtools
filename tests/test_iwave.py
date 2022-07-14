import os
from os.path import join
import pytest
import subprocess
import torch
import deepirtools
from deepirtools import IWAVE
from simulators import *


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

sample_size = 10000


@pytest.mark.parametrize("model_type", ["grm", "gpcm", "poisson", "negative_binomial",
                                        "normal", "lognormal"])
@pytest.mark.parametrize("device", devices)
def test_latent_size_1(model_type, device):
    latent_size = 1
    n_items = 10
    ldgs = pydist.LogNormal(loc = torch.zeros([n_items, latent_size], device=device),
                            scale = torch.ones([n_items, latent_size], device=device).mul(0.5)).sample()
    cov_mat = torch.eye(latent_size, device=device)
    if model_type in ("grm", "gpcm"):
        n_cats = [2] * n_items
        ints = pydist.Uniform(-1.5, 1.5).sample([n_items]).to(device)
        iwave_kwargs = {"n_cats" : n_cats}
    else:
        sim_kwargs = {}
        
        if model_type != "normal":
            ldgs.mul_(0.3).clamp_(max = 0.7)
            ints = pydist.Uniform(0.1, 0.5).sample([n_items]).to(device)
            if model_type == "negative_binomial":
                sim_kwargs["probs"] = pydist.Uniform(0.5, 0.7).sample([n_items]).to(device)
            elif model_type == "lognormal":
                sim_kwargs["residual_std"] = pydist.Uniform(1, 1.2).sample([n_items]).to(device)
        else:
            ints = torch.randn(n_items, device=device).mul(0.1)
            sim_kwargs["residual_std"] = pydist.Uniform(0.6, 0.8).sample([n_items]).to(device)
        Y = SIMULATORS[model_type](loadings = ldgs, intercepts = ints,
                                   cov_mat = cov_mat, **sim_kwargs).sample(sample_size)
        iwave_kwargs = {"n_items" : n_items}
        
    iwave = IWAVE(
                  learning_rate = 1e-3,
                  device = device,
                  model_type = model_type,
                  input_size = n_items,
                  inference_net_sizes = [100],
                  latent_size = latent_size,
                  **iwave_kwargs,
                 )
    iwave.fit(Y, batch_size = 128, iw_samples = 5)
    return iwave
    
model_type = "lognormal"; device = "cpu"
out = test_latent_size_1(model_type, device)

    