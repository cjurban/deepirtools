import os
from os.path import join
import pytest
import subprocess
import torch
import deepirtools
from . import simulators as sim


DATA_DIR = os.path.join("tests", "data")
EXPECTED_DIR = os.path.join("tests", "expected")


deepirtools.manual_seed(1234)
torch.set_default_dtype(torch.float64)
devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")

sample_size = 1000
batch_size = 32
n_indicators = 4
latent_size = 4
n_items = int(n_indicators * latent_size)

ldgs_sim = sim.LoadingsSimulator(n_indicators, latent_size)
grd_ints_sim = sim.GradedInterceptsSimulator(n_items)
ngrd_ints_sim = sim.NonGradedInterceptsSimulator(n_items)
cov_mat_sim = sim.CovarianceMatrixSimulator(latent_size)


@pytest.mark.parametrize("model_type", ["grm", "gpcm", "poisson", "negative_binomial",
                                        "normal", "lognormal"])
@pytest.mark.parametrize("device", devices)
def test_latent_factor_model(model_type, device):
    ldgs_list = ldgs_sim.sample()
    cov_mat = cov_mat_sim.sample()
    if model_type in ("grm", "gpcm"):
        ints_list = grd_ints_sim.sample()
    else:
        ints_list = ngrd_ints_sim.sample()
    
    