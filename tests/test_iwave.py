import os
from os.path import join
import pytest
import torch
from deepirtools import IWAVE
from deepirtools.utils import *
from sim_utils import *


abs_tol = 0.1
expected_dir = "expected"
data_dir = "data"

os.makedirs(expected_dir, exist_ok = True)
os.makedirs(data_dir, exist_ok = True)

deepirtools.manual_seed(1234)
devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")

sample_size = 1000


@pytest.mark.parametrize("model_type", ["grm", "gpcm", "poisson", "negative_binomial",
                                        "normal", "lognormal"])
@pytest.mark.parametrize("cov_type", [0, 1])
@pytest.mark.parametrize("latent_size", [1, 5])
@pytest.mark.parametrize("device", devices)
def test_latent_sizes(model_type, latent_size, cov_type, device):
    n_indicators = 5
    Y = simulate_and_save_data(model_type, n_indicators, latent_size, cov_type,
                               expected_dir, data_dir)
        
    model = IWAVE(learning_rate = 1e-3,
                  device = device,
                  model_type = model_type,
                  input_size = n_items,
                  inference_net_sizes = [100],
                  latent_size = latent_size,
                  **iwave_kwargs,
                  )
    model.fit(Y, batch_size = 128, iw_samples = 5)
    
    assert(invert_factors(model.loadings).add(-ldgs).abs().le(abs_tol).all())
    assert(model.intercepts.add(-ints).abs().le(abs_tol).all())
    