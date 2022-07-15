import os
from os.path import join
import pytest
import torch
from deepirtools import IWAVE
from factor_analyzer import Rotator
from deepirtools.utils import *
from sim_utils import *


abs_tol = 0.1
expected_dir = "expected"
data_dir = "data"

deepirtools.manual_seed(1234)
devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")

sample_size = 1000
n_indicators = 5


@pytest.mark.parametrize("model_type", ["grm", "gpcm", "poisson", "negative_binomial",
                                        "normal", "lognormal"])
@pytest.mark.parametrize("cov_type", [0, 1])
@pytest.mark.parametrize("latent_size", [1, 5])
@pytest.mark.parametrize("device", devices)
def test_exploratory_iwave(model_type, latent_size, cov_type, device):
    _expected_dir = os.path.join(expected_dir, "exploratory", "model_type_{}".format(model_type),
                                 "cov_type_{}".format(cov_type), "latent_size_{}".format(latent_size))
    _data_dir = os.path.join(data_dir, "exploratory", "model_type_{}".format(model_type),
                             "cov_type_{}".format(cov_type), "latent_size_{}".format(latent_size))
    os.makedirs(_expected_dir, exist_ok = True)
    os.makedirs(_data_dir, exist_ok = True)
    
    Y, iwave_kwargs = simulate_and_save_data(model_type, n_indicators, latent_size, cov_type,
                                             _expected_dir, _data_dir)
        
    model = IWAVE(learning_rate = 1e-3,
                  device = device,
                  model_type = model_type,
                  input_size = n_items,
                  inference_net_sizes = [100],
                  latent_size = latent_size,
                  **iwave_kwargs,
                  )
    model.fit(Y, batch_size = 128, iw_samples = 5)
    
    exp_params = (np.loadtxt(os.path.join(_expected_dir, "ldgs.csv"), delimiter = ","),
                  np.loadtxt(os.path.join(_expected_dir, "ints.csv"), delimiter = ","),
                  np.loadtxt(os.path.join(_expected_dir, "cov_mat.csv"), delimiter = ","))
    exp_ldgs, exp_ints, exp_cov_mat = (torch.from_numpy(p) for p in exp_params)
    
    if latent_size > 1:
        if cov_type == 0:
            rotator = Rotator(method = "varimax")
            est_cov_mat = None
        elif cov_type == 1:
            rotator = Rotator(method = "geomin_obl")
        est_ldgs = rotator.fit_transform(model.loadings)
        est_cov_mat = rotator.phi_
    else:
        est_ldgs = model.loadings
        est_cov_mat = None
    est_ints = model.intercepts
    
    assert(invert_factors(est_ldgs).add(-exp_ldgs).abs().le(abs_tol).all())
    assert(est_ints.add(-exp_ints).abs().le(abs_tol).all())
    if est_cov_mat is not None:
        assert(invert_cov(est_cov_mat, est_ldgs).add(-exp_cov_mat.abs().le(abs_tol).all())
    