import os
from os.path import join
import pytest
import torch
import numpy as np
import deepirtools
from deepirtools import IWAVE
from deepirtools.utils import invert_factors, invert_cov, match_columns
from factor_analyzer import Rotator
from test_utils import simulate_and_save_data, match_columns, load_torch_from_csv


ABS_TOL = 0.1
EXPECTED_DIR = "expected"
DATA_DIR = "data"


deepirtools.manual_seed(123)
devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")

sample_size = 10000
n_indicators = 5


@pytest.mark.parametrize("model_type", ["grm", "gpcm", "poisson", "negative_binomial",
                                        "normal", "lognormal"])
@pytest.mark.parametrize("cov_type", [0, 1])
@pytest.mark.parametrize("latent_size", [1, 5])
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("all_same_n_cats", [True, False])
def test_exploratory_iwave(model_type, latent_size, cov_type, device, all_same_n_cats):
    """Test parameter recovery in the exploratory setting."""
    if cov_type == 1 and latent_size == 1:
        return
    if model_type not in ("grm", "gpcm") and not all_same_n_cats:
        return
    
    expected_dir = os.path.join(EXPECTED_DIR, "exploratory", "model_type_{}".format(model_type),
                                "cov_type_{}".format(cov_type), "latent_size_{}".format(latent_size),
                                "device_{}".format(device), "all_same_n_cats_{}".format(all_same_n_cats * 1))
    data_dir = os.path.join(DATA_DIR, "exploratory", "model_type_{}".format(model_type),
                            "cov_type_{}".format(cov_type), "latent_size_{}".format(latent_size),
                            "device_{}".format(device), "all_same_n_cats_{}".format(all_same_n_cats * 1))
    os.makedirs(expected_dir, exist_ok = True)
    os.makedirs(data_dir, exist_ok = True)
    
    simulate_and_save_data(model_type, n_indicators, latent_size, cov_type, sample_size,
                           expected_dir, data_dir, all_same_n_cats)
    exp_ldgs, exp_ints, exp_cov_mat = (load_torch_from_csv(k + ".csv", expected_dir) for
                                       k in ("ldgs", "ints", "cov_mat"))
    Y = load_torch_from_csv("data.csv", data_dir)
    
    n_items = Y.shape[1]
    lr = (0.1/(latent_size+1))*5**-1
    iwave_kwargs = {}
    if model_type in ("grm", "gpcm"):
        exp_ints *= -1
        iwave_kwargs["n_cats"] = np.loadtxt(os.path.join(expected_dir, "n_cats.csv"),
                                            delimiter = ",").astype(int).tolist()
    else:
        iwave_kwargs["n_items"] = n_items
        if model_type == "lognormal": # Lognormal needs a small learning rate for stability.
            lr *= 1e-2

    model = IWAVE(learning_rate = lr,
                  device = device,
                  model_type = model_type,
                  input_size = n_items,
                  inference_net_sizes = [100],
                  latent_size = latent_size,
                  **iwave_kwargs,
                  )
    model.fit(Y, batch_size = 128, iw_samples = 5)
    
    if latent_size > 1:
        if cov_type == 0:
            rotator = Rotator(method = "varimax")
        elif cov_type == 1:
            rotator = Rotator(method = "geomin_obl")
        est_ldgs = torch.from_numpy(rotator.fit_transform(model.loadings.numpy()))
        est_cov_mat = (torch.from_numpy(rotator.phi_) if cov_type == 1 else None)  
    else:
        est_ldgs = model.loadings
        est_cov_mat = None
        exp_ldgs = exp_ldgs.unsqueeze(1)
    est_ints = model.intercepts
    if model_type == "gpcm" and len(exp_ints.shape) == 2:
        if exp_ints.shape[1] > 1:
            est_ints = est_ints.cumsum(dim = 1)
    
    ldgs_err = match_columns(est_ldgs, exp_ldgs).add(-exp_ldgs).abs()
    ints_err = est_ints.add(-exp_ints)[~exp_ints.isnan()].abs()
    assert(ldgs_err.mean().le(ABS_TOL)), print(est_ldgs)
    assert(ints_err.mean().le(ABS_TOL)), print(est_ints)
    if est_cov_mat is not None:
        cov_err = invert_cov(est_cov_mat, est_ldgs).add(-exp_cov_mat).abs()
        assert(cov_err.mean().le(ABS_TOL)), print(est_cov_mat)