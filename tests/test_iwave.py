import os
from os.path import join
from itertools import product
import pytest
import torch
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects.vectors import StrVector
import deepirtools
from deepirtools import IWAVE
from deepirtools.utils import invert_cov
from factor_analyzer import Rotator
from sim_utils import (simulate_and_save_data,
                       match_columns,
                       load_torch_from_csv,
                       get_constraints,
                      )


ABS_TOL = 0.1
EXPECTED_DIR = "expected"
DATA_DIR = "data"

utils = ro.packages.importr("utils")
utils.chooseCRANmirror(ind = 1)
pkgnames = ["mirt"]
names_to_install = [pkg for pkg in pkgnames if not ro.packages.isinstalled(pkg)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

deepirtools.manual_seed(123)
devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")

sample_size = 10000
n_indicators = 5


def _test_args():
    def enumerated_product(*args):
        yield from zip(product(*(range(len(x)) for x in args)), product(*args))
    
    prods = enumerated_product(["none", "binary", "linear"],
                               ["grm", "gpcm", "poisson", "negative_binomial", "normal", "lognormal", "mixed"],
                               [1, 5],
                               ["fixed_variances_no_covariances", "fixed_variances", "free"],
                               ["fixed_means", "latent_regression", "free"]
                               [True, False],
                               devices
                              )
    return [["_".join((str(i) for i in idx))] + [p for p in prod] for
            idx, prod in prods if not ((prod[2] == 1 and prod[3] == "fixed_variances") or
                                       (prod[1] not in ("grm", "gpcm") and not prod[4])
                                      )
           ]


# TODO: Test the following:
#           - Mixed item types
#           - Unconstrained factor means + masked intercepts
#           - Unconstrained variances + reference indicator
#           - Recovery of residual stds. + probs.
#           - GPU
@pytest.mark.parametrize(("idx, constraint_type, model_type, latent_size, "
                          "cov_type, mean_type, all_same_n_cats, device"), _test_args())
def test_param_recovery(idx:             str,
                        constraint_type: str,
                        model_type:      str,
                        latent_size:     int,
                        cov_type:        str,
                        mean_type:       str,
                        all_same_n_cats: bool,
                        device:          str,
                       ):
    """Test parameter recovery for I-WAVE."""
    expected_dir = os.path.join(EXPECTED_DIR, "test_" + idx)
    data_dir = os.path.join(DATA_DIR,  "test_" + idx)
    os.makedirs(expected_dir, exist_ok = True)
    os.makedirs(data_dir, exist_ok = True)
    
    simulate_and_save_data(model_type, n_indicators, latent_size, cov_type, mean_type,
                           sample_size, expected_dir, data_dir, all_same_n_cats)
    exp_ldgs, exp_ints, exp_cov_mat = (load_torch_from_csv(k + ".csv", expected_dir) for
                                       k in ("ldgs", "ints", "cov_mat"))
    if latent_size == 1:
        exp_ldgs = exp_ldgs.unsqueeze(1)
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
    if "fixed_variances" not in cov_type:
        iwave_kwargs["fixed_variances"] = False
    if "no_covariances" not in cov_type:
        iwave_kwargs["correlated_factors"] = [i for i in range(latent_size)]
    constraints = get_constraints(latent_size, n_indicators, constraint_type)

    model = IWAVE(learning_rate = lr,
                  device = device,
                  model_type = model_type,
                  input_size = n_items,
                  inference_net_sizes = [100],
                  latent_size = latent_size,
                  **{**iwave_kwargs, **constraints},
                  )
    model.fit(Y, batch_size = 128, iw_samples = 5)
    
    if latent_size > 1 and constraint_type == "none":
        if cov_type == "fixed_variances_no_covariances":
            rotator = Rotator(method = "varimax")
        elif cov_type == "fixed_variances":
            rotator = Rotator(method = "geomin_obl")
        est_ldgs = torch.from_numpy(rotator.fit_transform(model.loadings.numpy()))
        est_cov_mat = (torch.from_numpy(rotator.phi_) if cov_type == 1 else None)
    else:
        est_ldgs, est_cov_mat = model.loadings, model.cov
    est_ints = model.intercepts
    if model_type == "gpcm" and len(exp_ints.shape) == 2:
        if exp_ints.shape[1] > 1:
            est_ints = est_ints.cumsum(dim = 1)
    
    ldgs_err = match_columns(est_ldgs, exp_ldgs).add(-exp_ldgs).abs()
    ints_err = est_ints.add(-exp_ints)[~exp_ints.isnan()].abs()
    assert(ldgs_err[ldgs_err != 0].mean().le(ABS_TOL)), print(est_ldgs)
    assert(ints_err.mean().le(ABS_TOL)), print(est_ints)
    if est_cov_mat is not None:
        cov_err = invert_cov(est_cov_mat, est_ldgs).add(-exp_cov_mat).abs()
        assert(cov_err[cov_err != 0].mean().le(ABS_TOL)), print(est_cov_mat)