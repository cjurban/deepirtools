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
from sim_utils import (get_params_and_data,
                       match_columns,
                       get_constraints,
                      )


ABS_TOL = 0.1

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
                               ["fixed_means", "latent_regression", "free"],
                               [True, False],
                               devices
                              )
    return [["_".join((str(i) for i in idx))] + [p for p in prod] for
            idx, prod in prods if not ((prod[2] == 1 and prod[3] == "fixed_variances") or
                                       (prod[1] not in ("grm", "gpcm") and not prod[5]) or
                                       (prod[0] != "linear" and prod[3] == "free")
                                      )
           ]


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
    res = get_params_and_data(model_type, n_indicators, latent_size, cov_type, mean_type,
                              sample_size, all_same_n_cats)
    
    n_items = res["Y"].shape[1]
    lr = (0.1/(latent_size+1))*5**-1
    iwave_kwargs = {"model_type" : res["model_type"], "ints_mask" : res["ints_mask"]}
    if model_type in ("grm", "gpcm", "mixed"):
        iwave_kwargs["n_cats"] = res["n_cats"]
    else:
        iwave_kwargs["n_items"] = n_items
    if (model_type == "lognormal") or ("lognormal" in res["model_type"]):
        lr *= 1e-2 # Lognormal needs a small learning rate for stability.
    if cov_type == "free":
        iwave_kwargs["fixed_variances"] = False
    if mean_type == "latent_regression":
        iwave_kwargs["covariate_size"] = 2
    elif mean_type == "free":
        iwave_kwargs["fixed_means"] = False
    if cov_type != "fixed_variances_no_covariances":
        iwave_kwargs["correlated_factors"] = [i for i in range(latent_size)]
    constraints = get_constraints(latent_size, n_indicators, constraint_type)

    model = IWAVE(learning_rate = lr,
                  device = device,
                  input_size = n_items,
                  inference_net_sizes = [100],
                  latent_size = latent_size,
                  **{**iwave_kwargs, **constraints},
                  )
    model.fit(res["Y"], covariates = res["covariates"], batch_size = 128, iw_samples = 5)
    
    exp_ldgs, exp_ints, exp_cov_mat = res["loadings"], res["intercepts"], res["cov_mat"]
    exp_res_std, exp_probs = res["residual_std"], res["probs"]
    
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
    if "gpcm" in res["model_type"] and len(exp_ints.shape) == 2:
        if exp_ints.shape[1] > 1:
            gpcm_idxs = torch.Tensor([i for i, m in enumerate(sim.model_types) if m == "gpcm"]).long()
            est_ints[gpcm_idxs] = est_ints[gpcm_idxs].cumsum(dim = 1)
    if not exp_res_std.isnan().all():
        est_res_std = model.residual_std
    if not exp_probs.isnan().all():
        exp_probs = model.probs
    
    ldgs_err = match_columns(est_ldgs, exp_ldgs).add(-exp_ldgs).abs()
    ints_err = est_ints.add(-exp_ints)[~exp_ints.isnan()].abs()
    assert(ldgs_err[ldgs_err != 0].mean().le(ABS_TOL)), print(est_ldgs)
    assert(ints_err[ints_err != 0].mean().le(ABS_TOL)), print(est_ints)
    if est_cov_mat is not None:
        cov_err = invert_cov(est_cov_mat, est_ldgs).add(-exp_cov_mat).tril().abs()
        assert(cov_err[cov_err != 0].mean().le(ABS_TOL)), print(est_cov_mat)
    if not exp_res_std.isnan().all():
        res_std_err = est_res_std.add(-exp_res_std).abs()
        assert(res_std_err[~res_std_err.isnan()].mean().le(ABS_TOL)), print(est_res_std)
    if not exp_probs.isnan().all():
        res_probs = est_probs.add(-exp_probs).abs()
        assert(res_probs[~res_probs.isnan()].mean().le(ABS_TOL)), print(est_probs)