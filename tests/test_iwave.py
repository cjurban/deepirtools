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
from deepirtools.utils import (invert_cov,
                               invert_mean,
                               invert_latent_regression_weight,
                              )
from factor_analyzer import Rotator
from .sim_utils import (get_params_and_data,
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
    
    prods = enumerated_product(["mixed", "lognormal", "gpcm", "grm", "negative_binomial", "normal", "poisson"],
                               ["none", "binary", "linear"],
                               [1, 5],
                               ["fixed_variances_no_covariances", "fixed_variances", "free"],
                               ["fixed_means", "latent_regression", "free"],
                               [True, False],
                               devices
                              )
    return [["_".join((str(i) for i in idx))] + [p for p in prod] for
            idx, prod in prods if not ((prod[2] == 1 and prod[3] == "fixed_variances") or
                                       (prod[0] not in ("grm", "gpcm") and not prod[5]) or
                                       (prod[1] != "linear" and prod[3] == "free") or
                                       (prod[1] == "linear" and prod[3] != "free") or
                                       (prod[1] == "none" and prod[4] != "fixed_means")
                                      )
           ]


@pytest.mark.parametrize(("idx, model_type, constraint_type, latent_size, "
                          "cov_type, mean_type, all_same_n_cats, device"), _test_args())
def test_param_recovery(idx:             str,
                        model_type:      str,
                        constraint_type: str,
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
    if "lognormal" in res["model_type"] or model_type == "mixed":
        lr *= 0.1 # Lognormal and mixed benefit from small learning rates for stability.
    if cov_type == "free":
        iwave_kwargs["fixed_variances"] = False
    if mean_type == "latent_regression":
        iwave_kwargs["covariate_size"] = 2
    elif mean_type == "free":
        iwave_kwargs["fixed_means"] = False
    if constraint_type != "none":
        if latent_size > 1 and cov_type != "fixed_variances_no_covariances":
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
    exp_mean, exp_lreg_weight = res["mean"], res["latent_regression_weight"]
    
    if latent_size > 1 and constraint_type == "none":
        if cov_type == "fixed_variances_no_covariances":
            rotator = Rotator(method = "varimax")
        elif cov_type == "fixed_variances":
            rotator = Rotator(method = "geomin_obl")
        est_ldgs = torch.from_numpy(rotator.fit_transform(model.loadings.numpy()))
        est_cov_mat = (torch.from_numpy(rotator.phi_) if rotator.phi_ is not None else None)
    else:
        est_ldgs, est_cov_mat = model.loadings, model.cov
    est_ints = model.intercepts
    if "gpcm" in res["model_types"] and len(exp_ints.shape) == 2:
        if exp_ints.shape[1] > 1:
            gpcm_idxs = torch.Tensor([i for i, m in enumerate(res["model_types"]) if m == "gpcm"]).long()
            est_ints[gpcm_idxs] = est_ints[gpcm_idxs].cumsum(dim = 1)
    est_res_std, est_probs = model.residual_std, model.probs
    est_mean, est_lreg_weight = model.mean, model.latent_regression_weight
    
    ldgs_err = match_columns(est_ldgs, exp_ldgs).add(-exp_ldgs).abs()
    ints_err = est_ints.add(-exp_ints)[~exp_ints.isnan()].abs()
    assert(ldgs_err[ldgs_err != 0].mean().le(ABS_TOL)), print(ldgs_err)
    assert(ints_err[ints_err != 0].mean().le(ABS_TOL)), print(ints_err)
    if est_cov_mat is not None:
        if ((latent_size > 1 and cov_type != "fixed_variances_no_covariances") or
            (latent_size == 1 and cov_type == "free")):
            cov_err = invert_cov(est_cov_mat, est_ldgs).add(-exp_cov_mat).tril().abs()
            assert(cov_err[cov_err != 0].mean().le(ABS_TOL)), print(cov_err)
    if est_res_std is not None:
        res_std_err = est_res_std.add(-exp_res_std).abs()
        assert(res_std_err[~res_std_err.isnan()].mean().le(ABS_TOL)), print(res_std_err)
    if est_probs is not None:
        probs_err = est_probs.add(-exp_probs).abs()
        assert(probs_err[~probs_err.isnan()].mean().le(ABS_TOL)), print(probs_err)
    if mean_type == "latent_regression":
        lreg_weight_err = invert_latent_regression_weight(est_lreg_weight, est_ldgs).add(-exp_lreg_weight).abs()
        assert(lreg_weight_err.mean().le(ABS_TOL)), print(lreg_weight_err)
    elif mean_type == "free":
        mean_err = invert_mean(est_mean, est_ldgs).add(-exp_mean).abs()
        assert(mean_err.mean().le(ABS_TOL)), print(mean_err)