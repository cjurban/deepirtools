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

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")

sample_size = 8000
n_indicators = 4


def _recovery_args():
    prods = product(["mixed", "gpcm", "grm", "nominal",
                     "negative_binomial", "poisson", "normal", "lognormal"],
                    ["none", "binary", "linear"],
                    [1, 3],
                    ["fixed_variances_no_covariances", "fixed_variances", "free"],
                    ["fixed_means", "latent_regression", "free"],
                    [True, False],
                    devices
                   )
    return [[p for p in prod] for prod in prods if
            not ((prod[2] == 1 and prod[3] == "fixed_variances") or
                 (prod[0] not in ("grm", "gpcm") and not prod[5]) or
                 (prod[1] != "linear" and prod[3] == "free") or
                 (prod[1] == "linear" and prod[3] != "free") or
                 (prod[1] == "none" and prod[4] != "fixed_means")
                )
           ]


@pytest.mark.parametrize(("model_type, constraint_type, latent_size, "
                          "cov_type, mean_type, all_same_n_cats, device"), _recovery_args())
def test_param_recovery(model_type:      str,
                        constraint_type: str,
                        latent_size:     int,
                        cov_type:        str,
                        mean_type:       str,
                        all_same_n_cats: bool,
                        device:          str,
                       ):
    """Test parameter recovery for I-WAVE."""
    deepirtools.manual_seed(123)
    
    res = get_params_and_data(model_type, n_indicators, latent_size, cov_type, mean_type,
                              sample_size, all_same_n_cats)
    
    # Get arguments for IWAVE() and fit().
    n_items = res["Y"].shape[1]
    iwave_kwargs = {"model_type" : res["model_type"], "ints_mask" : res["ints_mask"]}
    if model_type in ("grm", "gpcm", "mixed"):
        iwave_kwargs["n_cats"] = res["n_cats"]
    else:
        iwave_kwargs["n_items"] = n_items
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

    model = IWAVE(
        learning_rate = (0.01 if constraint_type != "linear" else 0.001), # Marker constraints impact likelihood surface shape,
        device = device,                                                  # making optimization more challenging.
        latent_size = latent_size,
        n_intervals = (20 if constraint_type != "linear" else 100), # Convergence happens in fewer iterations with a larger learning rate.
        **{**iwave_kwargs, **constraints},
    )
    model.fit(res["Y"], covariates = res["covariates"], batch_size = 256, iw_samples = 64) # Large batch_size and iw_samples to reduce
                                                                                           # both bias and variance.
    
    # Extract parameters.
    true_params = [res["loadings"], res["cov_mat"], res["intercepts"],
                   res["residual_std"], res["probs"],
                   (res["mean"] if mean_type == "free" else None), 
                   res["latent_regression_weight"],
                  ]
    est_params = []
    if latent_size > 1 and constraint_type == "none":
        if cov_type == "fixed_variances_no_covariances":
            rotator = Rotator(method = "varimax")
        elif cov_type == "fixed_variances":
            rotator = Rotator(method = "geomin_obl")
        est_params.extend((torch.from_numpy(rotator.fit_transform(model.loadings.numpy())),
                           (torch.from_numpy(rotator.phi_) if
                            rotator.phi_ is not None else None)))
    else:
        est_params.extend((model.loadings,
                           (None if ((latent_size == 1 and "free" not in cov_type) or
                                     (latent_size > 1 and "no_covariances" in cov_type))
                            else model.cov)))
    est_params.extend((model.intercepts, model.residual_std, model.probs,
                       (model.mean if mean_type == "free" else None),
                       model.latent_regression_weight))
    
    # Prepare parameters for comparison.
    if latent_size > 1 and est_params[1] is not None:
        est_params[1] = invert_cov(est_params[1], est_params[0])
    if mean_type == "free":
        est_params[2], true_params[2] = (est_params[2][true_params[2].eq(0).logical_not()],
                                         true_params[2][true_params[2].eq(0).logical_not()])
    if est_params[-2] is not None:
        est_params[-2] = invert_mean(est_params[-2], est_params[0])
    if est_params[-1] is not None:
        est_params[-1] = invert_latent_regression_weight(est_params[-1], est_params[0])
    est_params[0], match_idxs = match_columns(est_params[0], true_params[0])
    est_params[0], true_params[0] = (est_params[0][est_params[0].eq(0).logical_not()],
                                     true_params[0][est_params[0].eq(0).logical_not()])
    if constraint_type != "none":
        est_params[0], true_params[0] = (est_params[0][true_params[0].eq(1).logical_or(true_params[0].eq(0)).logical_not()],
                                         true_params[0][true_params[0].eq(1).logical_or(true_params[0].eq(0)).logical_not()])
    if est_params[1] is not None:
        est_params[1] = est_params[1][match_idxs][:, match_idxs].tril(
            diagonal = (0 if cov_type == "free" else -1)
        )
        est_params[1], true_params[1] = (est_params[1][est_params[1].eq(0).logical_not()],
                                         true_params[1][est_params[1].eq(0).logical_not()])
    
    errs = []
    for i, params in enumerate(zip(est_params, true_params)):
        est, true = params[0], params[1]
        if true is not None and est is not None:
            errs.append(est.add(-true)[true.isnan().logical_not()])
    assert(torch.cat(errs, dim = 0).abs().mean().lt(ABS_TOL))
        

@pytest.mark.parametrize("model_type", ["mixed", "gpcm", "grm", "nominal",
                                        "negative_binomial", "poisson", "normal", "lognormal"])
@pytest.mark.parametrize("latent_size", [1, 3])
@pytest.mark.parametrize("device", devices)
def test_scores_recovery(model_type:      str,
                         latent_size:     int,
                         device:          str,
                        ):
    """Test factor scores recovery for I-WAVE."""
    deepirtools.manual_seed(123)
    
    res = get_params_and_data(model_type, n_indicators, latent_size, "fixed_variances",
                              "fixed_means", sample_size)
    
    # Get arguments for IWAVE() and fit().
    n_items = res["Y"].shape[1]
    iwave_kwargs = {"model_type" : res["model_type"], "ints_mask" : res["ints_mask"]}
    if model_type in ("grm", "gpcm", "mixed"):
        iwave_kwargs["n_cats"] = res["n_cats"]
    else:
        iwave_kwargs["n_items"] = n_items
    if latent_size > 1:
        iwave_kwargs["correlated_factors"] = [i for i in range(latent_size)]
    constraints = get_constraints(latent_size, n_indicators, "binary")

    model = IWAVE(
        learning_rate = 0.01,
        device = device,
        latent_size = latent_size,
        n_intervals = 20,
        **{**iwave_kwargs, **constraints},
    )
    model.fit(res["Y"], batch_size = 256, iw_samples = 64)
    
    est_scores = model.scores(res["Y"], mc_samples = 100, iw_samples = 100)
    for i in range(latent_size):
        combined_scores = torch.stack((est_scores[:, i], res["scores"][:, i]), dim = 0)
        assert(torch.corrcoef(combined_scores)[1, 0].gt(0.5))