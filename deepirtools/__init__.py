import torch

from .importance_weighted import ImportanceWeightedEstimator
from .utils import (manual_seed,
                    invert_factors,
                    invert_cov,
                    normalize_loadings,
                    normalize_ints)
from .figures import *
__version__ = "1.0.8"