import numpy as np
import deepirtools
from sim_utils import *

sample_size = 8000
latent_size = 4
n_indicators = 3
n_items = int(latent_size * n_indicators)

deepirtools.manual_seed(123)

loadings = get_loadings(n_indicators, latent_size)
intercepts = get_categorical_intercepts(n_items)
cov_mat = get_covariance_matrix(latent_size, "fixed_variances")
mean = get_mean(latent_size, "fixed_means")[0]

grm = GradedResponseModelSimulator(loadings, intercepts, cov_mat, mean)
data, scores = grm.sample(sample_size, return_scores = True)

np.savetxt("../deepirtools/data/loadings.csv", loadings, delimiter = ",")
np.savetxt("../deepirtools/data/intercepts.csv", intercepts, delimiter = ",")
np.savetxt("../deepirtools/data/cov_mat.csv", cov_mat, delimiter = ",")
np.savetxt("../deepirtools/data/factor_scores.csv", scores, delimiter = ",")
np.savetxt("../deepirtools/data/data.csv", data, delimiter = ",")