# DeepIRTools

DeepIRTools is a small Python package that uses scalable deep learning methods (e.g., [Urban and Bauer, 2021](https://link.springer.com/article/10.1007/s11336-021-09748-3)) to fit several kinds of exploratory and confirmatory latent factors models, with a particular focus on item response theory (IRT) models.

## Description

Latent factor models reduce the dimensionality of data by converting a large number of discrete or continuous observed variables (called *items*) into a smaller number of continuous unobserved variables (called *latent factors*), potentially making the data easier to understand. Latent factors models for discrete items are called *item response theory* (IRT) models.

Traditional maximum likelihood (ML) estimation methods for IRT models are computationally intensive when the sample size, the number of items, and the number of latent factors are all large. This issue can be avoided by approximating the ML estimator using an *importance-weighted amortized variational estimator* (I-WAVE) from the field of deep learning ([Burda, Grosse, and Salakhutdinov, 2016](https://arxiv.org/abs/1509.00519); [Tucker, Lawson, Gu, and Maddison, 2019](https://arxiv.org/abs/1810.04152)). As an estimation byproduct, I-WAVE allows users to compute approximate factor scores and log-likelihoods for any observation &mdash; even observations that were not used for model fitting.

DeepIRTools' main functionality is the stand-alone ``ImportanceWeightedEstimator`` class contained in the  ``importance_weighted`` module. This class includes ``fit()``, ``scores()``, and ``log_likelihood`` methods for fitting a latent factor model and for computing approximate factor scores and log-likelihoods for the fitted model.

The following (multidimensional) latent factor models are currently available:

    1. Graded response model
    2. Generalized partial credit model
    3. Normal (linear) factor model
    4. Lognormal factor model

## Requirements

## Installation

## Examples

### Sample Code

### Tutorial

[`examples/big_5_tutorial.ipynb`](examples/big_5_tutorial.ipynb) gives a tutorial on using DeepIRTools to fit several kinds of latent factor models using large-scale data. 

## References

[Burda, Y., Grosse, R. & Salakhutdinov, R. (2016). Importance weighted autoencoders. In 4th International Conference on Learning Representations. ICLR.](https://arxiv.org/abs/1509.00519)

[Tucker, G., Lawson, D., Gu, S., & Maddison, C. J. (2019). Doubly reparameterized gradient estimators for Monte Carlo objectives. In 7th International Conference on Learning Representations. ICLR.](https://arxiv.org/abs/1810.04152)

[Urban, C. J., & Bauer, D. J. (2021). A deep learning algorithm for high-dimensional exploratory item factor analysis. Psychometrika, 86(1), 1-29.](https://link.springer.com/article/10.1007/s11336-021-09748-3)
