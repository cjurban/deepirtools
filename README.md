<h1 align='center'>DeepIRTools</h1>
<h2 align='center'>Deep Learning-Based Estimation and Inference for Item Response Theory Models</h2>

DeepIRTools is a small Python package that uses scalable deep learning methods to fit several kinds of confirmatory and exploratory latent factors models, with a particular focus on item response theory (IRT) models. Graphics processing unit (GPU) support is available to parallelize some computations.

## Description

Latent factor models reduce the dimensionality of data by converting a large number of discrete or continuous observed variables (called *items*) into a smaller number of continuous unobserved variables (called *latent factors*), potentially making the data easier to understand. Latent factor models for discrete items are called *item response theory* (IRT) models.

Traditional maximum likelihood (ML) estimation methods for IRT models are computationally intensive when the sample size, the number of items, and the number of latent factors are all large. This issue can be avoided by approximating the ML estimator using an *importance-weighted amortized variational estimator* (I-WAVE) from the field of deep learning (for details, see [Urban and Bauer, 2021](https://link.springer.com/article/10.1007/s11336-021-09748-3)). As an estimation byproduct, I-WAVE allows researchers to compute approximate factor scores and log-likelihoods for any observation &mdash; even new observations that were not used for model fitting.

DeepIRTools' main functionality is the stand-alone ``IWAVE`` class contained in the  ``iwave`` module. This class includes ``fit()``, ``scores()``, and ``log_likelihood()`` methods for fitting a latent factor model and for computing approximate factor scores and log-likelihoods for the fitted model.

The following (multidimensional) latent factor models are currently available...

- ...for binary and ordinal items:
  - Graded response model
  - Generalized partial credit model
- ...for continuous items:
  - Normal (linear) factor model
  - Lognormal factor model
- ...for count data:
  - Poisson factor model
  - Negative binomial factor model

All models are estimable in both confirmatory and exploratory contexts. In the confirmatory context, constraints on the factor loadings, intercepts, and factor covariance matrix are implemented by providing appropriate arguments to ``fit()``. In the exploratory context, the ``screeplot()`` function in the ``figures`` module may help identify the number of latent factors underlying the data.

## Requirements

-  Python 3.6 or higher
-  ``torch``
-  ``pyro-ppl``
-  ``numpy``

## Installation

To install the latest stable version:

``pip install deepirtools``

To install the latest version on GitHub:

``pip install git+https://github.com/cjurban/deepirtools``

## Examples

### Sample Code

### Tutorial

[`examples/big_5_tutorial.ipynb`](examples/big_5_tutorial.ipynb) gives a tutorial on using DeepIRTools to fit several kinds of latent factor models using large-scale data.

## Documentation

## Citation

To cite DeepIRTools in publications, use:

* Urban, C. J., & He, S. (2022). DeepIRTools: Deep learning-based estimation and inference for item response theory models. Python package version 1.0.0. [https://github.com/cjurban/deepirtools](https://github.com/cjurban/deepirtools)

To cite the method, use:

  * Urban, C. J., & Bauer, D. J. (2021). A deep learning algorithm for high-dimensional exploratory  item factor analysis. Psychometrika, 86(1), 1-29. [https://link.springer.com/article/10.1007/s11336-021-09748-3](https://link.springer.com/article/10.1007/s11336-021-09748-3)

BibTeX entries for LaTeX users are:
```bibtex
@Manual{DeepIRTools,
title = {{D}eep{IRT}ools: {D}eep learning-based estimation and inference for item response theory models},
     author = {Urban, Christopher J. and He, Shara},
     year = {2022},
     note = {Python package version 1.0.0},
     url = {https://github.com/cjurban/deepirtools},
}
```
and:
```bibtex
@article{UrbanBauer2021,
    author = {Urban, Christopher J. and Bauer, Daniel J.},
    year={2021},
    title={{A} deep learning algorithm for high-dimensional exploratory item factor analysis},
    journal = {Psychometrika},
    volume = {86},
    number = {1},
    pages = {1--29}
}
```
