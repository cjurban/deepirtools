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

```python
In [1]: import torch
   ...: from deepirtools import IWAVE, load_grm

In [2]: data = load_grm()["data"]

In [3]: n_items = data.shape[1]

In [4]: model = IWAVE(
   ...:      learning_rate = 1e-2,
   ...:      model_type = "grm",
   ...:      Q = torch.block_diag(*[torch.ones([3, 1])] * 4),
   ...:      input_size = n_items,
   ...:      inference_net_sizes = [100],
   ...:      latent_size = 4,
   ...:      n_cats = [3] * n_items,
   ...:      correlated_factors = [0, 1, 2, 3],
   ...: )

Initializing model parameters
Initialization ended in  0.0  seconds

In [5]: model.fit(data, iw_samples = 5)

Fitting started
Epoch =     750 Iter. =  24001 Cur. loss =   11.08   Intervals no change = 100
Fitting ended in  83.68  seconds

In [6]: model.loadings
Out[6]: 
tensor([[1.3244, 0.0000, 0.0000, 0.0000],
        [1.4748, 0.0000, 0.0000, 0.0000],
        [0.5382, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.6075, 0.0000, 0.0000],
        [0.0000, 1.2025, 0.0000, 0.0000],
        [0.0000, 1.6361, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.7412, 0.0000],
        [0.0000, 0.0000, 0.6334, 0.0000],
        [0.0000, 0.0000, 0.9824, 0.0000],
        [0.0000, 0.0000, 0.0000, 1.7153],
        [0.0000, 0.0000, 0.0000, 0.7477],
        [0.0000, 0.0000, 0.0000, 0.7211]])

In [7]: model.intercepts
Out[7]: 
tensor([[-1.2395,  1.4840],
        [-0.6968,  1.2617],
        [-0.4047,  0.3069],
        [-2.0498,  1.3285],
        [-2.8727,  1.0828],
        [-0.2089,  1.8970],
        [-1.6389,  0.6855],
        [-0.4799,  0.9020],
        [-1.1862,  1.7222],
        [-1.1674,  0.2377],
        [-0.6638,  2.5533],
        [-2.8514,  2.7459]])

In [8]: model.cov
Out[8]: 
tensor([[1.0000, 0.1653, 0.1868, 0.1920],
        [0.1653, 1.0000, 0.1767, 0.1846],
        [0.1868, 0.1767, 1.0000, 0.1410],
        [0.1920, 0.1846, 0.1410, 1.0000]])
```

### Tutorial

[`examples/big_5_tutorial.ipynb`](examples/big_5_tutorial.ipynb) gives a tutorial on using DeepIRTools to fit several kinds of latent factor models using large-scale data.

## Documentation

## Citation

To cite DeepIRTools in publications, use:

* Urban, C. J., & He, S. (2022). DeepIRTools: Deep learning-based estimation and inference for item response theory models. Python package. [https://github.com/cjurban/deepirtools](https://github.com/cjurban/deepirtools)

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
