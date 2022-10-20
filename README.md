<h1 align='center'>DeepIRTools</h1>
<h2 align='center'>Deep Learning-Based Estimation and Inference for Item Response Theory Models</h2>

<div align="center">

[![PyPI version](https://badge.fury.io/py/deepirtools.svg)](https://badge.fury.io/py/deepirtools)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Documentation Status](https://readthedocs.org/projects/deepirtools/badge/?version=latest)](https://deepirtools.readthedocs.io/en/latest/?badge=latest)
[![python_sup](https://img.shields.io/badge/python-3.7+-black.svg?)](https://www.python.org/downloads/release/python-370/)

DeepIRTools is a small Pytorch-based Python package that uses scalable deep learning methods to fit a number of different confirmatory and exploratory latent factor models, with a particular focus on item response theory (IRT) models. Graphics processing unit (GPU) support is available for most computations.

</div>

## Description

Latent factor models reduce the dimensionality of data by converting a large number of discrete or continuous observed variables (called *items*) into a smaller number of continuous unobserved variables (called *latent factors*), potentially making the data easier to understand. Latent factor models for discrete items are called *item response theory* (IRT) models.

Traditional maximum likelihood (ML) estimation methods for IRT models are computationally intensive when the sample size, the number of items, and the number of latent factors are all large. This issue can be avoided by approximating the ML estimator using an *importance-weighted amortized variational estimator* (I-WAVE) from the field of deep learning (for details, see [Urban and Bauer, 2021](https://link.springer.com/article/10.1007/s11336-021-09748-3)). As an estimation byproduct, I-WAVE allows researchers to compute approximate factor scores and log-likelihoods for any observation &mdash; even new observations that were not used for model fitting.

DeepIRTools' main functionality is the stand-alone ``IWAVE`` class contained in the  ``iwave`` module. This class includes ``fit()``, ``scores()``, and ``log_likelihood()`` methods for fitting a latent factor model and for computing approximate factor scores and log-likelihoods for the fitted model.

The following (multidimensional) latent factor models are currently available...

- ...for binary and ordinal items:
  - Graded response model
  - Generalized partial credit model
  - Nominal response model
- ...for continuous items:
  - Normal (linear) factor model
  - Lognormal factor model
- ...for count data:
  - Poisson factor model
  - Negative binomial factor model

DeepIRTools supports mixing item types, handling missing completely at random data, and predicting the mean of the latent factors with covariates (i.e., latent regression modeling); all models are estimable in both confirmatory and exploratory contexts. In the confirmatory context, constraints on the factor loadings, intercepts, and factor covariance matrix are implemented by providing appropriate arguments to ``fit()``. In the exploratory context, the ``screeplot()`` function in the ``figures`` module may help identify the number of latent factors underlying the data.

## Requirements

-  Python 3.7 or higher
-  ``torch``
-  ``pyro-ppl``
-  ``numpy``

## Installation

To install the latest version:

``pip install deepirtools``

## Documentation

Official documentation is available [here](https://deepirtools.readthedocs.io/en/latest/).

## Examples

### Tutorial

See [`big_5_tutorial.ipynb`](https://github.com/cjurban/deepirtools/blob/master/examples/big_5_tutorial.ipynb) for a tutorial on using DeepIRTools to fit several kinds of latent factor models using large-scale data.

### Demonstration

See [`mnist_demo.ipynb`](https://github.com/cjurban/deepirtools/blob/master/examples/mnist_demo.ipynb) for a demonstration of how DeepIRTools may be used to fit a flexible and identifiable model for generating realistic handwritten digits.

### Quick Example

```python
In [1]: import deepirtools
   ...: from deepirtools import IWAVE
   ...: import torch

In [2]: deepirtools.manual_seed(123)

In [3]: data = deepirtools.load_grm()["data"]

In [4]: n_items = data.shape[1]

In [5]: model = IWAVE(
   ...:       model_type = "grm",
   ...:       latent_size = 4,
   ...:       n_cats = [3] * n_items,
   ...:       Q = torch.block_diag(*[torch.ones([3, 1])] * 4),
   ...:       correlated_factors = [0, 1, 2, 3],
   ...: )

Initializing model parameters
Initialization ended in  0.0  seconds

In [6]: model.fit(data, iw_samples = 10)

Fitting started
Epoch =      100 Iter. =    25201 Cur. loss =   10.68   Intervals no change =  100
Fitting ended in  109.23  seconds

In [7]: model.loadings # Loadings matrix.
Out[7]: 
tensor([[0.8295, 0.0000, 0.0000, 0.0000],
        [0.5793, 0.0000, 0.0000, 0.0000],
        [0.7116, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.7005, 0.0000, 0.0000],
        [0.0000, 1.1687, 0.0000, 0.0000],
        [0.0000, 1.2890, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.9268, 0.0000],
        [0.0000, 0.0000, 1.2653, 0.0000],
        [0.0000, 0.0000, 1.5622, 0.0000],
        [0.0000, 0.0000, 0.0000, 1.0346],
        [0.0000, 0.0000, 0.0000, 1.3641],
        [0.0000, 0.0000, 0.0000, 1.1348]])

In [8]: model.intercepts # Category intercepts.
Out[8]: 
tensor([[ 2.4245, -0.1637],
        [ 1.8219, -1.0013],
        [ 2.0811, -1.1320],
        [ 0.0948, -1.7253],
        [ 2.6597, -2.3412],
        [ 0.2610, -1.4938],
        [ 2.8196, -1.3281],
        [ 0.4833, -2.8053],
        [ 1.6395, -2.2220],
        [ 1.3482, -1.8870],
        [ 2.1606, -2.8600],
        [ 2.5318, -0.1333]])

In [9]: model.cov # Factor covariance matrix.
Out[9]: 
tensor([[ 1.0000, -0.0737,  0.2130,  0.2993],
        [-0.0737,  1.0000, -0.1206, -0.3031],
        [ 0.2130, -0.1206,  1.0000,  0.1190],
        [ 0.2993, -0.3031,  0.1190,  1.0000]])
        
In [10]: model.log_likelihood(data) # Approximate log-likelihood.

Computing approx. LL
Approx. LL computed in 33.27 seconds
Out[6]: -85961.69088745117

In [11]: model.scores(data) # Approximate factor scores.
Out[11]: 
tensor([[-0.6211,  0.1301, -0.7207, -0.7485],
        [ 0.2189, -0.2649,  0.0109, -0.2363],
        [ 0.0544,  0.9308,  0.7940, -0.8851],
        ...,
        [-0.2964, -0.9597, -0.8885, -0.0057],
        [-1.6015,  0.9812,  0.0486,  0.1773],
        [ 2.0448,  0.0583,  1.2005, -0.9317]])
```

## Citation

To cite DeepIRTools in publications, use:

* Urban, C. J., & He, S. (2022). DeepIRTools: Deep learning-based estimation and inference for item response theory models. Python package. [https://github.com/cjurban/deepirtools](https://github.com/cjurban/deepirtools)

To cite the method, use:

  * Urban, C. J., & Bauer, D. J. (2021). A deep learning algorithm for high-dimensional exploratory  item factor analysis. Psychometrika, 86(1), 1-29. [https://link.springer.com/article/10.1007/s11336-021-09748-3](https://link.springer.com/article/10.1007/s11336-021-09748-3)
  
  and/or:
  
  * Urban, C. J. (2021). *Machine learning-based estimation and goodness-of-fit for large-scale confirmatory item factor analysis* (Publication No. 28772217) [Master's thesis, University of North Carolina at Chapel Hill]. ProQuest Dissertations Publishing. [https://www.proquest.com/docview/2618877227/21C6C467D6194C1DPQ/](https://www.proquest.com/docview/2618877227/21C6C467D6194C1DPQ/)

BibTeX entries for LaTeX users are:
```bibtex
@Manual{DeepIRTools,
title = {{D}eep{IRT}ools: {D}eep learning-based estimation and inference for item response theory models},
     author = {Urban, Christopher J. and He, Shara},
     year = {2022},
     note = {Python package},
     url = {https://github.com/cjurban/deepirtools},
}
```

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

```bibtex
@phdthesis{Urban2021,
    author  = {Urban, Christopher J.},
    title   = {{M}achine learning-based estimation and goodness-of-fit for large-scale confirmatory item factor analysis},
    publisher = {ProQuest Dissertations Publishing},
    school  = {University of North Carolina at Chapel Hill},
    year    = {2021},
    type    = {Master's thesis},
}
```
