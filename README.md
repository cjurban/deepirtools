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
Epoch =       78 Iter. =    19701 Cur. loss =   11.09   Intervals no change =  100
Fitting ended in  74.52  seconds

In [7]: model.loadings # Loadings matrix.
Out[7]: 
tensor([[0.8456, 0.0000, 0.0000, 0.0000],
        [0.6506, 0.0000, 0.0000, 0.0000],
        [0.6670, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.6398, 0.0000, 0.0000],
        [0.0000, 1.1146, 0.0000, 0.0000],
        [0.0000, 1.2914, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.8141, 0.0000],
        [0.0000, 0.0000, 1.3549, 0.0000],
        [0.0000, 0.0000, 1.4892, 0.0000],
        [0.0000, 0.0000, 0.0000, 1.0622],
        [0.0000, 0.0000, 0.0000, 1.3446],
        [0.0000, 0.0000, 0.0000, 1.1498]])

In [8]: model.intercepts # Category intercepts.
Out[8]: 
tensor([[-0.1202,  2.4450],
        [-0.9971,  1.8080],
        [-1.1140,  2.0810],
        [-1.7245,  0.0466],
        [-2.2886,  2.5354],
        [-1.5748,  0.2589],
        [-1.2884,  2.8013],
        [-2.8611,  0.4436],
        [-2.1389,  1.6103],
        [-1.8367,  1.3795],
        [-2.7932,  2.1621],
        [-0.1536,  2.4888]])

In [9]: model.cov # Factor covariance matrix.
Out[9]: 
tensor([[ 1.0000, -0.0819,  0.2108,  0.2989],
        [-0.0819,  1.0000, -0.1252, -0.2714],
        [ 0.2108, -0.1252,  1.0000,  0.1322],
        [ 0.2989, -0.2714,  0.1322,  1.0000]])
        
In [10]: model.log_likelihood(data) # Approximate log-likelihood.

Computing approx. LL
Approx. LL computed in 29.01 seconds
Out[8]: -86194.04898071289

In [11]: model.scores(data) # Approximate factor scores.
Out[11]: 
tensor([[-0.9682,  2.0518, -1.3156, -0.8087],
        [-1.0009,  0.4465,  0.3619,  0.9397],
        [-0.2995, -0.2615,  1.5835,  0.5501],
        ...,
        [-0.1856, -2.0889, -0.0113, -0.0937],
        [ 0.4250, -0.4520, -0.8553, -1.5701],
        [-1.9710, -1.0868,  0.6147,  1.4418]])
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
