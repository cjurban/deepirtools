===========
DeepIRTools
===========

----------------------------------------------------------------------------
Deep Learning-Based Estimation and Inference for Item Response Theory Models
----------------------------------------------------------------------------

DeepIRTools is a small Pytorch-based Python package that uses scalable deep learning methods to fit a number of different confirmatory and exploratory latent factors models, with a particular focus on item response theory (IRT) models. Graphics processing unit (GPU) support is available to speed up some computations.

Description
===========

Latent factor models reduce the dimensionality of data by converting a large number of discrete or continuous observed variables (called *items*) into a smaller number of continuous unobserved variables (called *latent factors*), potentially making the data easier to understand. Latent factor models for discrete items are called *item response theory* (IRT) models.

Traditional maximum likelihood (ML) estimation methods for IRT models are computationally intensive when the sample size, the number of items, and the number of latent factors are all large. This issue can be avoided by approximating the ML estimator using an *importance-weighted amortized variational estimator* (I-WAVE) from the field of deep learning (for details, see `Urban and Bauer, 2021 <https://link.springer.com/article/10.1007/s11336-021-09748-3>`_). As an estimation byproduct, I-WAVE allows researchers to compute approximate factor scores and log-likelihoods for any observation -- even new observations that were not used for model fitting.

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

Requirements
============

-  Python 3.7 or higher
-  ``torch``
-  ``pyro-ppl``
-  ``numpy``

Installation
============

To install the latest stable version:

``pip install deepirtools``

To install the latest version on GitHub:

``pip install git+https://github.com/cjurban/deepirtools``

Documentation
=============

Examples
========

Tutorial
--------

`examples/big_5_tutorial.ipynb <examples/big_5_tutorial.ipynb>`_ gives a tutorial on using DeepIRTools to fit several kinds of latent factor models using large-scale data.

Quick Example
-------------

.. code:: python

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
  
  In [6]: model.fit(data, iw_samples = 5)
  
  Fitting started
  Epoch =     846 Iter. =  27101 Cur. loss =   11.15   Intervals no change = 100
  Fitting ended in  95.14  seconds
  
  In [7]: model.loadings # Loadings matrix.
  Out[7]: 
  tensor([[1.4004, 0.0000, 0.0000, 0.0000],
          [1.3816, 0.0000, 0.0000, 0.0000],
          [0.5557, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.5833, 0.0000, 0.0000],
          [0.0000, 1.0996, 0.0000, 0.0000],
          [0.0000, 1.7175, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.7294, 0.0000],
          [0.0000, 0.0000, 0.5775, 0.0000],
          [0.0000, 0.0000, 1.1082, 0.0000],
          [0.0000, 0.0000, 0.0000, 1.6827],
          [0.0000, 0.0000, 0.0000, 0.7021],
          [0.0000, 0.0000, 0.0000, 0.6706]])
  
  In [8]: model.intercepts # Category intercepts.
  Out[8]: 
  tensor([[-1.2907,  1.4794],
          [-0.6921,  1.2275],
          [-0.4097,  0.3086],
          [-2.0435,  1.3194],
          [-2.8560,  1.0286],
          [-0.2557,  1.9871],
          [-1.6538,  0.6874],
          [-0.4569,  0.8666],
          [-1.2310,  1.7704],
          [-1.1810,  0.2015],
          [-0.6825,  2.5192],
          [-2.8031,  2.7023]])
  
  In [9]: model.cov # Factor covariance matrix.
  Out[9]: 
  tensor([[1.0000, 0.1679, 0.1489, 0.2227],
          [0.1679, 1.0000, 0.1406, 0.2248],
          [0.1489, 0.1406, 1.0000, 0.1452],
          [0.2227, 0.2248, 0.1452, 1.0000]])

Citation
========

To cite DeepIRTools in publications, use:

* Urban, C. J., & He, S. (2022). DeepIRTools: Deep learning-based estimation and inference for item response theory models. Python package. `https://github.com/cjurban/deepirtools <https://github.com/cjurban/deepirtools>`_

To cite the method, use:

* Urban, C. J., & Bauer, D. J. (2021). A deep learning algorithm for high-dimensional exploratory  item factor analysis. Psychometrika, 86(1), 1-29. `https://link.springer.com/article/10.1007/s11336-021-09748-3 <https://link.springer.com/article/10.1007/s11336-021-09748-3>`_

BibTeX entries for LaTeX users are:

.. code:: bibtex

  @Manual{DeepIRTools,
       title = {{D}eep{IRT}ools: {D}eep learning-based estimation and inference for item response theory models},
       author = {Urban, Christopher J. and He, Shara},
       year = {2022},
       note = {Python package},
       url = {https://github.com/cjurban/deepirtools},
  }

and:

.. code:: bibtex

  @article{UrbanBauer2021,
      author = {Urban, Christopher J. and Bauer, Daniel J.},
      year={2021},
      title={{A} deep learning algorithm for high-dimensional exploratory item factor analysis},
      journal = {Psychometrika},
      volume = {86},
      number = {1},
      pages = {1--29}
  }
