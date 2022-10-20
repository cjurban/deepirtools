.. deepirtools documentation master file, created by
   sphinx-quickstart on Wed Aug 17 14:42:02 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=========================================
Welcome to the DeepIRTools documentation!
=========================================

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
  - Nominal response model

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

To install the latest version:

``pip install deepirtools``

Citation
========

To cite DeepIRTools in publications, use:

* Urban, C. J., & He, S. (2022). DeepIRTools: Deep learning-based estimation and inference for item response theory models. Python package. `https://github.com/cjurban/deepirtools <https://github.com/cjurban/deepirtools>`_

To cite the method, use:

* Urban, C. J., & Bauer, D. J. (2021). A deep learning algorithm for high-dimensional exploratory  item factor analysis. *Psychometrika*, *86* (1), 1--29. `https://link.springer.com/article/10.1007/s11336-021-09748-3 <https://link.springer.com/article/10.1007/s11336-021-09748-3>`_

and/or:

* Urban, C. J. (2021). *Machine learning-based estimation and goodness-of-fit for large-scale confirmatory item factor analysis* (Publication No. 28772217) [Master's thesis, University of North Carolina at Chapel Hill]. ProQuest Dissertations Publishing. `https://www.proquest.com/docview/2618877227/21C6C467D6194C1DPQ/ <https://www.proquest.com/docview/2618877227/21C6C467D6194C1DPQ/>`_

BibTeX entries for LaTeX users are:

.. code:: bibtex

  @Manual{DeepIRTools,
       title = {{D}eep{IRT}ools: {D}eep learning-based estimation and inference for item response theory models},
       author = {Urban, Christopher J. and He, Shara},
       year = {2022},
       note = {Python package},
       url = {https://github.com/cjurban/deepirtools},
  }

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

.. code:: bibtex

  @phdthesis{Urban2021,
      author  = {Urban, Christopher J.},
      title   = {{M}achine learning-based estimation and goodness-of-fit for large-scale confirmatory item factor analysis},
      publisher = {ProQuest Dissertations Publishing},
      school  = {University of North Carolina at Chapel Hill},
      year    = {2021},
      type    = {Master's thesis},
  }

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   deepirtools

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
