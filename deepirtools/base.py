import torch
from torch.utils.data import Dataset
import numpy as np
import os
import timeit
import warnings
from typing import List, Optional
from deepirtools.utils import ConvergenceChecker, tensor_dataset


class OptimizationWarning(UserWarning):
    pass


class BaseEstimator():
    """Base class from which other estimation methods inherit.
    
    Includes methods to fit new models as well as to save and load fitted models.
    
    Parameters
    __________
    device : str, default = "cpu"
        Computing device used for fitting.
    log_interval : int, default = 100
        Number of mini-batches between printed updates during fitting.
    verbose : bool, default = True
        Whether to print updates during fitting.
    n_intervals : int, default = 100
        Number of 100-mini-batch intervals after which fitting is terminated if
        best average loss does not improve.
    
    Attributes
    __________
    device : str
        Computing device used for fitting.
    verbose : bool
        Whether to print updates during fitting.
    global_iter : int
        Number of mini-batches processed during fitting.
    timerecords : dict
        Stores run times for various processes (e.g., fitting).
    """

    def __init__(self,
                 device:                str = "cpu",
                 log_interval:          int = 100,
                 verbose:               bool = True,
                 n_intervals:           int = 100,
                ):
        self.device = device
        self.verbose = verbose

        self.global_iter = 0
        self.checker = ConvergenceChecker(log_interval = log_interval, n_intervals = n_intervals)
        
        self.model = None
        self.optimizer = None
        self.runtime_kwargs = {}
        self.timerecords = {}

    def loss_function(self):
        raise NotImplementedError
                    
    def step(self,
             batch,
             **model_kwargs,
            ):
        """One fitting iteration."""
        
        if self.model.training:
            self.optimizer.zero_grad()
            
        batch = {k : v.to(self.device).float() if v is not None else v for k, v in batch.items()}
        output = self.model(**batch, **model_kwargs, **self.runtime_kwargs)
        loss = self.loss_function(*output)

        if self.model.training and not torch.isnan(loss):
            loss.backward()
            self.optimizer.step()

        return loss
    
    def train(self,
              train_loader: Dataset,
              epoch:        int,
              **model_kwargs,
             ):
        """Full pass through data set."""
        
        self.model.train()
        train_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            if not self.checker.converged:
                self.global_iter += 1 
                loss = self.step(batch, **model_kwargs)
                
                if torch.isnan(loss):
                    warnings.warn(("NaN loss obtained, ending fitting. "
                                   "Consider increasing batch size or reducing learning rate."),
                                  OptimizationWarning)
                    self.checker.converged = True
                    break
                    
                self.checker.check_convergence(epoch, self.global_iter, loss.item())
            else:
                break

    @torch.no_grad()
    def test(self,
             test_loader: Dataset,
             **model_kwargs,
            ):
        """Evaluate model on a data set."""
        
        self.model.eval()
        test_loss = 0
        
        for batch in test_loader:
            loss = self.step(batch, **model_kwargs)
            test_loss += loss.item()
        
        self.model.train()
            
        return test_loss

    def fit(self,
            data:           torch.Tensor,
            batch_size:     int = 32,
            missing_mask:   Optional[torch.Tensor] = None,
            covariates:     Optional[torch.Tensor] = None,
            max_epochs:     int = 100000,
            **model_kwargs,
           ):
        r"""Fit model to a data set.

        Parameters
        __________
        data : Tensor
            Data set.

            A :math:`\text{sample_size} \times \text{n_items}` matrix.
        batch_size : int, default = 32
            Mini-batch size for stochastic gradient optimizer.
        missing_mask : Tensor, default = None
            Binary mask indicating missing item responses.

            A :math:`\text{sample_size} \times \text{n_items}` matrix.
        covariates : Tensor, default = None
            Matrix of covariates.

            A :math:`\text{sample_size} \times \text{covariate_size}` matrix .
        max_epochs : int, default = 100000
            Number of passes through the full data set after which fitting should be
            terminated if convergence not achieved.
        mc_samples : int, default = 1
            Number of Monte Carlo samples.

            Increasing this decreases the log-likelihood estimator's variance.
        iw_samples : int, default = 5000
            Number of importance-weighted samples.

            Increasing this decreases the log-likelihood estimator's bias.
        """
        
        if self.verbose:
            print("\nFitting started", end = "\n")
        start = timeit.default_timer()

        train_loader =  torch.utils.data.DataLoader(
                            tensor_dataset(data=data, mask=missing_mask,
                                           covariates = covariates),
                            batch_size = batch_size, shuffle = True,
                            pin_memory = self.device == "cuda",
                        )
        
        epoch = 0
        while not self.checker.converged:
            self.train(train_loader, epoch, **model_kwargs)

            epoch += 1
            if epoch == max_epochs and not self.checker.converged:
                warnings.warn(("Failed to converge within " + str(max_epochs) + " epochs."), OptimizationWarning)
                break
                
        stop = timeit.default_timer()
        self.timerecords["fit"] = stop - start
        if self.verbose:
            print("\nFitting ended in ", round(stop - start, 2), " seconds", end = "\n")
        
    def save_model(self,
                   model_name: str,
                   save_path:  str,
                  ):
        """Save fitted model.
        
        Parameters
        __________
        model_name : str
            Name for fitted model.
        save_path : str
            Where to save fitted model.
        """
        
        with torch.no_grad():
            torch.save(self.model.state_dict(), 
                       os.path.join(save_path, model_name) + ".pth")

    def load_model(self,
                   model_name: str,
                   load_path: str,
                  ):
        """Load fitted model.
        
        The initialized model should have the same hyperparameter settings as the
        fitted model that is being loaded (e.g., the same number of latent variables).
        
        Parameters
        __________
        model_name : str
            Name of fitted model.
        load_path : str
            Where to load fitted model from.
        """
        
        self.model.load_state_dict(torch.load(os.path.join(load_path, model_name) + ".pth"))