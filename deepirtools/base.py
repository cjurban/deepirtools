import torch
from torch.utils.data import Dataset
import numpy as np
import os
import timeit
from typing import List, Optional
from deepirtools.utils import ConvergenceChecker, tensor_dataset


class BaseEstimator():

    def __init__(self,
                 device:                str,
                 log_interval:          int = 100,
                 verbose:               bool = True,
                ):
        """
        Base class from which other estimation methods inherit.
        
        Args:
            device       (str):  Computing device used for fitting.
            log_interval (str):  Frequency of updates printed during fitting.
            verbose      (bool): Whether to print updates during fitting.
        """
        self.device = device
        self.verbose = verbose

        self.global_iter = 0
        self.checker = ConvergenceChecker(log_interval = log_interval)
        
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
            
        if isinstance(batch, list):
            batch, mask = batch[0], batch[1]
            mask = mask.to(self.device)
        else:
            mask = None
        batch =  batch.to(self.device)
        output = self.model(batch, mask=mask, **model_kwargs, **self.runtime_kwargs)
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
                    print(("\nNaN loss obtained, ending fitting. "
                           "Consider increasing batch size or reducing learning rate."),
                          end = "\n")
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
            max_epochs:     int = 100000,
            **model_kwargs,
           ):
        """
        Fit model to a data set.
        
        Args:
            data         (Tensor): Data set containing item responses.
            batch_size   (int):    Mini-batch size for stochastic gradient optimizer.
            missing_mask (Tensor): Binary mask indicating missing item responses.
            max_epochs   (int):    Number of passes through the full data set after which
                                   fitting should be terminated if convergence not achieved.
            model_kwargs (dict):   Named parameters passed to self.model.forward().
        """
        print("\nFitting started", end = "\n")
        start = timeit.default_timer()

        train_loader =  torch.utils.data.DataLoader(
                            tensor_dataset(data=data, mask=missing_mask),
                            batch_size = batch_size, shuffle = True,
                            pin_memory = self.device == "cuda",
                        )
        
        epoch = 0
        while not self.checker.converged:
            self.train(train_loader, epoch, **model_kwargs)

            epoch += 1
            if epoch == max_epochs and not self.checker.converged:
                print("\nFailed to converge within " + str(max_epochs) + " epochs.")
                break
                
        stop = timeit.default_timer()
        self.timerecords["fit"] = stop - start
        print("\nFitting ended in ", round(stop - start, 2), " seconds", end = "\n")
        
    def save_model(self,
                   model_name: str,
                   save_path:  str,
                  ):
        with torch.no_grad():
            torch.save(self.model.state_dict(), 
                       os.path.join(save_path, model_name) + ".pth")

    def load_model(self,
                   model_name: str,
                   load_path: str,
                  ):
        self.model.load_state_dict(torch.load(os.path.join(load_path, model_name) + ".pth"))