import torch
from torch.utils.data import Dataset
import numpy as np
import os
import timeit
from typing import List, Optional

from utils import tensor_dataset


class BaseEstimator():

    def __init__(self,
                 device:                str,
                 log_interval:          int = 100,
                 verbose:               bool = True,
                ):
        """
        Args:
        """
        self.device = device
        self.log_interval = log_interval
        self.verbose = verbose

        self.global_iter = 0
        self.converged = False
        self.loss_list = []
        self.best_avg_loss = None
        self.loss_improvement_counter = 0
        
        self.model = None
        self.optimizer = None
        self.runtime_kwargs = {}

    def loss_function(self):
        raise NotImplementedError

    def check_convergence(self,
                          loss:  float,
                          epoch: int,
                         ):
        cur_mean_loss = None
        
        self.loss_list.append(loss.item())
        if len(self.loss_list) > 100:
            self.loss_list.pop(0)
            
        if (self.global_iter - 1) % 100 == 0 and self.global_iter != 1:
            cur_mean_loss = np.mean(self.loss_list)

            if self.best_avg_loss is None:
                self.best_avg_loss = cur_mean_loss
            elif cur_mean_loss < self.best_avg_loss:
                self.best_avg_loss = cur_mean_loss
                if self.loss_improvement_counter >= 1:
                    self.loss_improvement_counter = 0
            elif cur_mean_loss >= self.best_avg_loss:
                self.loss_improvement_counter += 1
                if self.loss_improvement_counter >= 100:
                    self.converged = True
            if (self.global_iter - 1) % self.log_interval == 0: # issue here -- log_interval multiple/divisble by 100?
                if self.verbose:
                    print("Epoch = {:7d}".format(epoch),
                          "Iter. = {:6d}".format(self.global_iter),
                          "  Current mean loss = {:5.2f}".format(cur_mean_loss),
                          "  Intervals no change = {:3d}".format(self.loss_improvement_counter),
                          end = "\r")
                    
    def step(self,
             batch,
             **model_kwargs,
            ):
        """
        One fitting iteration.
        """
        if self.model.training:
            self.optimizer.zero_grad()
            
        if isinstance(batch, list):
            batch, mask = batch[0], batch[1]
            mask = mask.to(self.device).float()
        else:
            mask = None
        batch =  batch.to(self.device).float() 
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
        """
        Full pass through data set.
        """
        self.model.train()
        train_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            if not self.converged:
                self.global_iter += 1 
                loss = self.step(batch, **model_kwargs)
                
                if torch.isnan(loss):
                    print(("\nNaN loss obtained, ending fitting. "
                           "Consider increasing batch size or reducing learning rate."),
                          end = "\n")
                    self.converged = True
                    break
                    
                self.check_convergence(loss, epoch)
            else:
                break

    @torch.no_grad()
    def test(self,
             test_loader: Dataset,
             **model_kwargs,
            ):
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
        print("\nFitting started", end = "\n")
        start = timeit.default_timer()

        train_loader =  torch.utils.data.DataLoader(
                            tensor_dataset(data=data, mask=missing_mask),
                            batch_size = batch_size, shuffle = True
                        )
        
        epoch = 0
        while not self.converged:
            self.train(train_loader, epoch, **model_kwargs)

            epoch += 1
            if epoch == max_epochs and not self.converged:
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