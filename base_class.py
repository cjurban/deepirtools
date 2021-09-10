#!/usr/bin/env python
#
# Code author: Christopher J. Urban
#
# Purpose: A base class for VAE-type models with functions for model fitting
# and evaluation as well as saving and loading fitted models.
#
###############################################################################

from __future__ import print_function
import torch
import numpy as np
import os

# The base class.
class BaseClass():

    def __init__(self,
                 input_dim,
                 inference_model_dims,
                 latent_dim,
                 learning_rate,
                 device,
                 log_interval,
                 steps_anneal = 0,
                 verbose = False):
        """
        Args:
            input_dim            (int): Input vector dimension.
            inference_model_dims (list of int): Inference model neural network layer dimensions.
            latent_dim           (int): Latent vector dimension.
            learning_rate        (float): Learning rate for stochastic gradient descent.
            device               (str): String specifying whether to run on CPU or GPU.
            log_interval         (int): Number of batches to wait before printing progress report.
            steps_anneal         (int): Number of batches over which to linearly anneal KL divergence.
            verbose              (Boolean): Print progress reports.
        """
        self.input_dim = input_dim
        self.inf_dims = inference_model_dims
        self.latent_dim = latent_dim
        self.lr = learning_rate
        self.device = device
        self.log_interval = log_interval
        self.steps_anneal = steps_anneal
        self.verbose = verbose
        
        self.global_iter = 0 # Keeps track of number of fitting iterations (i.e., batches).
        self.loss_list = [] # List to mointor loss over batches.
        self.best_avg_loss = None # Keeps track of best average loss.
        self.loss_improvement_counter = 0 # Keeps track of number of iterations since average loss has not improved.
        self.converged = False
        
        # To be implemented by subclasses.
        self.model = None
        self.optimizer = None

    def loss_function(self):
        raise NotImplementedError

    def print_function(self):
        raise NotImplementedError

    # A single fitting iteration.
    def step(self,
             data,
             mc_samples,
             iw_samples):
        if self.model.training:
            self.optimizer.zero_grad()
        output = self.model(data, mc_samples, iw_samples)

        loss = self.loss_function(data, *output, mc_samples, iw_samples)

        if self.model.training and not torch.isnan(loss):
            loss.backward()
            self.optimizer.step()

        return loss

    # Fit for one epoch.
    def train(self,
              train_loader,
              eval_loader,
              epoch,
              mc_samples,
              iw_samples):
        
        # Switch to training mode.
        self.model.train()
        train_loss = 0

        for batch_idx, data in enumerate(train_loader):
            if not self.converged:
                self.global_iter += 1
                data = data.to(self.device).float()    
                loss = self.step(data, mc_samples, iw_samples)
                
                if torch.isnan(loss):
                    print(("NaN loss obtained, ending fitting. "
                           "Consider increasing batch size or reducing learning rate."))
                    self.converged = True
                    break
                    
                # Append to loss list.
                if self.global_iter >= self.steps_anneal:
                    self.loss_list.append(loss.item())
                    if len(self.loss_list) > 100:
                        self.loss_list.pop(0)

                    # Determine whether to terminate fitting.
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

                if (self.global_iter - 1) % self.log_interval == 0:
                    if self.verbose:
                        try:
                            self.print_function(eval_loader, eval_loss, epoch = epoch,
                                                batch = data, batch_idx = batch_idx, train_loader = train_loader)
                        except UnboundLocalError:
                            self.print_function(train_loader, loss, epoch = epoch,
                                                batch = data, batch_idx = batch_idx)

            else:
                break

    # Evaluate the model.
    def test(self,
             eval_loader,
             mc_samples,
             iw_samples,
             print_result = True):
        # Switch to evaluation mode.
        self.model.eval()
        eval_loss = 0

        with torch.no_grad():
            for data in eval_loader:
                data = data.to(self.device).float()
                loss = self.step(data, mc_samples, iw_samples)
                eval_loss += loss.item()
        
        if print_result:
            self.print_function(eval_loader, eval_loss)
            self.model.train()
        else:
            self.model.train()
            return eval_loss

    # Fit the model.
    def run_training(self,
                     train_loader,
                     eval_loader,
                     max_epochs = 3000,
                     mc_samples = 1,
                     iw_samples = 1,
                     log_likelihood = False):
        for epoch in range(max_epochs):
            if not self.converged:
                self.train(train_loader, eval_loader, epoch, mc_samples, iw_samples)

                if self.verbose:
                    self.test(eval_loader, mc_samples, iw_samples)

    # Save the model.
    def save_model(self,
                   model_name,
                   save_path):
        with torch.no_grad():
            torch.save(self.model.state_dict(), 
                       os.path.join(save_path, model_name) + ".pth")

    # Load a model.
    def load_model(self,
                   model_name,
                   load_path):
        self.model.load_state_dict(torch.load(os.path.join(load_path, model_name) + ".pth"))