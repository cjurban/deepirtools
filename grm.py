#!/usr/bin/env python
#
# Purpose: 
#
###############################################################################

import torch
import numpy as np
import os
from read_data import tensor_dataset
import pandas as pd
import timeit

  
# Variational autoencoder for exploratory IFA instance class.
class MIRTVAEClass(BaseClass):
    
    def __init__(self,
         input_size:            int,
         inference_model_sizes: List[int],
         latent_size:           int,
         n_cats:               List[int],
         learning_rate:        float,
         device:               torch.device,
         log_interval:         int,
         gradient_estimator:   str = "dreg",
         Q                     = None,
         A                     = None,
         b                     = None,
         correlated_factors    = [],
         inf_grad_estimator:   str = "dreg",
         verbose:              bool = True):

        """
        New args:
            n_cats             (list of int): List containing number of categories for each observed variable.
            gradient_estimator (str): Inference model gradient estimator.
                                      "iwae" = IWAE, "dreg" = DReG.
            Q                     (Tensor): Matrix with binary entries indicating measurement structure.
            A                     (Tensor): Matrix implementing linear constraints.
            b                     (Tensor): Vector implementing linear constraints.
            correlated_factors    correlated_factors    (list of int): List of correlated factors.
        """
        super().__init__(input_size, inference_model_sizes, latent_size, learning_rate,
                         device, log_interval, verbose)
        
        self.Q = Q
        self.A = A
        self.b = b
        
        self.n_cats = n_cats
        self.grad_estimator = gradient_estimator
        self.correlated_factors = correlated_factors
        self.inf_grad_estimator = inf_grad_estimator
        self.model = MIRTVAE(input_size = self.input_size,
                             inference_model_sizes = self.inf_sizes,
                             latent_size = self.latent_size,
                             n_cats = self.n_cats,
                             Q = self.Q,
                             A = self.A,
                             b = self.b,
                             device = self.device,
                             correlated_factors = self.correlated_factors).to(self.device)
        self.optimizer = optim.Adam([{"params" : self.model.parameters()}],
                                    lr = self.lr,
                                    amsgrad = True)
        self.timerecords = dict.fromkeys(["Fitted Model", "Log-Likelihood" , "Rotated Loadings"])
        
                
    # Compute loss for one batch.
    def loss_function(self,
                      x,
                      recon_x,
                      mu,
                      std,
                      z,
                      mc_samples,
                      iw_samples):
        
        # Compute log p(x | z).        
        idxs = x.long().expand(recon_x[..., -1].shape).unsqueeze(-1)
        log_px_z = -(torch.gather(recon_x, dim = -1, index = idxs).squeeze(-1)).clamp(min = EPS).log().sum(dim = -1, keepdim = True)
        
        # Compute log p(z).
        if self.correlated_factors != []:
            log_pz = dist.MultivariateNormal(torch.zeros_like(z).to(self.device),
                                             scale_tril = self.model.cholesky.weight).log_prob(z).unsqueeze(-1)
            
        else:
            log_pz = dist.Normal(torch.zeros_like(z).to(self.device),
                                 torch.ones_like(z).to(self.device)).log_prob(z).sum(-1, keepdim = True)
            
        # Compute log q(z | x).
        if self.model.training and iw_samples > 1 and self.inf_grad_estimator == "dreg":
            mu_, std_ = mu.detach(), std.detach()
            qz_x = dist.Normal(mu_, std_)
        else:
            qz_x = dist.Normal(mu, std)
        log_qz_x = qz_x.log_prob(z).sum(dim = -1, keepdim = True)
        
        # Compute ELBO.
        elbo = log_px_z + log_qz_x - log_pz
        
        # Compute ELBO.
        if iw_samples == 1:
            elbo = elbo.squeeze(0).mean(0)
            if self.model.training:
                return elbo.mean()
            else:
                return elbo.sum()

        # Compute IW-ELBO.
        elif self.inf_grad_estimator == "iwae":
            elbo *= -1
            iw_elbo = math.log(elbo.size(0)) - elbo.logsumexp(dim = 0)
                    
            if self.model.training:
                return iw_elbo.mean()
            else:
                return iw_elbo.mean(0).sum()

        # Compute IW-ELBO with DReG estimator.
        elif self.inf_grad_estimator == "dreg":
            elbo *= -1
            with torch.no_grad():
                # Compute normalized importance weights.
                w_tilda = (elbo - elbo.logsumexp(dim = 0)).exp()
                
                if z.requires_grad:
                    z.register_hook(lambda grad: w_tilda * grad)
            
            if self.model.training:
                return (-w_tilda * elbo).sum(0).mean()
            else:
                return (-w_tilda * elbo).sum()
        
    # Fit for one epoch.
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
            
            # Set fixed loadings to zero.
            if self.model.Q is not None:
                self.model.loadings.weight.data.mul_(self.model.Q)

        return loss
    
    @property
    # Return unrotated loadings
    def get_unrotated_loadings(self):
        return self.model.loadings.weight.data.numpy()
    
    @property
    # Return intercepts
    def get_intercepts(self):
        return self.model.intercepts.bias.data.numpy()
    
    @property
    # Return dictionary with all time records for running functions
    def get_time_records(self):

        return self.timerecords
        
    # Compute pseudo-BIC.
    def bic(self,
            csv_test:           pd.DataFrame,
            iw_samples:         int = 1,
            data_loader_kwargs: dict = {}):
        eval_loader = torch.utils.data.DataLoader(
        csv_dataset(data = csv_test.reset_index(drop = True), 
                    which_split = "full"),
        batch_size = 32, shuffle = True, **data_loader_kwargs)
        
    
        print("\nComputing approx. LL", end="")
        start = timeit.default_timer()
        # Get size of data set.
        N = len(eval_loader.dataset)
        
        # Switch to IWAE bound.
        old_estimator = self.grad_estimator
        self.grad_estimator = "iwae"
        
        # Approximate marginal log-likelihood.
        ll = self.test(eval_loader,
                       mc_samples =  1,
                       iw_samples = iw_samples)
        
        # Switch back to previous bound.
        self.grad_estimator = old_estimator
        
        # Get number of estimated parameters.
        n_params = self.model.loadings.weight.data.numel() + self.model.intercepts.bias.data.numel()
            
        # Compute BIC.
        bic = 2 * ll + np.log(N) * n_params
        stop = timeit.default_timer()
        self.timerecords["Log-Likelihood"] = round(stop - start, 2)
        print("\nApprox. LL computed in", round(stop - start, 2), "seconds\n", end = "")

        return [bic, -ll, n_params]
    
    # Compute EAP estimates of factor scores.
    def scores(self,
               csv_test,
               mc_samples = 1,
               iw_samples = 1):
        
        eval_loader = torch.utils.data.DataLoader(
        csv_dataset(data = csv_test.reset_index(drop = True), 
                    which_split = "full"),
        batch_size = 32, shuffle = True, **data_loader_kwargs)
        # Switch to evaluation mode.
        
        self.model.eval()
        
        scores_ls = []

        with torch.no_grad():
            for data in eval_loader:
                data = data.to(self.device).float()
                recon_x, mu, logstd, z = self.model(data, mc_samples, iw_samples)
                                                    
                if iw_samples == 1:
                    scores_ls.append(mu.mean(1).squeeze())
                else:
                    # Compute ELBO components.
                    cross_entropy, kld = self.elbo_components(data, recon_x, mu, logstd, z, mc_samples, iw_samples)
                    elbo = cross_entropy + kld
                    
                    # Compute normalized importance weights.
                    elbo = -elbo
                    reweight = (elbo - elbo.logsumexp(dim = 0)).exp()
                    
                    # Conduct sampling-importance-resampling and compute EAPs.
                    iw_idxs = dist.Categorical(probs = reweight.T).sample().reshape(-1)
                    mc_idxs = torch.from_numpy(np.arange(mc_samples)).repeat(data.size(0))
                    batch_idxs = torch.from_numpy(np.arange(data.size(0))).repeat_interleave(mc_samples)
                    scores_ls.append(z[iw_idxs, mc_idxs, batch_idxs, ...].reshape(data.size(0), mc_samples, self.latent_size).mean(-2))
        
        self.model.train()                                    
        return torch.cat(scores_ls, dim = 0)
    
    def get_rotated_loadings(self,
                             loadings, 
                             method, 
                             loadings_only = True):
    
        start = timeit.default_timer()
        rotator = Rotator(method = method)
        rot_loadings = rotator.fit_transform(loadings)

        stop = timeit.default_timer()
        self.timerecords["Rotated Loadings"] = round(stop - start, 2)

        print("\rRotated loadings computed in", round(stop - start, 2), "seconds", end = "")
        sys.stdout.flush()
        if loadings_only: 
            return rot_loadings
        else:
            return rot_loadings, rotator.phi_, rotator.rotation_