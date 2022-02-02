#!/usr/bin/env python
#
# Author: Christopher J. Urban
#
# Purpose: Model and instance class for amortized IWVI for MIRT.
#
###############################################################################

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.distributions as dist
from utils import *
from helper_layers import *
from base_class import BaseClass
from read_data import csv_dataset
import timeit

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from factor_analyzer import Rotator
from pylab import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from typing import List

EPS = 1e-7

    
# Variational autoencoder for MIRT module.
class MIRTVAE(nn.Module):
    
    def __init__(self,
         input_dim:            int,
         inference_model_dims: List[int],
         latent_dim:           int,
         n_cats:               List[int],
         Q,
         A,
         b,
         correlated_factors,
         device:               torch.device):


        """
        Args:
            input_dim            (int): Input vector dimension.
            inference_model_dims (list of int): Inference model neural network layer dimensions.
            latent_dim           (int): Latent vector dimension.
            n_cats               (list of int): List containing number of categories for each observed variable.
            device               (str): String specifying whether to run on CPU or GPU.
        """
        super(MIRTVAE, self).__init__()

        self.input_dim = input_dim
        self.inf_dims = inference_model_dims
        self.latent_dim = latent_dim
        self.n_cats = n_cats
        self.device = device
        self.Q = Q
        self.A = A
        self.b = b
        self.correlated_factors = correlated_factors
        
        # Define inference model neural network.
        if self.inf_dims != []:
            inf_list = []
            inf_list.append(nn.Linear(self.input_dim, self.inf_dims[0]))
            inf_list.append(nn.ELU())
            if len(self.inf_dims) > 1:
                for k in range(len(self.inf_dims) - 1):
                    inf_list.append(nn.Linear(self.inf_dims[k], self.inf_dims[k + 1]))
                    inf_list.append(nn.ELU())
            inf_list.append(nn.Linear(self.inf_dims[len(self.inf_dims) - 1], 
                                      self.latent_dim + self.latent_dim))
            self.inf = nn.Sequential(*inf_list)
        else:
            self.inf = nn.Linear(self.inf_dims[len(self.inf_dims) - 1], 
                                 self.latent_dim + self.latent_dim)
        
        # Define loadings matrix.
        if Q is not None:
            self.loadings = nn.Linear(self.latent_dim, len(self.n_cats), bias = False)
            init_sparse_xavier_uniform_(self.loadings.weight, Q)
        elif self.A is not None:
            self.loadings = LinearConstraints(self.latent_dim, len(self.n_cats), self.A, self.b)
        else:
            self.loadings = nn.Linear(self.latent_dim, len(self.n_cats), bias = False)
            nn.init.xavier_uniform_(self.loadings.weight)
        
        # Define intercept vector.
        self.intercepts = CatBiasReshape(self.n_cats, self.device)
        
        # Define Cholesky decomposition of factor covariance matrix.
        self.cholesky = Spherical(self.latent_dim, self.correlated_factors, self.device)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.inf[-1].weight, mean=0., std=0.001)
        nn.init.normal_(self.inf[-1].bias[0:self.latent_dim], mean=0., std=0.001)
        nn.init.normal_(self.inf[-1].bias[self.latent_dim:], mean=math.log(math.exp(1) - 1), std=0.001)

    def encode(self,
               x,
               mc_samples,
               iw_samples):
        hidden = self.inf(x)

        # Expand for Monte Carlo samples.
        hidden = hidden.unsqueeze(0).expand(torch.Size([mc_samples]) + hidden.shape)
        
        # Expand for importance-weighted samples.
        hidden = hidden.unsqueeze(0).expand(torch.Size([iw_samples]) + hidden.shape)
        
        mu, std = hidden.chunk(chunks = 2, dim = -1)
        std = F.softplus(std)
            
        return mu, std + EPS

    def reparameterize(self,
                       mu,
                       std):
        # Impute factor scores.
        z = mu + std * torch.randn_like(mu)

        return z
        
    def decode(self,
               z):
        Bz = self.loadings(z)
        cum_probs = self.intercepts(Bz.unsqueeze(-1).expand(Bz.shape +
                                                                 torch.Size([max(self.n_cats) - 1]))).sigmoid()
        upper_probs = F.pad(cum_probs, (0, 1), value=1.)
        lower_probs = F.pad(cum_probs, (1, 0), value=0.)
        probs = upper_probs - lower_probs
        
        return probs

    def forward(self,
                x,
                mc_samples = 1,
                iw_samples = 1):
        mu, std = self.encode(x, mc_samples, iw_samples)
        z = self.reparameterize(mu, std)
        recon_x = self.decode(z)
        
        return recon_x, mu, std, z
    
    
# Variational autoencoder for exploratory IFA instance class.
class MIRTVAEClass(BaseClass):
    
    def __init__(self,
         input_dim:            int,
         inference_model_dims: List[int],
         latent_dim:           int,
         n_cats:               List[int],
         learning_rate:        float,
         device:               torch.device,
         log_interval:         int,
         gradient_estimator:   str = "dreg",
         Q                     = None,
         A                     = None,
         b                     = None,
         correlated_factors    = [],
         steps_anneal:         int = 0,
         inf_grad_estimator:   str = "dreg",
         verbose:              bool = False):

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
        super().__init__(input_dim, inference_model_dims, latent_dim, learning_rate,
                         device, log_interval, steps_anneal, verbose)
        
        self.Q = Q
        self.A = A
        self.b = b
        
        self.n_cats = n_cats
        self.grad_estimator = gradient_estimator
        self.correlated_factors = correlated_factors
        self.inf_grad_estimator = inf_grad_estimator
        self.model = MIRTVAE(input_dim = self.input_dim,
                             inference_model_dims = self.inf_dims,
                             latent_dim = self.latent_dim,
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
        
        # Compute ELBO with annealed KL divergence.
        anneal_reg = (linear_annealing(0, 1, self.global_iter, self.steps_anneal)
                      if self.model.training else 1)
        elbo = log_px_z + anneal_reg * (log_qz_x - log_pz)
        
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
                    cross_entropy, kld, annealed_kld = self.elbo_components(data, recon_x, mu, logstd, z, mc_samples, iw_samples)
                    elbo = cross_entropy + annealed_kld
                    
                    # Compute normalized importance weights.
                    elbo = -elbo
                    reweight = (elbo - elbo.logsumexp(dim = 0)).exp()
                    
                    # Conduct sampling-importance-resampling and compute EAPs.
                    iw_idxs = dist.Categorical(probs = reweight.T).sample().reshape(-1)
                    mc_idxs = torch.from_numpy(np.arange(mc_samples)).repeat(data.size(0))
                    batch_idxs = torch.from_numpy(np.arange(data.size(0))).repeat_interleave(mc_samples)
                    scores_ls.append(z[iw_idxs, mc_idxs, batch_idxs, ...].reshape(data.size(0), mc_samples, self.latent_dim).mean(-2))
        
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

            
            
def screeplot(latent_dims:           List[int], # list of dimensions in ascending order
              csv_data:              pd.DataFrame,
              categories:            List[int],
              n_cats:                List[int], 
              which_split:           str, 
              test_size:             float,
              data_loader_kwargs:    dict = {},
              learning_rate:         float = 5e-3,
              device:                torch.device = "cpu",
              log_interval:          int = 100,
              steps_anneal:          int = 1000,
              iw_samples_training:   int = 5,
              iw_samples_bic:        int = 5000,
              xlabel:                str = "Number of Factors",
              ylabel:                str = "Predicted Approximate Negative Log-Likelihood",
              title:                 str = "Approximate Log-Likelihood Scree Plot"):
    seed = 1
    ll_ls = []
    
    csv_train, csv_test = train_test_split(csv_data, train_size = 1 - test_size, test_size = test_size, random_state = 45)
    
    csv_test = pd.DataFrame(csv_test)
            
    for latent_dim in latent_dims:

        print("\rStarting fitting for P =", latent_dim, end="")

        # Set random seeds.
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize model.
        start = timeit.default_timer()

        ipip_vae = MIRTVAEClass(input_dim = csv_data.shape[1],
                                inference_model_dims = [int(np.ceil(csv_data.shape[1]/2))], # adjust NN size for different P
                                latent_dim = latent_dim,
                                n_cats = n_cats,
                                learning_rate = learning_rate,
                                device = device,
                                log_interval = log_interval,
                                steps_anneal = steps_anneal,
                                Q = None,
                                A = None,
                                b = None)

        # Fit model.
        # run training on training set
        
        ipip_vae.run_training(csv_data, categories, iw_samples = iw_samples_training)
        stop = timeit.default_timer()
        
        
        print("\rModel fitted for P =", latent_dim, ", run time =", round(stop - start, 2), "seconds", end = "")

        # Save predicted approximate negative log-likelihood.
        torch.manual_seed(seed)
        np.random.seed(seed)
        start = timeit.default_timer()
        
        bic_result = ipip_vae.bic(csv_test, iw_samples = iw_samples_bic)
        ll_ls.append(bic_result[1]) # I set iw_samples = 5000 in the paper
        stop = timeit.default_timer()
   
        # print("Approx. LL stored, computed in", round(stop - start, 2), "seconds", end="")

    print("\n")
    for idx, dim in enumerate(latent_dims):
        print("Latent dimension =", dim, "Approx. LL =", round(-ll_ls[idx], 2))

        
    fig, ax = plt.subplots(constrained_layout = True)
    fig.set_size_inches(5, 5, forward = True)
    fig.set_size_inches(5, 5, forward = True)
    
    ax.plot(latent_dims, [ll for ll in ll_ls], "k-o")
    
    ax.set_xticks(np.arange(min(latent_dims) - 1, max(latent_dims) + 2).tolist())
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    fig.suptitle(title)
    fig.show()

    pdf = matplotlib.backends.backend_pdf.PdfPages("scree_plot.pdf")
    pdf.savefig(fig, dpi = 300)
    pdf.close()
    return ll_ls, ipip_vae

    
def loadings_heatmap(rot_loadings: [float], 
                     loadings:     [float],
                     x_label:      str = "Factor", 
                     y_label:      str = "Item", 
                     title:        str = "Factor Loadings"):
    norm_loadings = normalize_loadings(rot_loadings)
    c = pcolor(invert_factors(norm_loadings))
    set_cmap("gray_r")
    colorbar() 
    c = pcolor(invert_factors(norm_loadings), edgecolors = "w", linewidths = 1, vmin = 0) 
    xlabel(x_label)
    ylabel(y_label)
    xticks(np.arange(loadings.shape[1]) + 0.5,
           ["1", "2", "3", "4", "5"])
    yticks(np.array([10, 20, 30, 40, 50]) - 0.5, ["10", "20", "30", "40", "50"])
    plt.gca().invert_yaxis()
    suptitle(title, y = 0.93)
    savefig("loadings_heatmap.pdf")
