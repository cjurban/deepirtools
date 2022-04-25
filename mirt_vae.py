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
         input_size:            int,
         inference_model_sizes: List[int],
         latent_size:           int,
         n_cats:                List[int],
         Q,
         A,
         b,
         correlated_factors,
         device:               torch.device):


        """
        Args:
            input_size            (int): Input vector dimension.
            inference_model_sizes (list of int): Inference model neural network layer dimensions.
            latent_size           (int): Latent vector dimension.
            n_cats                (list of int): List containing number of categories for each observed variable.
            device                (str): String specifying whether to run on CPU or GPU.
        """
        super(MIRTVAE, self).__init__()
        
        # Inference model neural network.
        inf_sizes = [input_size] + inference_model_sizes
        inf_list = sum( ([nn.Linear(size1, size2), nn.ELU()] for size1, size2 in
                          zip(inf_sizes[0:-1], inf_sizes[1:])), [] )
        if inf_list != []:
            self.inf_net = nn.Sequential(*inf_list)
        else:
            self.inf_net = nn.Linear(inf_sizes[0], int(2 * latent_size))
        
        self.projector = CatProjector(latent_size, n_cats, Q, A, b)
        
        self.cholesky = Spherical(latent_size, correlated_factors, device)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.inf_net[-1].weight, mean=0., std=0.001)
        nn.init.normal_(self.inf_net[-1].bias[0:self.latent_size], mean=0., std=0.001)
        nn.init.normal_(self.inf_net[-1].bias[self.latent_size:], mean=math.log(math.exp(1) - 1), std=0.001)

    def encode(self,
               x,
               mc_samples,
               iw_samples):
        hidden = self.inf_net(x)

        # Expand for Monte Carlo samples.
        hidden = hidden.unsqueeze(0).expand(torch.Size([mc_samples]) + hidden.shape)
        
        # Expand for importance-weighted samples.
        hidden = hidden.unsqueeze(0).expand(torch.Size([iw_samples]) + hidden.shape)
        
        mu, std = hidden.chunk(chunks = 2, dim = -1)
        std = F.softplus(std)
            
        return mu, std + EPS

    def forward(self,
                x,
                mc_samples = 1,
                iw_samples = 1):
        mu, std = self.encode(x, mc_samples, iw_samples)
        z = mu + std * torch.randn_like(mu)
        recon_x = self.projector(z)
        
        return recon_x, mu, std, z
    
    
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
         steps_anneal:         int = 0,
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
                         device, log_interval, steps_anneal, verbose)
        
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

            
            
def screeplot(latent_sizes:           List[int], # list of dimensions in ascending order
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
            
    for latent_size in latent_sizes:

        print("\rStarting fitting for P =", latent_size, end="")

        # Set random seeds.
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize model.
        start = timeit.default_timer()

        ipip_vae = MIRTVAEClass(input_size = csv_data.shape[1],
                                inference_model_sizes = [int(np.ceil(csv_data.shape[1]/2))], # adjust NN size for different P
                                latent_size = latent_size,
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
        
        
        print("\rModel fitted for P =", latent_size, ", run time =", round(stop - start, 2), "seconds", end = "")

        # Save predicted approximate negative log-likelihood.
        torch.manual_seed(seed)
        np.random.seed(seed)
        start = timeit.default_timer()
        
        bic_result = ipip_vae.bic(csv_test, iw_samples = iw_samples_bic)
        ll_ls.append(bic_result[1]) # I set iw_samples = 5000 in the paper
        stop = timeit.default_timer()
   
        # print("Approx. LL stored, computed in", round(stop - start, 2), "seconds", end="")

    print("\n")
    for idx, dim in enumerate(latent_sizes):
        print("Latent dimension =", dim, "Approx. LL =", round(-ll_ls[idx], 2))

        
    fig, ax = plt.subplots(constrained_layout = True)
    fig.set_size_inches(5, 5, forward = True)
    fig.set_size_inches(5, 5, forward = True)
    
    ax.plot(latent_sizes, [ll for ll in ll_ls], "k-o")
    
    ax.set_xticks(np.arange(min(latent_sizes) - 1, max(latent_sizes) + 2).tolist())
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
