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
import timeit

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from factor_analyzer import Rotator
from pylab import *


EPS = 1e-7

    
# Variational autoencoder for MIRT module.
class MIRTVAE(nn.Module):
    
    def __init__(self,
                 input_dim,
                 inference_model_dims,
                 latent_dim,
                 n_cats,
                 device):
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
        self.loadings = nn.Linear(self.latent_dim, len(self.n_cats), bias = False)
        nn.init.xavier_uniform_(self.loadings.weight)
        
        # Define intercept vector.
        self.intercepts = CatBiasReshape(self.n_cats, self.device)
        
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
                 input_dim,
                 inference_model_dims,
                 latent_dim,
                 n_cats,
                 learning_rate,
                 device,
                 log_interval,
                 gradient_estimator = "dreg",
                 steps_anneal = 0,
                 verbose = False):
        """
        New args:
            n_cats             (list of int): List containing number of categories for each observed variable.
            gradient_estimator (str): Inference model gradient estimator.
                                      "iwae" = IWAE, "dreg" = DReG.
        """
        super().__init__(input_dim, inference_model_dims, latent_dim, learning_rate,
                         device, log_interval, steps_anneal, verbose)
        
        self.n_cats = n_cats
        self.grad_estimator = gradient_estimator
        self.model = MIRTVAE(input_dim = self.input_dim,
                             inference_model_dims = self.inf_dims,
                             latent_dim = self.latent_dim,
                             n_cats = self.n_cats,
                             device = self.device).to(self.device)
        self.optimizer = optim.Adam([{"params" : self.model.parameters()}],
                                    lr = self.lr,
                                    amsgrad = True)
        
                 
    # Compute loss for one batch.
    def loss_function(self,
                      x,
                      recon_x,
                      mu,
                      logstd,
                      z,
                      mc_samples,
                      iw_samples):
        return self.log_likelihood(x, recon_x, mu, logstd, z, mc_samples, iw_samples)
        
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

        return loss
    
    # Return unrotated loadings
    def get_unrotated_loadings(self):
        return self.model.loadings.weight.data.numpy()
    
    # Return intercepts
    def get_intercepts(self):
        return self.model.intercepts.bias.data.numpy()
    
    # Compute ELBO components.
    def elbo_components(self,
                        x,
                        recon_x,
                        mu,
                        std,
                        z,
                        mc_samples,
                        iw_samples):
        # Compute cross-entropy (i.e., log p(x | z)).
        idxs = x.long().expand(recon_x[..., -1].shape).unsqueeze(-1)
        cross_entropy = -torch.gather(recon_x, dim = -1, index = idxs).squeeze(-1).clamp(min = EPS).log().sum(dim = -1, keepdim = True)
        
        # Compute log p(z).
        log_pz = (-0.5 * z.pow(2) - math.log(math.sqrt(2 * math.pi))).sum(dim = -1, keepdim = True)
            
        # Compute log q(z | x).
        if self.model.training and iw_samples > 1 and self.grad_estimator == "dreg":
            mu_, std_ = mu.detach(), std.detach()
            qz_x = dist.Normal(mu_, std_)
        else:
            qz_x = dist.Normal(mu, std)
        log_qz_x = qz_x.log_prob(z).sum(dim = -1, keepdim = True)
        
        # Compute KL divergence.
        kld = log_qz_x - log_pz
            
        # Anneal KL divergence.
        anneal_reg = (linear_annealing(0, 1, self.global_iter, self.steps_anneal)
                      if self.model.training else 1)
        annealed_kld = anneal_reg * kld

        return cross_entropy, kld, annealed_kld
    
    # Approximate marginal log-likelihood of the observed data.
    def log_likelihood(self,
                       x,
                       recon_x,
                       mu,
                       std,
                       z,
                       mc_samples,
                       iw_samples):
        # Compute ELBO components.
        cross_entropy, kld, annealed_kld = self.elbo_components(x, recon_x, mu, std, z, mc_samples, iw_samples)
        elbo = cross_entropy + annealed_kld
        
        if iw_samples == 1:
            elbo = elbo.squeeze(0).mean(0)
            if self.model.training:
                return elbo.mean()
            else:
                return elbo.sum()
        elif self.grad_estimator == "iwae":
            elbo = -elbo
            iw_elbo = math.log(elbo.size(0)) - elbo.logsumexp(dim = 0)
                    
            if self.model.training:
                return iw_elbo.mean()
            else:
                return iw_elbo.mean(0).sum()
        elif self.grad_estimator == "dreg":
            elbo = -elbo
            with torch.no_grad():
                # Compute re-weighting factor for inference model gradient.
                reweight = (elbo - elbo.logsumexp(dim = 0)).exp()
                
                if z.requires_grad:
                    z.register_hook(lambda grad: reweight * grad)
            
            if self.model.training:
                return (-reweight * elbo).sum(0).mean()
            else:
                return (-reweight * elbo).sum()
        
    # Compute pseudo-BIC.
    def bic(self,
            eval_loader,
            iw_samples = 1):
        print("\nComputing approx. LL")
        start = timeit.default_timer()
        # Get size of data set.
        N = len(eval_loader.dataset)
        
        # Switch to IWAE bound.
        old_estimator = self.grad_estimator
        self.grad_estimator = "iwae"
        
        # Approximate marginal log-likelihood.
        ll = self.test(eval_loader,
                       mc_samples =  1,
                       iw_samples = iw_samples,
                       print_result = False)
        
        # Switch back to previous bound.
        self.grad_estimator = old_estimator
        
        # Get number of estimated parameters.
        n_params = self.model.loadings.weight.data.numel() + self.model.intercepts.bias.data.numel()
            
        # Compute BIC.
        bic = 2 * ll + np.log(N) * n_params
        stop = timeit.default_timer()
        print("Approx. LL computed in", round(stop - start, 2), "seconds")

        return bic, -ll, n_params
    
    # Compute EAP estimates of factor scores.
    def scores(self,
               eval_loader,
               mc_samples = 1,
               iw_samples = 1):
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
            
            
def screeplot(latent_dims, # list of dimensions in ascending order
                            n_cats, 
                            ipip_train_loader, 
                            ipip_test_loader,
                            learning_rate = 5e-3,
                            device = "cpu",
                            log_interval = 100,
                            steps_anneal = 1000,
                            iw_samples_training = 5,
                            iw_samples_bic = 5000,
                            xlabel = "Number of Factors",
                            ylabel = "Predicted Approximate Negative Log-Likelihood",
                            title = "Approximate Log-Likelihood Scree Plot"):
    seed = 1
    ll_ls = []
    for latent_dim in latent_dims:

        print("\nStarting fitting for P =", latent_dim)

        # Set random seeds.
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize model.
        start = timeit.default_timer()
        ipip_vae = MIRTVAEClass(input_dim = ipip_train_loader.dataset.num_columns(),
                                inference_model_dims = [int(np.ceil(ipip_train_loader.dataset.num_columns()/2))], # adjust NN size for different P
                                latent_dim = latent_dim,
                                n_cats = n_cats,
                                learning_rate = learning_rate,
                                device = device,
                                log_interval = log_interval,
                                steps_anneal = steps_anneal)

        # Fit model.
        ipip_vae.run_training(ipip_train_loader, ipip_test_loader, iw_samples = iw_samples_training)
        stop = timeit.default_timer()

        print("Model fitted, run time =", round(stop - start, 2), "seconds")

        # Save predicted approximate negative log-likelihood.
        torch.manual_seed(seed)
        np.random.seed(seed)
        start = timeit.default_timer()
        ll_ls.append(-ipip_vae.bic(ipip_test_loader, iw_samples = iw_samples_bic)[1]) # I set iw_samples = 5000 in the paper
        stop = timeit.default_timer()

        print("Approx. LL stored, computed in", round(stop - start, 2), "seconds")

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
    return ll_ls

def get_rotated_loadings(loadings, method):
    start = timeit.default_timer()
    rotator = Rotator(method = method)
    rot_loadings = rotator.fit_transform(loadings)
    
    stop = timeit.default_timer()
    print("Rotated loadings computed in", round(stop - start, 2), "seconds")
    return rot_loadings, rotator.phi_

    
def loadings_heatmap(rot_loadings, loadings, x_label = "Factor", y_label = "Item", title = "Factor Loadings"):
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
