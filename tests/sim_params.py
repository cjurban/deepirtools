import torch
import pyro.distributions as pydist
import deepirtools


def sim_loadings(n_items:       int,
                 latent_size:   int,
                 n_indicators:  int,
                 loadings_type: str):
    size = torch.Size([n_items, latent_size])
    
    ldgs_list = []
    
    for i in range(3):
        if i == 0:
            mask = torch.block_diag(*[torch.ones([n_indicators, 1])] * latent_size)
        if i == 1:
            mask = torch.block_diag(*[-torch.ones([n_indicators, 1])] * latent_size)
        if i == 2:
            mask = torch.block_diag(*[torch.ones([n_indicators, 1])] * latent_size)
            mask_mul = torch.bernoulli(torch.ones(size).mul(0.5))
            mask_mul[mask_mul == 0] = -1
            mask = mask * mask_mul

        ldgs_dist = pydist.LogNormal(loc = torch.zeros(size),
                                     scale = torch.ones(size).mul(0.5))
        ldgs = ldgs_dist.sample() * mask
        ldgs_list.append(ldgs)