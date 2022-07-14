import torch
import pyro.distributions as pydist


def generate_loadings(n_indicators: int,
                      latent_size:  int,
                      shrink:       bool = False,
                     ):
    mask = torch.block_diag(*[torch.ones([n_indicators, 1])] * latent_size)
    ldgs_dist = pydist.LogNormal(loc = torch.zeros([n_items, latent_size]),
                                 scale = torch.ones([n_items, latent_size]).mul(0.5))
    ldgs = ldgs_dist.sample() * mask
    if shrink:
        ldgs.mul_(0.3).clamp_(max = 0.7)

    return ldgs


def generate_graded_intercepts(n_items: int,
                               all_same_n_cats: bool = True, 
                              ):
    if all_same_n_cats:
        n_cats = [2] * n_items
    else:
        cats = [2, 3, 4, 5, 6]
        assert(n_items >= len(cats))
        n_cats = cats * (n_items // len(cats)) + cats[:n_items % len(cats)]

    ints = []
    for n_cat in n_cats:
        if n_cat > 2:
            cuts = torch.linspace(-4, 4, n_cat)
            d = 4 / (n_cat - 1)
            ints.append(pydist.Uniform(-d, d).sample([n_cat - 1]) +
                        0.5 * (cuts[1:] + cuts[:-1]))
        else:
            ints.append(pydist.Uniform(-1.5, 1.5).sample([1]))

    return (torch.cat(ints, dim = 0), n_cats)


def generate_non_graded_intercepts(n_items: int,
                                   all_positive: bool = False,
                                  ):
    if all_positive:
        return pydist.Uniform(0.1, 0.5).sample([n_items])
    
    return torch.randn(n_items).mul(0.1)