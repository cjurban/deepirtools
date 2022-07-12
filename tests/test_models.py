import torch
from torch import nn
import torch.nn.functional as F
import pyro.distributions as pydist
from typing import List
    
    
class GRMTest(nn.Module):
    
    def __init__(self,
                 loadings: torch.Tensor,
                 intercepts: torch.Tensor,
                 n_cats: List[int],
                ):
        """Samejima's graded response model."""
        super(GRMTest, self).__init__()
        
        self.loadings = loadings
        idxs = np.cumsum([n_cat - 1 for n_cat in ([1] + n_cats)])
        self.sliced_ints = [intercepts[i:j] for i, j in zip(idxs[:-1], idxs[1:])]

    @torch.no_grad()
    def sample(self,
                x: torch.Tensor,
               ):
        n_items = self.loadings.shape[1]
        
        Bx = F.linear(x, self.loadings)
        cum_probs = torch.cat([Bx + ints.unsqueeze(0) for ints in self.sliced_ints])
        upper_probs = F.pad(cum_probs, (0, 1), value = 1.)
        lower_probs = F.pad(cum_probs, (1, 0), value = 0.)
        probs = upper_probs - lower_probs
        
        