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
        self.intercepts = intercepts
        self.n_cats = n_cats

    def forward(self,
                x: torch.Tensor,
               ):
        Bx = F.linear(x, self.loadings)
        