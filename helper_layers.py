#!/usr/bin/env python
#
# Author: Christopher J. Urban
#
# Purpose: Helpful layers for building models.
#
###############################################################################

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils import *

class CatBiasReshape(nn.Module):
    
    def __init__(self,
                 n_cats,
                 device):
        super(CatBiasReshape, self).__init__()
        self.n_cats = n_cats
        
        # Construct biases.
        bias = torch.empty(sum([n_cat - 1 for n_cat in self.n_cats]))
        bias.data = torch.from_numpy(np.hstack([logistic_thresholds(n_cat) for 
                                                n_cat in self.n_cats]))
        idxs = np.cumsum([n_cat - 1 for n_cat in ([1] + self.n_cats)])
        sliced_bias = [bias[idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]
        self.bias_reshape = nn.Parameter(torch.cat([F.pad(_slice,
                                                          (0, max(n_cats) - _slice.size(0) - 1),
                                                          value=9999.).unsqueeze(0) for
                                                    _slice in sliced_bias], axis = 0))
        
        # Construct drop indices.
        nan_mask = torch.cat([F.pad(_slice, (0, max(self.n_cats) - _slice.size(0) - 1),
                                    value=float("nan")).unsqueeze(0) for
                                    _slice in sliced_bias], axis = 0)
        cum_probs_mask = nan_mask * 0. + 1.
#        probs_mask = F.pad(cum_probs_mask, (1, 0), value=1.)
        self.drop_idxs = ~cum_probs_mask.view(-1).isnan().to(device)

    def forward(self, x):
        return self.bias_reshape + x
    
    @property
    def bias(self):
        return self.bias_reshape.view(-1)[self.drop_idxs]