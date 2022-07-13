import pytest
import torch
import deepirtools


deepirtools.manual_seed(1234)
torch.set_default_dtype(torch.float64)
devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")

sample_size = 1000
batch_size = 32


@pytest.mark.parametrize("device", devices)
def test_grm(device):
    

@pytest.mark.parametrize("device", devices)
def test_gpcm(device):
    
    
@pytest.mark.parametrize("device", devices)   
def test_poisson_factor_model(device):

    
@pytest.mark.parametrize("device", devices)    
def test_negative_binomial_factor_model(device):

    
@pytest.mark.parametrize("device", devices)    
def test_normal_factor_model(device):
    

@pytest.mark.parametrize("device", devices)
def test_lognormal_factor_model(device):