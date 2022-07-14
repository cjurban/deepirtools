import os
from os.path import join
import pytest
import torch
import deepirtools
from . import simulators as sim


DATA_DIR = os.path.join("tests", "data")
EXPECTED_DIR = os.path.join("tests", "expected")


deepirtools.manual_seed(1234)
torch.set_default_dtype(torch.float64)
devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")

sample_size = 1000
batch_size = 32
n_indicators = 4
latent_size = 4
n_items = int(n_indicators * latent_size)


@pytest.mark.parametrize("device", devices)
def test_latent_factor_model(device):