#!/usr/bin/env python
#
# Author: Christopher J. Urban
#
# Purpose: Some functions for loading data sets.
#
###############################################################################

from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
    
# Read in a data set from a CSV file.
class csv_dataset(Dataset):
    def __init__(self,
                 data,
                 which_split,
                 csv_header = 0,
                 categories = None,
                 test_size = .25,
                 val_size = None):
        """
        Args:
            csv_file    (string): Path to the CSV file.
            which_split (string): Return the training set, the validation set, or the test set.
                                  "full" = full data set;
                                  "train-only" = split data into train and test set, then return only train samples;
                                  "test-only" = split data into train and test set, then return only test samples;
                                  "train" = split data into train, val., and test sets, then return only train samples;
                                  "test" = split data into train, val., and test sets, then return only test samples;
                                  "val" = split data into train, val., and test sets, then return only val. samples.
            test_size   (int): Proportion of data to include in test set. Must be specified which_split is set to something
                               other than "full".
            val_size    (int): Proportion of data to include in validation set. Must be specified when which_split is set to
                               "train", "test", or "val".
            transform   (Transform): Tranformation of the output samples.
        """
        self.which_split = which_split
        self.transform = to_tensor()
        self.df = data
        
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # data is formatted as an numpy array: 
        # sample = self.df.iloc[idx, :].to_numpy()
        sample = self.df.iloc[idx, :].to_numpy()
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def num_columns(self):
        return self.df.shape[1]
    
# Convert Numpy arrays in sample to Tensors.
class to_tensor(object):
    def __call__(self, sample):
        return torch.from_numpy(sample)