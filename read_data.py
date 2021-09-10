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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
    
# Read in a data set from a CSV file.
class csv_dataset(Dataset):
    def __init__(self,
                 csv_file,
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

        csv_data = pd.read_csv(csv_file, sep = ",", header = csv_header)
        
        if self.which_split == "full":
            self.df = csv_data
            
        elif self.which_split == "train-only" or self.which_split == "test-only":
            # Split the data into a training set and a test set.
            csv_train, csv_test = train_test_split(csv_data, train_size = 1 - test_size, test_size = test_size, random_state = 45)
            
            if self.which_split == "train-only":
                self.df = csv_train
            elif self.which_split == "test-only":
                self.df = csv_test
            
        else:
            # Split the data into a training set, a validation set, and a test set.
            csv_train, csv_test = train_test_split(csv_data, train_size = 1 - test_size, test_size = test_size, random_state = 45)
            csv_train, csv_val = train_test_split(csv_train, train_size = 1 - val_size, test_size = val_size, random_state = 50)

            if self.which_split == "train":
                self.df = csv_train
            elif self.which_split == "val":
                self.df = csv_val
            elif self.which_split == "test":
                self.df = csv_test
        
        # Convert the integer data into one hot encoded data
        enc = OneHotEncoder(categories = categories)
        self.df = enc.fit_transform(self.df).toarray()
        
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # data is formatted as an numpy array: 
        # sample = self.df.iloc[idx, :].to_numpy()
        sample = self.df[idx, :]
        if self.transform:
            sample = self.transform(sample)

        return sample
    def num_columns(self):
        return self.df.shape[1]
# Convert Numpy arrays in sample to Tensors.
class to_tensor(object):
    def __call__(self, sample):
        return torch.from_numpy(sample)