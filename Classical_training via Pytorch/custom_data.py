import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import random
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import string
import contractions
import nltk
from nltk.tokenize import word_tokenize



class CustomDatasetClass(Dataset):
    def __init__(self, x_file, is_labeled=False, mode='all', training_ratio=0.9, x_transform=None, labels=None, y_transform=None):
        
        self.x_file = x_file
        self.data=pd.read_csv(x_file)
        self.labels = labels
        
        if self.labels:
            self.data = self.data[self.data["stars"].isin(labels)]
      
        if mode=="train":
            self.data = self.data[:int(len(self.data)*training_ratio)].reset_index(drop=True)
        elif mode=="test":
            self.data = self.data[int(len(self.data)*training_ratio):].reset_index(drop=True)
        else:
            pass
        
        self.is_labeled = is_labeled
        self.x_transform = x_transform
        self.y_transform = y_transform

    def __getitem__(self, idx):
        x = self.data["text"][idx]
        y = -1
        index = self.data["ID"][idx]
        
        if self.is_labeled:
            y = self.data["stars"][idx]
        
        if self.x_transform:
            x = self.x_transform(x)
            
        if self.y_transform:
            y = self.y_transform(y)

        return x, torch.tensor(y), index
    
    def __len__(self):
        return len(self.data)