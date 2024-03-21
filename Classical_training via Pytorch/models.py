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


class ClassificationHead(nn.Module):
    def __init__(self, logits=3):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(768, 768), 
                                    nn.ReLU(),
                                    nn.Dropout(p=0.1),
                                    nn.Linear(768, 128), 
                                    nn.ReLU(),
                                    nn.Dropout(p=0.1),
                                    nn.Linear(128, logits)
                                   )
    def forward(self, x):
        return self.layers(x)