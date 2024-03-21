import data
import models
import training

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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaForSequenceClassification

import string
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import nlpaug
import nlpaug.augmenter.word as naw
import nlpaug_master.nlpaug.augmenter.sentence as nas


class Augmentation:
    def __init__(self):
        self.syn_aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.2)
        self.sen_aug = nas.RandomSentAug()
        
    def text_augmentation(self, text):
        text = re.sub(r'[\n\t\r]', ' ', text)
        if random.random() > 0.5:
            text = self.clean_text(text)
        if random.random() > 0.5:
            text = self.syn_aug.augment(text)[0]
        if random.random() > 0.5:
            text = self.sen_aug.augment(text)[0]
        return text
            
    def clean_text(self, text):
        # Convert text to lowercase
        text = text.lower()
        # Expand contractions
        text = contractions.fix(text)
        # Remove special characters and numbers using regex
        text = re.sub(r'[^a-zA-Z.?!\s]', '', text)
        # Tokenize the text
        tokens = word_tokenize(text)
        # Remove stopwords
        #stop_words = set(stopwords.words('english'))
        #tokens = [word for word in tokens if word not in stop_words]
        # Lemmatize words
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        # Join the tokens back into a single string
        cleaned_text = ' '.join(tokens)
        return cleaned_text