import numpy as np
import data

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


class Training:
    def __init__(self, name, frozen_model = None, device="cpu"):
        self.min_loss = float("inf")
        self.max_accuracy = float("-inf")
        self.name = name
        self.frozen_model = frozen_model
        
        self.device = device
        
        if frozen_model:
            self.frozen_model.to(self.device)
            self.frozen_model.eval()
            
    def training_cycle(self, model, optimizer, criterion, training_loader, tokenizer):
        model.train()
        for i, (x, y, z) in enumerate(training_loader):
            optimizer.zero_grad()
            x = tokenizer(x,
                          padding='longest', 
                          truncation=True,
                          return_tensors='pt'
                         )
            x = {key: value.squeeze(1).to(self.device) for key, value in x.items()}
            
            if self.frozen_model:
                with torch.no_grad():
                    rep = self.frozen_model(**x).logits
                #print(rep.shape)
                pred = model(rep)
            else:
                pred = model(**x).logits
            
            #print(pred.shape, y.shape)
            loss = criterion(pred, y.to(self.device))
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                loss, current = loss.item(), (i + 1) * len(y)
                print(f"loss: {loss:>7f}  [{current:>5d}/{len(training_loader.dataset):>5d}]")
                
                
    def testing_cycle(self, model, criterion, testing_loader, tokenizer):
        total_loss, accuracy = 0, 0
        
        model.eval()
        with torch.no_grad():
            for i, (x, y, z) in enumerate(testing_loader):
                y = y.to(self.device)
                
                x = tokenizer(x,
                          padding='longest', 
                          truncation=True,
                          return_tensors='pt'
                         )
                x = {key: value.squeeze(1).to(self.device) for key, value in x.items()}
                if self.frozen_model:
                    rep = self.frozen_model(**x).logits
                    pred = model(rep)
                else:
                    pred = model(**x).logits
    
                total_loss += criterion(pred, y)
                accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()
                #print(f"batch {i} is complete")
                
        total_loss /= len(testing_loader)
        accuracy /= len(testing_loader.dataset)
        
        return total_loss, accuracy

    
    def complete_epoch(self, epochs, model, optimizer, scheduler, criterion, tokenizer, training_loader, testing_loader=None):
        losses = {'train': [], 'test': []}

        for epoch in range(epochs):
            self.training_cycle(model, optimizer, criterion, training_loader, tokenizer)
            
            training_loss, training_accuracy =  self.testing_cycle(model, criterion, training_loader, tokenizer)
            print(f"Epoch {epoch+1} has training_loss={training_loss}, training_accuracy={training_accuracy}")
            losses['train'].append(training_loss)
            
            if testing_loader:
                testing_loss, testing_accuracy =  self.testing_cycle(model, criterion, testing_loader, tokenizer)
                print(f"Epoch {epoch+1} has testing_loss={testing_loss}, testing_accuracy={testing_accuracy}")
            else:
                testing_loss = training_loss
                testing_accuracy = training_accuracy
            
            losses['test'].append(testing_loss)
            
            if epoch % 20 == 0:
                print(f"learning rate at epoch {epoch} = {optimizer.param_groups[0]['lr']}")
            
            
            if testing_loss < self.min_loss:
                self.min_loss = testing_loss
                # Save the model checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': testing_loss,
                }
                torch.save(checkpoint, f'{self.name}_low_loss_model_checkpoint.pth')
                
            if testing_accuracy > self.max_accuracy:
                self.max_accuracy = testing_accuracy
                # Save the model checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    #'optimizer_state_dict': optimizer.state_dict(),
                    #'best_loss': testing_loss,
                }
                torch.save(checkpoint, f'{self.name}_high_acc_model_checkpoint.pth')
                
            scheduler.step(training_loss)

        
        last_checkpoint = {
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }
        torch.save(last_checkpoint, f'{self.name}_last_model_checkpoint.pth')
        
        return losses
    
    
    def inference_from_path(self, model, tokenizer, test_path, checkpoint=None, file_name=""):
        results = pd.DataFrame(columns=["ID", "label"])
        
        inference_dataset = data.DatasetClass(test_path, mode='all')
        dataloader = DataLoader(inference_dataset, batch_size=64, shuffle=False)
        
        if checkpoint is not None:
            model.to("cpu").load_state_dict(torch.load(checkpoint)['model_state_dict'])
        
        model.to(self.device)
        model.eval()
        with torch.no_grad():
            for x, y, z in dataloader:
                x = tokenizer(x,
                          padding='longest', 
                          truncation=True,
                          return_tensors='pt'
                         )
                x = {key: value.squeeze(1).to(self.device) for key, value in x.items()}
                if self.frozen_model:
                    rep = self.frozen_model(**x).logits
                    pred = model(rep)
                else:
                    pred = model(**x).logits
                
                pred = pred.argmax(1).type(torch.int) + 1
                #print(z.shape)
                #print(pred.shape)
                
                temp = pd.DataFrame({"ID": z.cpu(), "label": pred.cpu()})
                results = pd.concat([results, temp], axis=0, ignore_index=True)
                
        results.to_csv(f"{self.name}_predictions_{file_name}.csv", index=False)
        return results