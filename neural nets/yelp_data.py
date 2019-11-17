# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 22:00:41 2019

@author: saiharshith
"""
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SequentialSampler
from torch.autograd import Variable

# Define your custom dataset class to be loaded by pytorch's dataloader

class YelpDataset(Dataset):
    def __init__(self, vocab, data, set_train=None):
        self.train_data, self.test_data = train_test_split(data, train_size=0.70, 
                                                 test_size=0.3, random_state=0)
        if set_train:
            self.sentence = self.train_data['sentence'].tolist()
            self.sentiment = self.train_data['sentiment'].tolist()
        
        else:
            self.sentence = self.test_data['sentence'].tolist()
            self.sentiment = self.test_data['sentiment'].tolist()
        
        self.vocab = vocab
        
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data_idx = {'sentence': [], 'sentiment': self.category[idx]}
        for i, word in enumerate(self.category[idx].split()):
            data_idx['sentiment'].append(self.vocab.get_idx(word))
    
        return data_idx  # return index rather than word
    
    def test(self, idx):
        return self.__getitem__(idx)
        
