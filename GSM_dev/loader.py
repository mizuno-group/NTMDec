# -*- coding: utf-8 -*-
"""
Created on 2023-07-06 (Thu) 18:47:50

dataloader

@author: I.Azuma
"""
#%%
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

import torch

#%%
class DataLoader(object):

    def __init__(self, data, bow_vocab, batch_size, device, shuffle=True):
        
        self.batch_size = batch_size
        self.bow_vocab = bow_vocab
        self.device = device
        
        self.index = 0
        self.pointer = np.array(range(len(data)))
        
        self.data = np.array(pd.DataFrame(data)) # FIXME: avoid error
        self.bow_data = np.array(pd.DataFrame([bow_vocab.doc2bow(s) for s in data])) # FIXME: avoid error
        
        # counting total word number
        word_count = []
        for bow in self.bow_data:
            wc = 0
            try:
                for (i, c) in bow:
                    wc += c
            except:
                pass
            word_count.append(wc)
        
        self.word_count = sum(word_count)
        self.data_size = len(data)
        
        self.shuffle = shuffle
        self.reset()

    
    def reset(self):
        
        if self.shuffle:
            self.pointer = shuffle(self.pointer)
        
        self.index = 0 
    
    
    # transform bow data into (1 x V) size vector.
    def _pad(self, batch):
        bow_vocab = len(self.bow_vocab)
        res_src_bow = np.zeros((len(batch), bow_vocab))
        
        for idx, bow in enumerate(batch):
            try:
                bow_k = [k for k, v in bow]
                bow_v = [v for k, v in bow]
                res_src_bow[idx, bow_k] = bow_v
            except:
                pass
            
        return res_src_bow
    
    def __iter__(self):
        return self

    def __next__(self):
        
        if self.index >= self.data_size:
            self.reset()
            raise StopIteration()
            
        ids = self.pointer[self.index: self.index + self.batch_size]
        batch = self.bow_data[ids]
        padded = self._pad(batch)
        tensor = torch.tensor(padded, dtype=torch.float, device=self.device)
        
        self.index += self.batch_size

        return tensor
    
    # for NTM.lasy_predict()
    def bow_and_text(self):
        if self.index >= self.data_size:
            self.reset()
            
        text = self.data[self.index: self.index + self.batch_size]
        batch = self.bow_data[self.index: self.index + self.batch_size]
        padded = self._pad(batch)
        tensor = torch.tensor(padded, dtype=torch.float, device=self.device)
        self.reset()

        return tensor, text