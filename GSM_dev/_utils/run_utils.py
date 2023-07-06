# -*- coding: utf-8 -*-
"""
Created on 2023-07-06 (Thu) 18:59:09

@author: I.Azuma
"""
#%%
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F

#%%
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def l1_penalty(para):
    return nn.L1Loss()(para, torch.zeros_like(para))


def check_sparsity(para, sparsity_threshold=1e-3):
    num_weights = para.shape[0] * para.shape[1]
    num_zero = (para.abs() < sparsity_threshold).sum().float()
    return num_zero / float(num_weights)


def update_l1(cur_l1, cur_sparsity, sparsity_target):
    diff = sparsity_target - cur_sparsity
    cur_l1.mul_(2.0 ** diff)

def compute_loss(model, dataloader, optimizer, epoch, target_sparsity, device):
    
    model.train()
    train_loss = 0
    
    for batch_idx, data_bow in enumerate(dataloader):
        data_bow = data_bow.to(device)
        
        # normalize data
        data_bow_norm = F.normalize(data_bow)
        optimizer.zero_grad()
        
        z, g, recon_batch, mu, logvar = model(data_bow_norm)
        
        loss = loss_function(recon_batch, data_bow, mu, logvar)
        loss = loss + model.l1_strength * l1_penalty(model.fcd1.weight)
        loss.backward()
        
        train_loss += loss.item()
        optimizer.step()
    
    sparsity = check_sparsity(model.fcd1.weight.data)
    print("Overall sparsity = %.3f, l1 strength = %.5f" % (sparsity, model.l1_strength))
    print("Target sparsity = %.3f" % target_sparsity)
    update_l1(model.l1_strength, sparsity, target_sparsity)
    
    avg_loss = train_loss / len(dataloader.data)
    
    print('Train epoch: {} Average loss: {:.4f}'.format(
        epoch, avg_loss))
    
    return sparsity, avg_loss

def compute_test_loss(model, dataloader, epoch, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data_bow in enumerate(dataloader):
            data_bow = data_bow.to(device)
            data_bow_norm = F.normalize(data_bow)

            _, _, recon_batch, mu, logvar = model(data_bow_norm)
            test_loss += loss_function(recon_batch, data_bow, mu, logvar).item()

    avg_loss = test_loss / len(dataloader.data)
    print('Test epoch : {} Average loss: {:.4f}'.format(epoch, avg_loss))
    return avg_loss


def compute_perplexity(model, dataloader, device):
    
    model.eval()
    loss = 0
    
    with torch.no_grad():
        for i, data_bow in enumerate(dataloader):
            data_bow = data_bow.to(device)
            data_bow_norm = F.normalize(data_bow)
            
            z, g, recon_batch, mu, logvar = model(data_bow_norm)
            
            #loss += loss_function(recon_batch, data_bow, mu, logvar).detach()
            loss += F.binary_cross_entropy(recon_batch, data_bow, size_average=False)
            
    loss = loss / dataloader.word_count
    perplexity = np.exp(loss.cpu().numpy())
    
    return perplexity


def lasy_predict(model, dataloader,vocab_dic, device, num_example=5, n_top_words=5):
    model.eval()
    docs, text = dataloader.bow_and_text()
    
    docs, text = docs[:num_example], text[:num_example]
    
    docs_device = docs.to(device)
    docs_norm = F.normalize(docs_device)
    z, _, _, _, _ = model(docs_norm)
    z_a = z.detach().cpu().argmax(1).numpy()
    z = torch.softmax(z, dim=1).detach().cpu().numpy()
    
    beta_exp = model.fcd1.weight.data.cpu().numpy().T
    topics = []
    for k, beta_k in enumerate(beta_exp):
        topic_words = [vocab_dic[w_id] for w_id in np.argsort(beta_k)[:-n_top_words - 1:-1]]
        topics.append(topic_words)
    
    for i, (zi, _z_a, t) in enumerate(zip(z, z_a, text)):
        print('\n===== # {}, Topic : {}, p : {:.4f} %'.format(i+1, _z_a,  zi[_z_a] * 100))
        print("Topic words :", ', '.join(topics[_z_a]))
        new_text = filter(lambda a:a is not None, t) # FIXME: avoid error
        print("Input :", ' '.join(new_text))
        

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform(m.weight)