#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vr308

"""

# TODO's:
# Mini-batching
# Snaphot param state and save
# Flexible variational family
# Amortised Inference
# Missing data dimensions (masking)

from data.unsupervised_datasets import load_unsupervised_data
from models.bayesianGPLVM import BayesianGPLVM
from matplotlib import pyplot as plt
import torch
import numpy as np
from tqdm import trange
from gpytorch.mlls import VariationalELBO
from torch.distributions import kl_divergence

if __name__ == '__main__':

    # Load some data
    
    Y, n, d, labels = load_unsupervised_data('oilflow')
      
    # Setting shapes
    
    n_data_dims = Y.shape[1]
    n_latent_dims = 12
    n_inducing = 50
    X_prior_mean = torch.nn.Parameter(torch.zeros(Y.shape[0], n_latent_dims))  # shape: N x Q
    
    # Declaring model with initial inducing inputs and latent prior
    
    prior_x = torch.distributions.Normal(X_prior_mean, torch.ones_like(X_prior_mean))
    
    model = BayesianGPLVM(Y = Y.T, 
                  latent_dim = n_latent_dims,
                  n_inducing = n_inducing, 
                  X_init = None, 
                  pca = True, 
                  prior_x = prior_x,
                  kernel = None,
                  likelihood = None,
                  inference='variational')
   
    # Declaring objective to be optimised along with optimiser
    
    mll = VariationalELBO(model.likelihood, model, num_data=len(Y.T))
    
    optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    ], lr=0.01)
    
    # Training loop - optimises the objective wrt kernel hypers, variational params and inducing inputs
    # using the optimizer provided.
    
    loss_list = []
    iterator = trange(1000, leave=True)
    for i in iterator:
       optimizer.zero_grad()
       if model.inference == 'variational':
           sample = model.sample_variational_latent_dist() 
           custom_terms = kl_divergence(model.q_x, prior_x).sum()
       else: 
           sample = model.X
           custom_terms = None
       output = model(sample)
       loss = -(mll(output, model.Y, added_loss_terms = custom_terms).sum())
       loss_list.append(loss.item())
       iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
       loss.backward()
       optimizer.step()
        
    # Plot result
    
    plt.figure(figsize=(8, 6))
    colors = ['r', 'b', 'g']
 
    X = model.q_mu.detach().numpy()
    std = torch.exp(model.q_log_sigma).detach().numpy()
    
    #X = model.X.detach().numpy()
    for i, label in enumerate(np.unique(labels)):
        X_i = X[labels == label]
        scale_i = std[labels == label]
        plt.scatter(X_i[:, 1], X_i[:, 11], c=[colors[i]], label=label)
        plt.errorbar(X_i[:, 1], X_i[:, 11], xerr=scale_i[:,1], yerr=scale_i[:,11], label=label,c=colors[i], fmt='none')