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
from models.latent_variable import *
from matplotlib import pyplot as plt
import torch
import numpy as np
from tqdm import trange
from gpytorch.mlls import VariationalELBO
from torch.distributions import kl_divergence

def _init_pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q = latent_dim)
    return torch.nn.Parameter(torch.matmul(Y, V[:,:latent_dim]))

if __name__ == '__main__':

    # Load some data
    
    Y, n, d, labels = load_unsupervised_data('oilflow')
      
    # Setting shapes
    
    n_data_dims = Y.shape[1]
    n_latent_dims = 12
    n_inducing = 50
    pca = True
    X_prior_mean = torch.zeros(Y.shape[0], n_latent_dims)  # shape: N x Q
    
    # Declaring model with initial inducing inputs and latent prior
    
    prior_x = torch.distributions.Normal(X_prior_mean, torch.ones_like(X_prior_mean))
    
    # Initialise X with PCA or 0s.
      
    if pca == True:
         X_init = _init_pca(Y, n_latent_dims) # Initialise X to PCA 
    else:
         X_init = torch.nn.Parameter(torch.zeros(Y.shape[1],n_latent_dims))
      
    # LatentVariable initialisation
    X = VariationalLatentVariable(Y.shape[0], n_latent_dims, X_init, prior_x)
    
    model = BayesianGPLVM(Y = Y.T, 
                  X = X,
                  latent_dim = n_latent_dims,
                  n_inducing = n_inducing, 
                  kernel = None,
                  likelihood = None)
   
    # Declaring objective to be optimised along with optimiser
    
    mll = VariationalELBO(model.likelihood, model, num_data=len(Y))
    
    optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    ], lr=0.01)
    
    # Training loop - optimises the objective wrt kernel hypers, variational params and inducing inputs
    # using the optimizer provided.
    
    loss_list = []
    iterator = trange(1000, leave=True)
    for i in iterator:
       optimizer.zero_grad()
       sample = model.X.forward()
       output = model(sample)
       loss = -mll(output, model.Y).sum()
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