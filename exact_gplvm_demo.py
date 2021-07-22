#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vr308

"""

from data.unsupervised_datasets import load_unsupervised_data
from models.gplvm import GPLVM
from models.exact_gplvm import ExactGPLVM
from matplotlib import pyplot as plt
import torch
import numpy as np
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import NormalPrior


if __name__ == '__main__':

    # Load some data
    
    Y, n, d, labels = load_unsupervised_data('oilflow')
      
    # Setting shapes
    
    n_data_dims = Y.shape[1]
    n_latent_dims = 2
    n_inducing = 32
    X_prior_mean = torch.zeros(Y.shape[0], n_latent_dims)  # shape: N x Q
    
    # Declaring model with initial inducing inputs and latent prior
    
    latent_prior = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))
    model = ExactGPLVM(Y = Y.T, 
                       latent_dim = n_latent_dims,
                       X_init = None, 
                       pca = True, 
                       latent_prior = None)
   
    # Declaring objective to be optimised along with optimiser
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    optimizer = torch.optim.Adam([{'params': model.parameters()},], lr=0.01)
    
    # Training loop
    model.train()
    losses = model.run(mll, optimizer, steps=1000)
    
    # Plot result
    
    plt.figure(figsize=(8, 6))
    colors = plt.get_cmap("tab10").colors[::]
 
    X = model.X.detach().numpy()
    for i, label in enumerate(np.unique(labels)):
        X_i = X[labels == label]
        plt.scatter(X_i[:, 0], X_i[:, 1], c=[colors[i]], label=label)
