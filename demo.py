#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vr308

"""

from data.unsupervised_datasets import load_unsupervised_data
from models.bayesianGPLVM import BayesianGPLVM
from matplotlib import pyplot as plt
import torch
import numpy as np
from tqdm import trange
from gpytorch.mlls import VariationalELBO
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
    model = BayesianGPLVM(Y = Y.T, 
                  latent_dim = n_latent_dims,
                  n_inducing = n_inducing, 
                  X_init = None, 
                  pca = True, 
                  latent_prior = None,
                  kernel = None,
                  likelihood = None)
   
    # Declaring objective to be optimised along with optimiser
    
    mll = VariationalELBO(model.likelihood, model, num_data=len(Y.T))
    
    optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    ], lr=0.01)
    
    # Training loop
    
    loss_list = []
    iterator = trange(1000, leave=True)
    for i in iterator:
       optimizer.zero_grad()
       output = model(model.q_mu)
       loss = -mll(output, model.Y).sum()
       loss_list.append(loss.item())
       print('Iter %d/%d - Loss: %.3f   q_mu: %.3f  q_sigma: %.3f  noise: %.3f' % (
           i + 1, 1000, loss.item(),
           model.q_mu[10,0].item(),
           torch.exp(model.q_log_sigma[10]),
           model.likelihood.noise[0].item()))
       iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
       loss.backward(retain_graph=True)
       optimizer.step()
    
    losses = model.run(mll, optimizer, steps=2000)
    
    # Plot result
    
    plt.figure(figsize=(8, 6))
    colors = plt.get_cmap("tab10").colors[::]
 
    X = model.q_mu.detach().numpy()
    for i, label in enumerate(np.unique(labels)):
        X_i = X[labels == label]
        plt.scatter(X_i[:, 0], X_i[:, 1], c=[colors[i]], label=label)