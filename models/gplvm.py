#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gpytorch.models import ExactGP, ApproximateGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from tqdm import trange
import torch
import gpytorch
import numpy as np

__all__ = ['ExactGPLVM']

class ExactGPLVM(ExactGP):
    def __init__(self, Y, latent_dim, likelihood, X_init=None, pca=True, kernel=None):
        super(ExactGPLVM, self).__init__(X_init, Y, likelihood)

        """The GPLVM model class for unsupervised learning. Learn point 
              estimates for X using the full multi-ouput GP marginal likelihood"""
              
        self.Y = Y
        self.batch_shape = torch.Size([Y.shape[0]])
        self.latent_dim = latent_dim
        
        # Register X as a parameter and initialise either with PCA or fixed tensor
        # If none is provided, it reverts to 0s.
        if pca == True:
            U, S, V = torch.pca_lowrank(self.Y.T)
            self.X = torch.nn.Parameter(torch.matmul(self.Y.T, V[:,:self.latent_dim]))# initialise X to PCA 
        else:
            if X_init is not None: # Initialise X with a tensor of shape N x Q 
                self.X = torch.nn.Parameter(X_init)
            else:
                self.X = torch.nn.Parameter(torch.zeros(Y.shape[1],latent_dim))
        
        self.register_parameter(name="X", parameter=self.X)
        
        # Kernel 
        self.mean_module = ConstantMean(ard_num_dims=latent_dim, 
                                        batch_shape=self.batch_shape)
        if kernel is None:
            self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim, 
                                                batch_shape=self.batch_shape),batch_shape=self.batch_shape)
        else: 
            self.covar_module = kernel
        
        # Data Likelihood
        self.likelihood = likelihood
            
    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist   
    
        
    def get_trainable_param_names(self):
        
        ''' Prints a list of parameters (model + variational) which will be 
        learnt in the process of optimising the objective '''
        
        for name, value in self.named_parameters():
            print(name)       