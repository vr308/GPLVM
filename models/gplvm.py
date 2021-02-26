#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gpytorch.models import ApproximateGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from tqdm import trange
import torch
import numpy as np

__all__ = ['GPLVM']

class GPLVM(ApproximateGP):
    def __init__(self, Y, latent_dim, n_inducing, X_init=None, 
                 pca=True, latent_prior=None, kernel=None, likelihood=None):
        
        """The GPLVM model class for unsupervised learning. The current implementation 
           can learn:
        
        (a) Point estimates for latent X when latent_prior = None 
        (b) MAP Inference for X when latent prior = N(\mu, \sigma)

        :param Y (torch.Tensor): The unsupervised data set of size D x N (N data points)
        :param batch_shape (int): Size of the batch of GP mappings (D, one per dimension).
        :param latent_dim (int): Dimensionality of latent space.
        :param pca (bool): Whether to initialise with pca or zeros
        :param inducing_inputs (torch.Tensor): 
        :param latent_prior (gpytorch.priors): Can be None or Gaussian.
        :param kernel (gpytorch.kernel): The kernel that governs the GP mappings from
                                latent to data space, can select any from standard choices.
        :param likelihood (gpytorch.likelihoods): Gaussian likelihood for continuous targets,
                                                    Bernoulli for classification targets.
         
        """
        self.Y = Y
        self.batch_shape = torch.Size([Y.shape[0]])
        self.latent_dim = latent_dim
        self.n_inducing = n_inducing
        
        # Locations Z_{d} corresponding to u_{d}, they can be randomly initialized or 
        # regularly placed with shape (D x n_inducing x latent_dim).
        
        self.inducing_inputs = torch.randn(Y.shape[0], self.n_inducing, self.latent_dim)
        
        # Sparse Variational Formulation
        variational_distribution = CholeskyVariationalDistribution(self.inducing_inputs.size(-2), 
                                                                batch_shape=self.batch_shape) #q(u)
        variational_strategy = VariationalStrategy(self, self.inducing_inputs, 
                                                    variational_distribution, 
                                                    learn_inducing_locations=True)
        super(GPLVM, self).__init__(variational_strategy)
        
        # Register X as a parameter and initialise either with PCA or fixed tensor
        # If none is provided, it reverts to 0s.
        if pca == True:
            self._init_pca() # initialise X to PCA 
        else:
            if X_init is not None:
                self._init_latents(X_init)
            else:
                self.X = torch.nn.Parameter(torch.zeros(Y.shape[1],latent_dim))
        
        self.register_parameter(name="X", parameter=self.X)
        
        # Latent prior
        if latent_prior is not None:
              self.register_prior('prior_X', latent_prior, 'X')

        # Kernel 
        self.mean_module = ConstantMean(ard_num_dims=latent_dim, 
                                        batch_shape=self.batch_shape)
        if kernel is None:
            self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim, 
                                                batch_shape=self.batch_shape),batch_shape=self.batch_shape)
        else: 
            self.covar_module = kernel
        
        # Data Likelihood
        if likelihood is None:
            self.likelihood = GaussianLikelihood(batch_shape=self.batch_shape)
        else:
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
    
    def _init_pca(self):
          U, S, V = torch.pca_lowrank(self.Y.T)
          self.X = torch.nn.Parameter(torch.matmul(self.Y.T, V[:,:self.latent_dim]))
    
    def _init_latents(self, X_init):
         
         ''' Initialise X with a fixed tensor of shape N x Q
         
         :param X_init (torch.Tensor): Q dimensional latent points corresponding 
                                         to D dimensional data points Y.
    
         '''
         self.X = torch.nn.Parameter(X_init)
    
    def run(self, objective, optimizer, steps=1000):
        
         ''' Optimises the objective wrt kernel hypers and variational hyperparameters
             using the optimizer provided.
             
         :param objective (gpytorch.mlls): Select gpytorch.mlls.variational_elbo.VariationalELBO for the 
                                             variational evidence lower bound compatible with generic 
                                             factorised likelihoods.
         :param optimizer (torch.optim): Initialise an instance of an optimizer of specific class with custom
                                         parameter options.
                                         Eg: optimizer = torch.optim.Adam([
                                                {'params': model.parameters()},
                                                {'params': likelihood.parameters()},
                                                ], lr=0.02, betas=(0.9,0.999))
         :param batch_size (int):  Batch size to use in each iteration. 
             
         '''
        
         loss_list = []
         iterator = trange(steps, leave=True)
         for i in iterator:
            optimizer.zero_grad()
            output = self(self.X)
            loss = -objective(output, self.Y).sum()
            loss_list.append(loss.item())
            iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
            loss.backward(retain_graph=True)
            optimizer.step()
         return loss_list

