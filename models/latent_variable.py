#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latent Variable class with sub-classes that determine type of inference for the latent variable

"""
import gpytorch
import torch
from torch.distributions import kl_divergence

class LatentVariable(gpytorch.Module):
    
    """
    :param latent_dim (int): Dimensionality of latent space.
    :param n (int): Size of the latent space.

    """

    def __init__(self, n, dim):
        super().__init__()
        self.n = n
        self.dim = dim
        
    def forward(self, x):
        raise NotImplementedError
        
class PointLatentVariable(LatentVariable):
    
    def __init__(self, n, dim, X_init):
        super().__init__(n, dim)
        self.register_parameter('X', X_init)

    def forward(self, x):
        return x
        
class MAPLatentVariable(LatentVariable):
    
    def __init__(self, n, dim, X_init, prior_x):
        super().__init__()
        self.prior_x = prior_x
        self.register_parameter('X', X_init)
        self.register_prior('prior_x', prior_x, 'X')

    def forward(self, x):
        return x
        
class VariationalLatentVariable(LatentVariable):
    
    def __init__(self, n, dim, X_init, prior_x):
        super().__init__(n, dim)
        
        self.prior_x = prior_x
        # Local variational params per latent point
        self.q_mu = X_init
        self.q_log_sigma = torch.nn.Parameter(torch.tensor(torch.randn(n , dim)))
       
        # Variational distribution over the latent variable q(x)
        self.q_x = torch.distributions.Normal(self.q_mu, torch.exp(self.q_log_sigma))
        
        self.x_kl = kl_divergence(self.q_x, prior_x).sum()

    def forward(self):
        return self.q_x.rsample()
    
        