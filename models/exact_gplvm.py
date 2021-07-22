#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from typing import Optional, Tuple
from gpytorch.models import ExactGP, GP
from gpytorch.priors import Prior
from gpytorch.likelihoods import _GaussianLikelihoodBase, GaussianLikelihood

from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from tqdm import trange
import numpy as np

__all__ = ['GPLVM']


class _ExactGPLVM(ExactGP):
    """
    TODO: docs
    """

    def __init__(self, Y: torch.Tensor, latent_dim: int, likelihood: _GaussianLikelihoodBase,
                 X_init: Optional[torch.Tensor] = None, pca: bool = True,
                 latent_prior: Optional[Prior] = None):

        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise RuntimeError("ExactGP can only handle Gaussian likelihoods")

        GP.__init__(self)  # Bypass the exact GP constructor. This allows us to make train_inputs learnable
        self.register_buffer("Y", Y)
        self.likelihood = likelihood
        self.latent_dim = latent_dim

        # Define latent
        if pca == True:
            self._init_pca() # initialise X to PCA 
        else:
            if X_init is not None:
                self._init_latents(X_init)
            else:
                self.X = torch.nn.Parameter(torch.zeros(*Y.shape, latent_dim))
        
        # Latent prior
        if latent_prior is not None:
            self.register_prior('prior_X', latent_prior, 'X')

    @property
    def train_inputs(self):
        return (self.X,)

    @train_inputs.setter
    def train_inputs(self, val: Tuple[torch.Tensor]):
        if isinstance(val, torch.Tensor):
            raise ValueError("ExactGPLVM#train_inputs must be set to a tuple")
        elif len(val) != 1:
            raise ValueError(f"ExactGPLVM#train_inputs must be a tuple of length 1; got {len(val)}")
        val = val[0]

        self.X.data.copy_(val)

    @property
    def train_targets(self):
        return self.Y

    @property
    def train_targets(self):
        return self.Y

    @train_targets.setter
    def train_targets(self, value):
        object.__setattr__(self, "Y", value)
    
    def _init_pca(self):
        U, S, V = torch.pca_lowrank(self.Y.T)
        self.X = torch.nn.Parameter(torch.matmul(self.Y.T, V[:,:self.latent_dim]))

    def _init_latents(self, X_init):
        self.X = torch.nn.Parameter(X_init)


class ExactGPLVM(_ExactGPLVM):
    def __init__(self, Y, latent_dim, X_init=None, pca=True, latent_prior=None):
        self.batch_shape = torch.Size([Y.shape[0]])
        self.latent_dim = latent_dim
        
        likelihood = GaussianLikelihood()
        super().__init__(Y, latent_dim, likelihood, X_init=X_init, pca=pca, latent_prior=latent_prior)

        # Kernel 
        self.mean_module = ConstantMean(ard_num_dims=latent_dim, batch_shape=self.batch_shape)
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim, 
                                                  batch_shape=self.batch_shape), batch_shape=self.batch_shape)
       
    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist
    
    def run(self, objective, optimizer, steps=200):
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

