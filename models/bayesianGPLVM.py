#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gpytorch.models import ApproximateGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
import torch
from prettytable import PrettyTable

class BayesianGPLVM(ApproximateGP):
    def __init__(self, Y, latent_dim, n_inducing, X_init=None, 
                 pca=True, prior_x=None, kernel=None, likelihood=None, inference='variational'):
        
        """The GPLVM model class for unsupervised learning. The current implementation 
           can learn:
        
        (a) Point estimates for latent X when prior_x = None 
        (b) MAP Inference for X when prior_x is not None and inference == 'map'
        (c) Gaussian variational distribution q(X) when prior_x is not None and inference == 'variational'

        :param Y (torch.Tensor): The unsupervised data set of size D x N (N data points)
        :param batch_shape (int): Size of the batch of GP mappings (D, one per dimension).
        :param latent_dim (int): Dimensionality of latent space.
        :param pca (bool): Whether to initialise with pca / randn / zeros.
        :param inducing_inputs (torch.Tensor): The inputs corresponding to the inducing variables.
        :param prior_x (gpytorch.priors): Can be None or Gaussian.
        :param kernel (gpytorch.kernel): The kernel that governs the GP mappings from
                                latent to data space, can select any from standard choices.
        :param likelihood (gpytorch.likelihoods): Gaussian likelihood for continuous targets,
                                                    Bernoulli for classification targets.
         
        """
        self.Y = Y
        self.batch_shape = torch.Size([Y.shape[0]])
        self.latent_dim = latent_dim
        self.n_inducing = n_inducing
        self.n = Y.shape[1]
        self.inference = inference
        
        # Locations Z_{d} corresponding to u_{d}, they can be randomly initialized or 
        # regularly placed with shape (D x n_inducing x latent_dim).
        
        self.inducing_inputs = torch.randn(Y.shape[0], self.n_inducing, self.latent_dim)
        
        # Sparse Variational Formulation
        
        # Variational distribution of inducing variables
        q_u = CholeskyVariationalDistribution(self.inducing_inputs.size(-2), 
                                                                batch_shape=self.batch_shape) 
        # $q(f) = \int p(f|u)q(u)du$
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)
        
        super(BayesianGPLVM, self).__init__(q_f)
        
        # Initialise either with PCA or fixed tensor, if none is provided - it reverts to 0s.
        
        if pca == True:
            X_init = self._init_pca() # Initialise X to PCA 
        else:
            if X_init is not None: 
                X_init = torch.nn.Parameter(X_init) # Initialise X with a tensor of shape N x Q 
            else:
                X_init = torch.nn.Parameter(torch.zeros(Y.shape[1],latent_dim))
                
        
        if self.inference == 'variational':
            # Local variational params per latent point
            self.q_mu = X_init
            self.q_log_sigma = torch.nn.Parameter(torch.tensor(torch.randn(Y.shape[1],self.latent_dim)))
        
            # Variational distribution over the latent variable q(x)
            self.q_x = torch.distributions.Normal(self.q_mu, torch.exp(self.q_log_sigma))
  
            # Q: can't figure out how to register a prior for x without x being a parameter.
            self.prior_x = prior_x
        
        else:
            # Learn latents as point estimates 
            self.register_parameter('X', X_init)

            if prior_x is not None: # MAP inference
                self.prior_x = prior_x
                #self.register_prior('prior_x', prior_x, 'X')
            
        # Kernel 
        self.mean_module = ConstantMean(ard_num_dims=latent_dim)
        if kernel is None:
            self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
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
    
    def sample_variational_latent_dist(self):
        
        epsilon = self.prior_x.sample_n(1)
        # Reparameterisation trick - Q: Do I need to do this explicitly? or just calling sample_n() should 
        # take care of it.
        sample = epsilon*torch.exp(self.q_log_sigma) + self.q_mu
        return sample
    
    def get_trainable_param_names(self):
        
        ''' Prints a list of parameters (model + variational) which will be 
        learnt in the process of optimising the objective '''
        
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params+=param
        print(table)
        print(f"Total Trainable Params: {total_params}")
    
    def _init_pca(self):
          U, S, V = torch.pca_lowrank(self.Y.T, q = self.latent_dim)
          return torch.nn.Parameter(torch.matmul(self.Y.T, V[:,:self.latent_dim]))