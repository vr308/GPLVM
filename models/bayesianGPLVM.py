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
    def __init__(self, Y, X, latent_dim, n_inducing, kernel=None, likelihood=None):
        
        """The GPLVM model class for unsupervised learning. The current class supports
        
        (a) Point estimates for latent X when prior_x = None 
        (b) MAP Inference for X when prior_x is not None and inference == 'map'
        (c) Gaussian variational distribution q(X) when prior_x is not None and inference == 'variational'

        :param Y (torch.Tensor): The unsupervised data set of size D x N (N data points)
        :param X (LatentVariable): An instance of a sub-class of the LatentVariable class.
                                    One of PointLatentVariable / MAPLatentVariable / VariationalLatentVariable to
                                    facilitate inference with (a), (b) or (c) respectively.
        :param latent_dim (int): Dimensionality of latent space.
        :param n_inducing (torch.Tensor): Number of inducing variables.
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
        
        # Assigning Latent Variable 
        
        self.X = X 
        self.register_added_loss_term("x_kl")
       
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
        
        if type(X).__name__ == 'VariationalLatentVariable':
            self.update_added_loss_term('x_kl', X.x_kl_value)
            
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist
    
    def sample_variational_latent_dist(self):
        
        sample = self.q_x.rsample()
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
    
   