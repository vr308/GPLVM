#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gpytorch.models import ApproximateGP
from prettytable import PrettyTable

class BayesianGPLVM(ApproximateGP):
    def __init__(self, X, variational_strategy):
        
        """The GPLVM model class for unsupervised learning. The current class supports
        
        (a) Point estimates for latent X when prior_x = None 
        (b) MAP Inference for X when prior_x is not None and inference == 'map'
        (c) Gaussian variational distribution q(X) when prior_x is not None and inference == 'variational'

        :param X (LatentVariable): An instance of a sub-class of the LatentVariable class.
                                    One of,
                                    PointLatentVariable / 
                                    MAPLatentVariable / 
                                    VariationalLatentVariable to
                                    facilitate inference with (a), (b) or (c) respectively.
       
        """
     
        super(BayesianGPLVM, self).__init__(variational_strategy)
        
        # Assigning Latent Variable 
        self.X = X 
    
    def forward(self):
        raise NotImplementedError
          
    def sample_latent_variable(self):
        sample = self.X()
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
    
   
