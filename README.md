# GPLVM: Core Models 

Implementation of Gaussian Process Latent Variable models in pytorch/gpytorch

Key References:

1) [Gaussian process latent variable models for visualisation of high dimensional data](https://papers.nips.cc/paper/2003/file/9657c1fffd38824e5ab0472e022e577e-Paper.pdf)
2) [Bayesian GPLVM](http://proceedings.mlr.press/v9/titsias10a/titsias10a.pdf)
3) [Local distance preservation in the GPLVM](https://dl.acm.org/doi/abs/10.1145/1143844.1143909?casa_token=xk93fApyEQoAAAAA%3AacOknmr9fwAp7G2neUwxlDZakjcPDQyq5bbvtYvlNxFAgDB46nFRCtcA5d3F_dPgIF3mgbc_2fI)
4) [Stochastic Variational Inference for back-constrained GPLVM](https://www.blackboxworkshop.org/pdf/gplvm_blackbox_final.pdf)
5) [Gaussian Processes for Big data](https://arxiv.org/abs/1309.6835)


Models 

-------------

1) GPLVM 
2) B-GPLVM
3) Back-constrained GPLVM
4) SVI GPLVM (with and without back-constraints)

Inference

-------------

1) ML-II Optimisation 
2) Variational Inference with the collapsed bound
3) Stochastic Variational Inference with the uncollapsed bound


Code Layout 

---------------

data/*  has all the data loading utilities  
models/* model classes 

Usage

------------

See demo.py
