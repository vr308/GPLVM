 This file tracks a list of tasks for the project. 
 
 Utilities 
 ---------
 - [x] Data loading utilities 
  - [ ] Visualisation utilities
  - [ ] Submit PR for gpytorch tutorial, track open issue [here](https://github.com/cornellius-gp/gpytorch/issues/1412).
 - [ ] Sync up with @holmresner @jrojsel who wrote a version of it. 

Models & Inference 
----------

 - [x] Implement base class GPLVM 
 - [x] Functionality to learn X as a parameter (point est.) with sparse ELBO
 - [x] MAP Inference  with sparse ELBO
 - [x] Choleskly Variational (inducing point formulation)
 - [x] Support for different kernels / likelihoods
 - [x] Learn X variationally q(X) BGPLVM
 - [ ] Prediction on test data under Bayesian GPLVM 
 - [ ] Ability to specify flexible priors on X / priors with learnable parameters
 - [ ] Amortised Inference functionality with difference encoders.
 - [ ] Masked Likelihood for missing data 
 - [ ] Mini-batching the SVI ELBO
