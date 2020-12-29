#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vr308

"""

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.likelihoods import MultitaskGaussianLikelihood, GaussianLikelihood
from gpytorch.priors import NormalPrior
from matplotlib import pyplot as plt
import torch
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood
from tqdm.notebook import tqdm
import pandas as pd
from torch.nn import Parameter
import os

URL = "https://raw.githubusercontent.com/sods/ods/master/datasets/guo_qpcr.csv"

df = pd.read_csv(URL, index_col=0)
print("Data shape: {}\n{}\n".format(df.shape, "-" * 21))
print("Data labels: {}\n{}\n".format(df.index.unique().tolist(), "-" * 86))
print("Show a small subset of the data:")
df.head()

data = torch.tensor(df.values, dtype=torch.get_default_dtype())
# we need to transpose data to correct its shape
train_y_pyro = data.t()

capture_time = train_y_pyro.new_tensor([int(cell_name.split(" ")[0]) for cell_name in df.index.values])
# we scale the time into the interval [0, 1]
time = capture_time.log2() / 6

X_prior_mean = torch.zeros(train_y_pyro.size(1), 2)  # shape: 437 x 2
X_prior_mean[:, 0] = time

train_y = train_y_pyro
n_data_dims = train_y.shape[0]
n_latent_dims = 2


class bGPLVM(ApproximateGP):
    def __init__(self, n_inducing_points, n_latent_dims, n_data_points):
        batch_shape = torch.Size([n_data_dims])
        inducing_points = torch.randn(n_data_dims, n_inducing_points, n_latent_dims)
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=batch_shape,
        )
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, 
            learn_inducing_locations=True
        )
        super(bGPLVM, self).__init__(variational_strategy)
        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(
            RBFKernel(nu=1.5,batch_shape=batch_shape, ard_num_dims=2),
            batch_shape=batch_shape
        )
        
        # using time as the prior mean in the x-direction as in he Pyro tutorial
        self.X = Parameter(X_prior_mean.clone())
        self.register_parameter(
            name="X", 
            parameter=self.X
            )
        self.register_prior('prior_X', NormalPrior(X_prior_mean,torch.ones_like(X_prior_mean)), 'X')
        
    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist

model = bGPLVM(n_inducing_points=32, n_latent_dims=n_latent_dims, n_data_points = train_y.shape[1])
likelihood = GaussianLikelihood(num_tasks=n_data_dims, batch_shape=torch.Size([n_data_dims]))

model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.01)

mll = PredictiveLogLikelihood(likelihood, model, num_data=train_y.size(0))

try:
    model.load_state_dict(torch.load("hypers.mod"))
    print("Hypers loaded")
except:
    loss_list = []
    iterator = tqdm(range(1000))
    for i in iterator:
        optimizer.zero_grad()
        output = model(model.X)
        loss = -mll(output, train_y).sum()
        loss_list.append(loss.item())
        print(str(loss.item()) + ", iter no: " + str(i))
        iterator.set_postfix(loss=loss.item())
        loss.backward(retain_graph=True)
        optimizer.step()
        
    torch.save(model.state_dict(),"hypers.mod")

plt.figure(figsize=(8, 6))
colors = plt.get_cmap("tab10").colors[::-1]
labels = df.index.unique()

X = model.X.detach().numpy()
for i, label in enumerate(labels):
    X_i = X[df.index == label]
    plt.scatter(0.5*X_i[:, 0], -X_i[:, 1], c=[colors[i]], label=label)

plt.legend()
plt.xlabel("pseudotime", fontsize=14)
plt.ylabel("branching", fontsize=14)
plt.title("GPLVM on Single-Cell qPCR data", fontsize=16)
plt.grid(True)
plt.show()