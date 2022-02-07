import torch
from torch import nn
import torch.nn.functional as F

import numpy as np


def sample_normal(shape, seed=None):
    # sample from standard Normal with a given shape
    rng = np.random.RandomState() if seed is None else np.random.RandomState(seed)
    return torch.tensor(rng.normal(size=shape).astype(np.float32))


class BayesianLinear(nn.Module):
    """
    A Bayesian linear layer (having bias) with provisioning for mean field inference
    reference: https://github.com/Harry24k/bayesian-neural-network-pytorch/blob/master/torchbnn/modules/linear.py
    """

    def __init__(self, in_features, out_features, prior_var=1.0):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.prior_var = prior_var
        self.prior_log_sigma = np.log(np.sqrt(prior_var))

        self.weight_mu = nn.Parameter(torch.zeros((out_features, in_features)))
        self.weight_log_sigma = nn.Parameter(torch.zeros((out_features, in_features)))
        self.register_buffer('weight_eps', None)
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_log_sigma = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('bias_eps', None)

        self.reset_parameters()
        self.resample()

    def reset_parameters(self):
        # initialize means by random samples from a Normal distribution with a small std deviation
        # initialize the stds to be a small value (0.01)
        self.weight_mu.data.copy_(sample_normal(self.weight_mu.shape).data * np.sqrt(0.01))
        self.weight_log_sigma.data.fill_(np.log(0.01))
        self.bias_mu.data.copy_(sample_normal(self.bias_mu.shape).data * np.sqrt(0.01))
        self.bias_log_sigma.data.fill_(np.log(0.01))

    def resample(self):
        self.weight_eps = sample_normal(self.weight_mu.shape)
        self.bias_eps = sample_normal(self.bias_mu.shape)

    def forward(self, x):
        weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(self.weight_log_sigma)
        bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_log_sigma)
        return F.linear(x, weight, bias)

    def kl(self):
        # weights prior distribution
        weights_pdist = torch.distributions.Normal(loc=torch.zeros(self.weight_mu.shape),
                                                   scale=torch.ones(self.weight_mu.shape) * np.sqrt(self.prior_var))
        # weights posterior distribution
        weights_qdist = torch.distributions.Normal(loc=self.weight_mu,
                                                   scale=torch.exp(self.weight_log_sigma))
        # bias prior distribution
        bias_pdist = torch.distributions.Normal(loc=torch.zeros(self.bias_mu.shape),
                                                scale=torch.ones(self.bias_mu.shape) * np.sqrt(self.prior_var))
        # bias posterior distribution
        bias_qdist = torch.distributions.Normal(loc=self.bias_mu,
                                                scale=torch.exp(self.bias_log_sigma))
        # total kl
        kl = torch.distributions.kl_divergence(weights_qdist, weights_pdist).sum() + \
             torch.distributions.kl_divergence(bias_qdist, bias_pdist).sum()
        return kl

    def __repr__(self):
        return f"BayesianLinear({self.in_features, self.out_features}) with prior variance {self.prior_var}"


class BNN(nn.Module):
    activation = nn.Tanh()

    def __init__(self, in_dims: int, out_dims: int, num_layers: int = 1, num_units: int = 32, prior_var: float = 1.0):
        """
        Bayesian neural network with mean field inference for all the net parameters.
        Also, in case of multiple hidden layers, all the hidden layers get equal number of hidden units.

        Args:
            in_dims: net input dimensions
            out_dims: net putput dimensions
            num_layers: net number of layers
            num_units: number of units in hidden layers
            prior_var: prior variance of the net parameters for KL computation
        """
        super(BNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.num_layers = num_layers
        self.num_units = num_units
        self.prior_var = prior_var

        layers_in_dims = [in_dims] + [num_units] * num_layers
        layers_out_dims = [num_units] * num_layers + [out_dims]

        modules = []
        for i, (in_dims, out_dims) in enumerate(zip(layers_in_dims, layers_out_dims)):
            modules.append(BayesianLinear(in_features=in_dims, out_features=out_dims, prior_var=prior_var))
            activation = nn.Identity() if i == self.num_layers else self.activation
            modules.append(activation)
        self.net = nn.Sequential(*modules)

    def resample(self):
        for module in self.net.modules():
            if isinstance(module, BayesianLinear):
                module.resample()

    def forward(self, x):
        return self.net(x)

    def kl(self):
        kl = 0.
        for module in self.net.modules():
            if isinstance(module, BayesianLinear):
                kl = kl + module.kl()
        return kl
