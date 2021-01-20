import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, use_bias=True):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.use_bias = use_bias
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        hidden_states =  self.gamma * (x-mean) / (std + self.eps)

        if self.use_bias:
            hidden_states += self.beta

        return hidden_states
