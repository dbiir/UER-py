# -*- encoding:utf-8 -*-
import math
import torch

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))