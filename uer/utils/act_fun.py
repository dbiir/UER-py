import math
import torch
import torch.nn.functional as F
from packaging import version


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def relu(x):
    return F.relu(x)

def linear(x):
    return x

def _silu_python(x):
    """
    See Gaussian Error Linear Units (Hendrycks et al., https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear
    Unit) was originally introduced and coined, and see Sigmoid-Weighted Linear Units for Neural Network Function
    Approximation in Reinforcement Learning (Elfwing et al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated
    Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where the SiLU was experimented with
    later.
    """
    return x * torch.sigmoid(x)

  
if version.parse(torch.__version__) < version.parse("1.7"):
    silu = _silu_python
else:
    silu = F.silu
