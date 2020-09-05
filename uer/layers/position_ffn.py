import torch.nn as nn
from uer.utils.act_fun import gelu
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    """ Feed Forward Layer. """
    def __init__(self, hidden_size, feedforward_size, hidden_act):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(hidden_size, feedforward_size)
        self.linear_2 = nn.Linear(feedforward_size, hidden_size)
        if hidden_act == 'relu':
            self.act = F.relu
        elif hidden_act == 'gelu':
            self.act = gelu
        else:
            raise ValueError("Activation function should be relu or gelu.")

    def forward(self, x):
        inter = self.act(self.linear_1(x))
        output = self.linear_2(inter)
        return output
