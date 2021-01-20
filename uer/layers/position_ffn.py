import torch.nn as nn
from uer.utils import *


class PositionwiseFeedForward(nn.Module):
    """ Feed Forward Layer. """
    def __init__(self, hidden_size, feedforward_size, hidden_act, use_bias=True):
        super(PositionwiseFeedForward, self).__init__()

        self.linear_1 = nn.Linear(hidden_size, feedforward_size, bias=use_bias)
        if not isinstance(hidden_act, list):
            self.linear_2 = nn.Linear(feedforward_size, hidden_size, bias=use_bias)
            self.act = str2act[hidden_act]
        else:
            self.linear_2 = nn.Linear(hidden_size, feedforward_size, bias=use_bias)
            self.linear_3 = nn.Linear(feedforward_size, hidden_size, bias=use_bias)
            self.act = [str2act[act] for act in hidden_act]

    def forward(self, x):
        if not isinstance(self.act, list):
            inter = self.act(self.linear_1(x))
            output = self.linear_2(inter)
        else:
            inter_gelu = self.act[0](self.linear_1(x))
            inter_linear = self.act[1](self.linear_2(x))
            inter = inter_gelu * inter_linear
            output = self.linear_3(inter)

        return output