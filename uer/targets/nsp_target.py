# -*- encoding:utf-8 -*-
import math
import torch
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm
from uer.utils.act_fun import gelu


class NspTarget(nn.Module):
    """
    """
    def __init__(self, args, vocab_size):
        super(NspTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size

        self.linear = nn.Linear(args.hidden_size, args.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()


    def forward(self, memory_bank, tgt):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size]

        Returns:
            loss: Next sentence prediction loss.
            correct: Number of sentences that are predicted correctly.
        """

        output = self.linear(memory_bank[:, 0, :])
        loss = self.criterion(self.softmax(output), tgt)
        correct = self.softmax(output).argmax(dim=-1).eq(tgt).sum()

        return loss, correct
