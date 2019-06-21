# -*- encoding:utf-8 -*-
import math
import torch
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm
from uer.utils.act_fun import gelu


class LmTarget(nn.Module):
    """
    """
    def __init__(self, args, vocab_size):
        super(LmTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size

        self.softmax = nn.LogSoftmax(dim=-1)
        self.output_layer = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, memory_bank, tgt):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size x seq_length]

        Returns:
            loss: Language modeling loss.
            correct: Number of words that are predicted correctly.
            denominator: Number of predicted words.
        """

        # Language modeling (LM) with full softmax prediction.
        output = self.output_layer(memory_bank)
        output = output.contiguous().view(-1, self.vocab_size)
        # Full probability distribution.
        output = self.softmax(output)

        tgt = tgt.contiguous().view(-1,1)
        label_mask = (tgt > 0).float().to(torch.device(output.device))
        one_hot = torch.zeros(label_mask.size(0),  self.vocab_size). \
           to(torch.device(output.device)). \
           scatter_(1, tgt, 1.0)

        numerator = -torch.sum(output * one_hot, 1)
        label_mask = label_mask.contiguous().view(-1)
        tgt = tgt.contiguous().view(-1)
        numerator = torch.sum(label_mask * numerator)
        denominator = torch.sum(label_mask) + 1e-6
        loss = numerator / denominator
        correct = torch.sum(label_mask * (output.argmax(dim=-1).eq(tgt)).float())

        return loss, correct, denominator
