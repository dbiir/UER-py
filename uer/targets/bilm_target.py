# -*- encoding:utf-8 -*-
import math
import torch
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm
from uer.utils.act_fun import gelu
from uer.utils.misc import *


class BilmTarget(nn.Module):
    """
    """
    def __init__(self, args, vocab_size):
        super(BilmTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size // 2

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
        
        assert type(tgt) == tuple
        tgt_forward, tgt_backward = tgt[0], tgt[1]
        tgt_backward = flip(tgt_backward, 1)

        # Forward.
        output_forward = self.output_layer(memory_bank[:,:,:self.hidden_size])        
        output_forward = output_forward.contiguous().view(-1, self.vocab_size)
        output_forward = self.softmax(output_forward)        
        tgt_forward = tgt_forward.contiguous().view(-1,1)
        label_mask_forward = (tgt_forward > 0).float().to(torch.device(output_forward.device))
        one_hot_forward = torch.zeros(label_mask_forward.size(0),  self.vocab_size). \
           to(torch.device(output_forward.device)). \
           scatter_(1, tgt_forward, 1.0)
        numerator_forward = -torch.sum(output_forward * one_hot_forward, 1)
        label_mask_forward = label_mask_forward.contiguous().view(-1)
        tgt_forward = tgt_forward.contiguous().view(-1)
        numerator_forward = torch.sum(label_mask_forward * numerator_forward)
        denominator_forward = torch.sum(label_mask_forward) + 1e-6
        loss_forward = numerator_forward / denominator_forward
        correct_forward = torch.sum(label_mask_forward * (output_forward.argmax(dim=-1).eq(tgt_forward)).float())

        # Backward.
        output_backward = self.output_layer(memory_bank[:,:,self.hidden_size:])
        output_backward = output_backward.contiguous().view(-1, self.vocab_size)
        output_backward = self.softmax(output_backward)
        tgt_backward = tgt_backward.contiguous().view(-1,1)
        label_mask_backward = (tgt_backward > 0).float().to(torch.device(output_backward.device))
        one_hot_backward = torch.zeros(label_mask_backward.size(0),  self.vocab_size). \
           to(torch.device(output_backward.device)). \
           scatter_(1, tgt_backward, 1.0)
        numerator_backward = -torch.sum(output_backward * one_hot_backward, 1)
        label_mask_backward = label_mask_backward.contiguous().view(-1)
        tgt_backward = tgt_backward.contiguous().view(-1)
        numerator_backward = torch.sum(label_mask_backward * numerator_backward)
        denominator_backward = torch.sum(label_mask_backward) + 1e-6
        loss_backward = numerator_backward / denominator_backward
        correct_backward = torch.sum(label_mask_backward * (output_backward.argmax(dim=-1).eq(tgt_backward)).float())

        return loss_forward, loss_backward, correct_forward, correct_backward, denominator_backward
