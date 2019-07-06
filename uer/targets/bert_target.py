# -*- encoding:utf-8 -*-
import math
import torch
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm
from uer.utils.act_fun import gelu


class BertTarget(nn.Module):
    """
    BERT exploits masked language modeling (MLM) 
    and next sentence prediction (NSP) for pretraining.
    """
    def __init__(self, args, vocab_size):
        super(BertTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size

        # MLM.
        self.mlm_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.layer_norm = LayerNorm(args.hidden_size)
        self.mlm_linear_2 = nn.Linear(args.hidden_size, self.vocab_size)

        # NSP.
        self.nsp_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.nsp_linear_2 = nn.Linear(args.hidden_size, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.criterion = nn.NLLLoss()

    def mlm(self, memory_bank, tgt_mlm):
        # Masked language modeling (MLM) with full softmax prediction.
        output_mlm = gelu(self.mlm_linear_1(memory_bank))
        output_mlm = self.layer_norm(output_mlm)
        output_mlm = output_mlm.contiguous().view(-1, self.hidden_size)
        tgt_mlm = tgt_mlm.contiguous().view(-1)
        output_mlm = output_mlm[tgt_mlm>0,:]
        tgt_mlm = tgt_mlm[tgt_mlm>0]
        output_mlm = self.mlm_linear_2(output_mlm)
        output_mlm = self.softmax(output_mlm)

        one_hot = torch.zeros(output_mlm.size(0),  self.vocab_size). \
           to(torch.device(output_mlm.device)). \
           scatter_(1, tgt_mlm.contiguous().view(-1,1), 1.0)
        numerator = -torch.sum(output_mlm * one_hot, 1)
        denominator = torch.tensor(output_mlm.size(0) + 1e-6)
        loss_mlm = torch.sum(numerator) / denominator
        if output_mlm.size(0) == 0:
            correct_mlm = torch.tensor(0.0)
        else:
            correct_mlm = torch.sum((output_mlm.argmax(dim=-1).eq(tgt_mlm)).float())
        
        return loss_mlm, correct_mlm, denominator

    def forward(self, memory_bank, tgt):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: tuple with tgt_mlm [batch_size x seq_length] and tgt_nsp [batch_size]
            

        Returns:
            loss_mlm: Masked language modeling loss.
            loss_nsp: Next sentence prediction loss.
            correct_mlm: Number of words that are predicted correctly.
            correct_nsp: Number of sentences that are predicted correctly.
            denominator: Number of masked words.
        """

        # Masked language modeling (MLM).
        assert type(tgt) == tuple
        tgt_mlm, tgt_nsp = tgt[0], tgt[1]
        loss_mlm, correct_mlm, denominator = self.mlm(memory_bank, tgt_mlm)
           
        # Next sentence prediction (NSP).
        output_nsp = torch.tanh(self.nsp_linear_1(memory_bank[:, 0, :]))
        output_nsp = self.nsp_linear_2(output_nsp)
        loss_nsp = self.criterion(self.softmax(output_nsp), tgt_nsp)
        correct_nsp = self.softmax(output_nsp).argmax(dim=-1).eq(tgt_nsp).sum()

        return loss_mlm, loss_nsp, correct_mlm, correct_nsp, denominator
