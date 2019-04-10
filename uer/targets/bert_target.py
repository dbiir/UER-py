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
        self.transform_norm = LayerNorm(args.hidden_size)
        self.transform = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_mlm = nn.Linear(args.hidden_size, self.vocab_size)

        # NSP.
        self.pooler = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_nsp = nn.Linear(args.hidden_size, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.criterion = nn.NLLLoss()

    def mlm(self, memory_bank, tgt_mlm):
        # Masked language model (MLM) with full softmax prediction.
        output_mlm = gelu(self.transform(memory_bank))
        output_mlm = self.transform_norm(output_mlm)
        output_mlm = self.output_mlm(output_mlm)
        output_mlm = output_mlm.contiguous().view(-1, self.vocab_size)
        # Full probability distribution.
        output_mlm = self.softmax(output_mlm)

        tgt_mlm = tgt_mlm.contiguous().view(-1,1)
        label_mask = (tgt_mlm > 0).float()

        label_mask = (tgt_mlm > 0).float().to(torch.device(output_mlm.device))
        one_hot = torch.zeros(label_mask.size(0),  self.vocab_size). \
           to(torch.device(output_mlm.device)). \
           scatter_(1, tgt_mlm, 1.0)

        # one_hot = torch.cuda.FloatTensor(label_mask.size(0), self.vocab_size).fill_(0).scatter_(1, tgt_mlm, 1.0)

        numerator = -torch.sum(output_mlm * one_hot, 1)
        label_mask = label_mask.contiguous().view(-1)
        tgt_mlm = tgt_mlm.contiguous().view(-1)
        numerator = torch.sum(label_mask * numerator)
        denominator = torch.sum(label_mask) + 1e-6
        loss_mlm = numerator / denominator
        correct_mlm = torch.sum(label_mask * (output_mlm.argmax(dim=-1).eq(tgt_mlm)).float())

        return loss, correct_mlm, denominator

    def forward(self, memory_bank, tgt):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt_mlm: [batch_size x seq_length]
            tgt_nsp: [batch_size]

        Returns:
            loss_mlm: Masked language model loss.
            loss_nsp: Next sentence prediction loss.
            correct_mlm: Number of words that are predicted correctly.
            correct_nsp: Number of sentences that are predicted correctly.
            denominator: Number of masked words.
        """

        # Masked language model (MLM).
        assert type(tgt) == tuple
        tgt_mlm, tgt_nsp = tgt[0], tgt[1]
        loss_mlm, correct_mlm, denominator = self.mlm(memory_bank, tgt_mlm)
           
        # Next sentence prediction (NSP).
        output_nsp = torch.tanh(self.pooler(memory_bank[:, 0, :]))
        output_nsp = self.output_nsp(output_nsp)
        loss_nsp = self.criterion(self.softmax(output_nsp), tgt_nsp)
        correct_nsp = self.softmax(output_nsp).argmax(dim=-1).eq(tgt_nsp).sum()

        return loss_mlm, loss_nsp, correct_mlm, correct_nsp, denominator
