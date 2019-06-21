# -*- encoding:utf-8 -*-
import math
import torch
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm
from uer.utils.act_fun import gelu


class S2sTarget(nn.Module):
    """
    """
    def __init__(self, args, vocab_size):
        super(S2sTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size

        self.embedding_layer = nn.Embedding(vocab_size, args.emb_size)
        self.decoder = nn.LSTM(args.emb_size, args.hidden_size, 1, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_size, self.vocab_size)

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, memory_bank, tgt):

        emb = self.embedding_layer(tgt[:, :]) # bactch_size, seq_length, emb_size
        output = []
        hidden_state = (memory_bank[:,-1,:].unsqueeze(0).contiguous(), memory_bank[:,-1,:].unsqueeze(0).contiguous())
        for i, emb_i in enumerate(emb.split(1, dim=1)):
            output_i, hidden_state = self.decoder(emb_i, hidden_state)
            output.append(self.output_layer(output_i))

        output = torch.cat(output, dim=1)

        output = output.contiguous().view(-1, self.vocab_size)
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
