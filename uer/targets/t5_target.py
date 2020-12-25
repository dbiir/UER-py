import math
import torch
import torch.nn as nn
from uer.utils.act_fun import gelu
from uer.decoders import *
from uer.layers import *


class T5Target(nn.Module):
    """
    """
    def __init__(self, args, vocab_size):
        super(T5Target, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size
        self.layers_num = args.layers_num

        self.embedding = str2embedding[args.tgt_embedding](args, vocab_size)

        self.decoder = str2decoder[args.decoder](args)

        self.softmax = nn.LogSoftmax(dim=-1)
        self.output_layer = nn.Linear(self.hidden_size, self.vocab_size, bias = args.has_lmtarget_bias)

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
        tgt_in, tgt_out, src = tgt

        emb = self.embedding(tgt_in, None)
        
        hidden = self.decoder(memory_bank, emb, (src,))

        # Language modeling (LM) with full softmax prediction.
        output = self.output_layer(hidden)
        output = output.contiguous().view(-1, self.vocab_size)
        # Full probability distribution.
        output = self.softmax(output)

        tgt_out = tgt_out.contiguous().view(-1,1)
        label_mask = (tgt_out > 0).float().to(torch.device(output.device))
        one_hot = torch.zeros(label_mask.size(0),  self.vocab_size). \
            to(torch.device(output.device)). \
            scatter_(1, tgt_out, 1.0)

        numerator = -torch.sum(output * one_hot, 1)
        label_mask = label_mask.contiguous().view(-1)
        tgt_out = tgt_out.contiguous().view(-1)
        numerator = torch.sum(label_mask * numerator)
        denominator = torch.sum(label_mask) + 1e-6
        loss = numerator / denominator
        correct = torch.sum(label_mask * (output.argmax(dim=-1).eq(tgt_out)).float())

        return loss, correct, denominator
