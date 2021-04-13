import math
import torch
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm
from uer.utils.act_fun import gelu


class ClsTarget(nn.Module):
    """
    """
    def __init__(self, args, vocab_size):
        super(ClsTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size

        self.pooling = args.pooling
        self.linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_2 = nn.Linear(args.hidden_size, args.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()


    def forward(self, memory_bank, tgt):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size]

        Returns:
            loss: Classification loss.
            correct: Number of sentences that are predicted correctly.
        """

        if self.pooling == "mean":
            output = torch.mean(memory_bank, dim=1)
        elif self.pooling == "max":
            output = torch.max(memory_bank, dim=1)[0]
        elif self.pooling == "last":
            output = memory_bank[:, -1, :]
        else:
            output = memory_bank[:, 0, :]
        output = torch.tanh(self.linear_1(output))
        logits = self.linear_2(output)

        loss = self.criterion(self.softmax(logits), tgt)
        correct = self.softmax(logits).argmax(dim=-1).eq(tgt).sum()

        return loss, correct
