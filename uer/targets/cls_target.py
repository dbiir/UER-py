import torch
import torch.nn as nn
from uer.utils.misc import pooling


class ClsTarget(nn.Module):
    """
    Classification Target
    """
    def __init__(self, args, vocab_size):
        super(ClsTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size

        self.pooling_type = args.pooling
        self.linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_2 = nn.Linear(args.hidden_size, args.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()

    def forward(self, memory_bank, tgt, seg):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size]

        Returns:
            loss: Classification loss.
            correct: Number of sentences that are predicted correctly.
        """

        output = pooling(memory_bank, seg, self.pooling_type)
        output = torch.tanh(self.linear_1(output))
        logits = self.linear_2(output)

        loss = self.criterion(self.softmax(logits), tgt)
        correct = self.softmax(logits).argmax(dim=-1).eq(tgt).sum()

        return loss, correct
