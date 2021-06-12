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
        self.batch_size = args.batch_size


        self.criterion_img = nn.CrossEntropyLoss()
        self.criterion_text = nn.CrossEntropyLoss()

    def forward(self, memory_bank, tgt):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size]

        Returns:
            loss: Classification loss.
            correct: Number of sentences that are predicted correctly.
        """
        logits_per_image, logits_per_text = memory_bank

        tgt = torch.arange(self.batch_size).long()
        loss = (self.criterion_img(logits_per_image, tgt) + self.criterion_text(logits_per_text, tgt)) / 2
        correct = self.softmax(logits_per_image).argmax(dim=-1).eq(tgt).sum()

        return loss, correct
