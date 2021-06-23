import math
import torch
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm
from uer.utils.act_fun import gelu


class ClipTarget(nn.Module):
    """
    """
    def __init__(self, args, vocab_size):
        super(ClipTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size

        self.criterion_img = nn.CrossEntropyLoss()
        self.criterion_text = nn.CrossEntropyLoss()
        self.softmax = nn.LogSoftmax(dim=-1)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, memory_bank, tgt):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size]

        Returns:
            loss: Classification loss.
            correct: Number of sentences that are predicted correctly.
        """
        features_text, features_image = memory_bank

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * torch.matmul(features_text, features_image.transpose(-2, -1))
        logits_per_text = logit_scale * torch.matmul(features_image , features_text.transpose(-2, -1))

        tgt = torch.arange(self.batch_size, device = logits_per_image.device, dtype=torch.long)
        loss = (self.criterion_img(logits_per_image, tgt) + self.criterion_text(logits_per_text, tgt)) / 2
        correct = self.softmax(logits_per_image).argmax(dim=-1).eq(tgt).sum()

        return loss, correct
