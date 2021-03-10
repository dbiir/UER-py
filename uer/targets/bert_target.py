import torch
import torch.nn as nn
from uer.targets import *


class BertTarget(MlmTarget):
    """
    BERT exploits masked language modeling (MLM)
    and next sentence prediction (NSP) for pretraining.
    """

    def __init__(self, args, vocab_size):
        super(BertTarget, self).__init__(args, vocab_size)
        # NSP.
        self.nsp_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.nsp_linear_2 = nn.Linear(args.hidden_size, 2)

    def forward(self, memory_bank, tgt):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: tuple with tgt_mlm [batch_size x seq_length] and tgt_nsp [batch_size]

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
        output_nsp = torch.tanh(self.nsp_linear_1(memory_bank[:, 0, :]))
        output_nsp = self.nsp_linear_2(output_nsp)
        loss_nsp = self.criterion(self.softmax(output_nsp), tgt_nsp)
        correct_nsp = self.softmax(output_nsp).argmax(dim=-1).eq(tgt_nsp).sum()

        return loss_mlm, loss_nsp, correct_mlm, correct_nsp, denominator
