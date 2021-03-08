import torch
import torch.nn as nn
from uer.targets import *


class AlbertTarget(MlmTarget):
    """
    BERT exploits masked language modeling (MLM)
    and sentence order prediction (SOP) for pretraining.
    """

    def __init__(self, args, vocab_size):
        super(AlbertTarget, self).__init__(args, vocab_size)

        self.factorized_embedding_parameterization = True
        # SOP.
        self.sop_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.sop_linear_2 = nn.Linear(args.hidden_size, 2)

    def forward(self, memory_bank, tgt):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: tuple with tgt_mlm [batch_size x seq_length] and tgt_sop [batch_size]

        Returns:
            loss_mlm: Masked language model loss.
            loss_sop: Sentence order prediction loss.
            correct_mlm: Number of words that are predicted correctly.
            correct_sop: Number of sentences that are predicted correctly.
            denominator: Number of masked words.
        """

        # Masked language model (MLM).
        assert type(tgt) == tuple
        tgt_mlm, tgt_sop = tgt[0], tgt[1]

        loss_mlm, correct_mlm, denominator = self.mlm(memory_bank, tgt_mlm)

        # Sentence order prediction (SOP).
        output_sop = torch.tanh(self.sop_linear_1(memory_bank[:, 0, :]))
        output_sop = self.sop_linear_2(output_sop)
        loss_sop = self.criterion(self.softmax(output_sop), tgt_sop)
        correct_sop = self.softmax(output_sop).argmax(dim=-1).eq(tgt_sop).sum()

        return loss_mlm, loss_sop, correct_mlm, correct_sop, denominator
