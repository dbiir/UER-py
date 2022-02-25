import torch
import torch.nn as nn


class SpTarget(nn.Module):

    def __init__(self, args, vocab_size):
        super(SpTarget, self).__init__()

        self.sp_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.sp_linear_2 = nn.Linear(args.hidden_size, 2)

        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()

    def forward(self, memory_bank, tgt, seg):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: tuple with tgt_mlm [batch_size x seq_length] and tgt_sop [batch_size]

        Returns:
            loss_sop: Sentence order prediction loss.
            correct_sop: Number of sentences that are predicted correctly.
        """
        output_sp = torch.tanh(self.sp_linear_1(memory_bank[:, 0, :]))
        output_sp = self.sp_linear_2(output_sp)
        loss_sp = self.criterion(self.softmax(output_sp), tgt)
        correct_sp = self.softmax(output_sp).argmax(dim=-1).eq(tgt).sum()

        return loss_sp, correct_sp
