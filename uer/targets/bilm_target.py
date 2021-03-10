from uer.targets import *
from uer.utils.misc import *


class BilmTarget(LmTarget):
    """
    """
    def __init__(self, args, vocab_size):
        args.hidden_size = args.hidden_size // 2
        super(BilmTarget, self).__init__(args, vocab_size)

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
        
        assert type(tgt) == tuple
        tgt_forward, tgt_backward = tgt[0], tgt[1]
        # Forward.
        loss_forward, correct_forward, denominator_forward = \
            self.lm(memory_bank[:, :, :self.hidden_size], tgt_forward)
        # Backward.
        loss_backward, correct_backward, denominator_backward = \
            self.lm(memory_bank[:, :, self.hidden_size:], tgt_backward)

        return loss_forward, loss_backward, correct_forward, correct_backward, denominator_backward
