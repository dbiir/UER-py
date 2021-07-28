from uer.decoders import *
from uer.layers import *
from uer.targets import *


class Seq2seqTarget(LmTarget):
    """
    """

    def __init__(self, args, vocab_size):
        super(Seq2seqTarget, self).__init__(args, vocab_size)

        self.embedding = str2embedding[args.tgt_embedding](args, vocab_size)

        self.decoder = str2decoder[args.decoder](args)

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
        tgt_in, tgt_out, seg = tgt

        emb = self.embedding(tgt_in, None)

        hidden = self.decoder(memory_bank, emb, (seg,))

        # Language modeling (LM) with full softmax prediction.
        loss, correct, denominator = self.lm(hidden, tgt_out)

        return loss, correct, denominator
