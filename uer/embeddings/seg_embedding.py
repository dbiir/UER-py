import torch.nn as nn


class SegEmbedding(nn.Module):
    """
    BERT Segment Embedding
    """
    def __init__(self, args, _):
        super(SegEmbedding, self).__init__()
        self.embedding = nn.Embedding(3, args.emb_size)

    def forward(self, _, seg):
        """
        Args:
            seg: [batch_size x seq_length]
        Returns:
            emb: [batch_size x seq_length x hidden_size]
        """

        seg_emb = self.embedding(seg)

        return seg_emb
