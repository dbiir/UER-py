import torch
import torch.nn as nn


class PosEmbedding(nn.Module):
    """
    Learnable Position Embedding
    """

    def __init__(self, args, _):
        super(PosEmbedding, self).__init__()
        if "speech" in args.embedding:
            self.max_seq_length = max(args.max_seq_length, args.max_audio_frames)
        else:
            self.max_seq_length = args.max_seq_length
        self.embedding = nn.Embedding(self.max_seq_length, args.emb_size)

    def forward(self, _, seg):
        """
        Args:
            src: [batch_size x seq_length]
            seg: [batch_size x seq_length]
        Returns:
            emb: [batch_size x seq_length x hidden_size]
        """

        seq_length = seg.size(1)
        batch_size = seg.size(0)
        device = seg.device

        pos_emb = self.embedding(
            torch.arange(0, seq_length, device=device, dtype=torch.long)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        return pos_emb
