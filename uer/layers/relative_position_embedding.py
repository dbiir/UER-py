import math
import torch

class RelativePositionEmbedding(nn.Module):
    """ Relative Position Embedding
        https://arxiv.org/abs/1910.10683
        https://github.com/bojone/bert4keras/blob/db236eac110a67a587df7660f6a1337d5b2ef07e/bert4keras/layers.py#L663
    """
    def __init__(self, bidirectional = True, num_buckets = 32, max_distance = 128):
        super(RelativePositionEmbedding, self).__init__()
        self.num_buckets = num_buckets
        self.bidirectional = bidirectional
        self.max_distance = max_distance


    def forward(self, emb):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
        Returns:
            position_bias: [batch_size x 1 x seq_length x seq_length]
        """
        q_idxs = torch.arange(0, emb.size()[1])
        q_idxs = torch.unsqueeze(q_idxs, 1)
        v_idxs = torch.arange(0, emb.size()[1])
        v_idxs = torch.unsqueeze(v_idxs, 0)

        pos_ids = v_idxs - q_idxs

        num_buckets, max_distance = self.num_buckets, self.max_distance
        position_bias = 0
        n = -pos_ids
        if self.bidirectional:
            num_buckets //= 2
            position_bias += (n < 0).float() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.where(n < 0, torch.full_like(n, 0), n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (torch.log(n.float()) / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        val_if_large = torch.where(val_if_large < num_buckets - 1, torch.full_like(val_if_large, num_buckets - 1), val_if_large)
        position_bias += torch.where(is_small, n.float(), val_if_large)

        return position_bias.expand(emb.size()[0], 1, emb.size()[1], emb.size()[1])



