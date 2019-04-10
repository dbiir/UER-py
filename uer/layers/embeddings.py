# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm


class BertEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size):
        super(BertEmbedding, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.emb_norm = LayerNorm(args.emb_size)
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.positon_embedding = PositionEmbedding(args.emb_size)
        self.segment_embedding = SegmentEmbedding(args.emb_size)

    def forward(self, src, seg):
        word_emb = self.word_embedding(src)
        pos_emb = self.positon_embedding(word_emb.size(0), word_emb.size(1), word_emb.device)
        seg_emb = self.segment_embedding(seg)
        emb = word_emb + pos_emb + seg_emb
        emb = self.dropout(self.emb_norm(emb))
        return emb


class SegmentEmbedding(nn.Module):
    """
    Distinguish the first sentence and the second sentence.
    E.g. [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0]
    """
    def __init__(self, emb_size):
        super(SegmentEmbedding, self).__init__()
        self.embedding_layer = nn.Embedding(3, emb_size)

    def forward(self, ids):
        return self.embedding_layer(ids)


class PositionEmbedding(nn.Module):
    """
    We assume that the sequence length is less than 512.
    """
    def __init__(self, emb_size, max_length=512):
        super(PositionEmbedding, self).__init__()
        self.max_length = max_length
        self.embedding_layer = nn.Embedding(max_length, emb_size)

    def forward(self, batch_size, seq_length, device):
        assert(seq_length <= self.max_length)
        ids = torch.arange(0, seq_length, device=device, dtype=torch.long)
        ids = ids.unsqueeze(0).repeat(batch_size, 1)
        emb = self.embedding_layer(ids)
        return emb

