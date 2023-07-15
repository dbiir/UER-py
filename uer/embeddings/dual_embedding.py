from argparse import Namespace
import torch.nn as nn
import copy
from uer.layers.layer_norm import LayerNorm


class DualEmbedding(nn.Module):
    """
    """
    def __init__(self, args, vocab_size):
        super(DualEmbedding, self).__init__()
        from uer.embeddings import str2embedding
        from uer.embeddings.embedding import Embedding

        stream_0_args = copy.deepcopy(vars(args))
        stream_0_args.update(args.stream_0)
        stream_0_args = Namespace(**stream_0_args)
        self.embedding_0 = Embedding(stream_0_args)
        for embedding_name in stream_0_args.embedding:
            self.embedding_0.update(str2embedding[embedding_name](stream_0_args, vocab_size), embedding_name)
        self.stream_0_remove_embedding_layernorm = stream_0_args.remove_embedding_layernorm
        if not self.stream_0_remove_embedding_layernorm:
            self.stream_0_layer_norm = LayerNorm(stream_0_args.emb_size)

        stream_1_args = copy.deepcopy(vars(args))
        stream_1_args.update(args.stream_1)
        stream_1_args = Namespace(**stream_1_args)
        self.embedding_1 = Embedding(stream_1_args)
        for embedding_name in stream_1_args.embedding:
            self.embedding_1.update(str2embedding[embedding_name](stream_1_args, vocab_size), embedding_name)
        self.stream_1_remove_embedding_layernorm = stream_1_args.remove_embedding_layernorm
        if not self.stream_1_remove_embedding_layernorm:
            self.stream_1_layer_norm = LayerNorm(stream_1_args.emb_size)
        self.dropout = nn.Dropout(args.dropout)

        if args.tie_weights:
            self.embedding_0 = self.embedding_1

    def forward(self, src, seg):
        """
        Args:
            src: ([batch_size x seq_length], [batch_size x seq_length])
            seg: ([batch_size x seq_length], [batch_size x seq_length])
        Returns:
            emb_0: [batch_size x seq_length x hidden_size]
            emb_1: [batch_size x seq_length x hidden_size]
        """
        emb_0 = self.get_embedding_0(src[0], seg[0])
        emb_1 = self.get_embedding_1(src[1], seg[1])

        emb_0 = self.dropout(emb_0)
        emb_1 = self.dropout(emb_1)

        return emb_0, emb_1

    def get_embedding_0(self, src, seg):
        emb = self.embedding_0(src, seg)
        if not self.stream_0_remove_embedding_layernorm:
            emb = self.stream_0_layer_norm(emb)
        return emb

    def get_embedding_1(self, src, seg):
        emb = self.embedding_1(src, seg)
        if not self.stream_1_remove_embedding_layernorm:
            emb = self.stream_1_layer_norm(emb)
        return emb
