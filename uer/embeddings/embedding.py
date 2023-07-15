import torch.nn as nn
import torch
from uer.layers.layer_norm import LayerNorm


class Embedding(nn.Module):
    def __init__(self, args):
        super(Embedding, self).__init__()
        self.embedding_name_list = []
        self.dropout = nn.Dropout(args.dropout)
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        if not self.remove_embedding_layernorm and "dual" not in args.embedding:
            self.layer_norm = LayerNorm(args.emb_size)

    def update(self, embedding, embedding_name):
        setattr(self, embedding_name, embedding)
        self.embedding_name_list.append(embedding_name)

    def forward(self, src, seg):
        if self.embedding_name_list[0] == "dual":
            return self.dual(src, seg)

        for i, embedding_name in enumerate(self.embedding_name_list):
            embedding = getattr(self, embedding_name)

            if i == 0:
                emb = embedding(src, seg)
            else:
                emb += embedding(src, seg)

        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb
