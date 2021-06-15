import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Pretraining models consist of three parts:
        - embedding
        - encoder
        - target
    """
    def __init__(self, args, embedding, encoder, target):
        super(Model, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.target = target

        if args.target in ["bert", "mlm", "albert"] and args.tie_weights:
            self.target.mlm_linear_2.weight = self.embedding.word_embedding.weight
        elif args.target in ["lm", "t5", "gsg", "bart"] and args.tie_weights:
            self.target.output_layer.weight = self.embedding.word_embedding.weight

        if args.target in ["t5", "gsg", "bart"] and args.share_embedding:
            self.target.embedding.word_embedding.weight = self.embedding.word_embedding.weight

    def forward(self, src, tgt, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        loss_info = self.target(output, tgt)
        return loss_info
