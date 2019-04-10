# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    BertModel consists of three parts:
        - embedding: token embedding, position embedding, segment embedding
        - encoder: multi-layer transformer encoders
        - target: mlm and nsp tasks
    """
    def __init__(self, args, embedding, encoder, target):
        super(Model, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.target = target

    def forward(self, src, tgt, seg):
        # [batch_size, seq_length, emb_size]
        emb = self.embedding(src, seg) 

        output = self.encoder(emb, seg)            

        loss_info = self.target(output, tgt)
            
        return loss_info
