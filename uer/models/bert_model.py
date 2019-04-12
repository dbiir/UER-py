# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn

class BertModel(nn.Module):
    """
    BertModel consists of three parts:
        - embedding: token embedding, position embedding, segment embedding
        - encoder: multi-layer transformer encoders
        - target: mlm and nsp tasks
    """
    def __init__(self, args, embedding, encoder, target):
        super(BertModel, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.target = target

    def forward(self, src, tgt_mlm, tgt_nsp, seg):
        # [batch_size, seq_length, emb_size]
        emb = self.embedding(src, seg) 
        seq_length = emb.size(1)
        # Generate mask according to segment indicators.
        mask = (seg > 0). \
                unsqueeze(1). \
                repeat(1, seq_length, 1). \
                unsqueeze(1)

        mask = mask.float()
        mask = (1.0 - mask) * -10000.0
        output = self.encoder(emb, mask)            

        loss_mlm, loss_nsp, correct_mlm, correct_nsp, \
            denominator = self.target(output, tgt_mlm, tgt_nsp)
            
        return loss_mlm, loss_nsp, correct_mlm, \
               correct_nsp, denominator
