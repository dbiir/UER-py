# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
from uer.utils.constants import *
from uer.utils.subword import *


class Model(nn.Module):
    """
    BertModel consists of three parts:
        - embedding: token embedding, position embedding, segment embedding
        - encoder: multi-layer transformer encoders
        - target: mlm and nsp tasks
    """
    def __init__(self, args, embedding, encoder, target, subencoder = None):
        super(Model, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.target = target
        
        # Subencoder.
        if subencoder is not None:
            self.vocab, self.sub_vocab = args.vocab, args.sub_vocab
            self.subword_type = args.subword_type
            self.subencoder = subencoder
        else:
            self.subencoder = None

    def forward(self, src, tgt, seg):
        # [batch_size, seq_length, emb_size]

        emb = self.embedding(src, seg) 

        if self.subencoder is not None:
            sub_ids = word2sub(src, self.vocab, self.sub_vocab, self.subword_type)
            emb = emb + self.subencoder(sub_ids).contiguous().view(*emb.size())

        output = self.encoder(emb, seg)            

        loss_info = self.target(output, tgt)
            
        return loss_info
