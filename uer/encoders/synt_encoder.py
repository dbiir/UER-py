# -*- encoding:utf-8 -*-
import torch.nn as nn
from uer.layers.synthesizer import DenseSynthesizer, RandomSynthesizer
from uer.layers.synthesizer import SYNT_TYPE_MAP




class SyntEncoder(nn.Module):
    """
    Synthesizer encoder exploits 12 or 24 synthesizer layers to extract features.
    """
    def __init__(self, args):
        super(SyntEncoder, self).__init__()
        self.layers_num = args.layers_num
        
        try:
            synt_type = SYNT_TYPE_MAP[args.synt_type]
        except KeyError:
            raise KeyError('synt_type must be in ({})'.format(SYNT_TYPE_MAP.keys()))

        self.synthesizer = nn.ModuleList([
            synt_type(args) for _ in range(self.layers_num)
        ])
        
    def forward(self, emb, seg):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]

        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """

        seq_length = emb.size(1)
        # Generate mask according to segment indicators.
        # mask: [batch_size x 1 x seq_length x seq_length]
        mask = (seg > 0). \
                unsqueeze(1). \
                repeat(1, seq_length, 1). \
                unsqueeze(1)

        mask = mask.float()
        mask = (1.0 - mask) * -10000.0

        hidden = emb
        for i in range(self.layers_num):
            hidden = self.synthesizer[i](hidden, mask)
        return hidden
