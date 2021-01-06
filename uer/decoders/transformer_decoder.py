import torch
import torch.nn as nn
from uer.layers import *
from uer.layers.transformer import TransformerDecoderLayer
from uer.layers.layer_norm import LayerNorm


class TransformerDecoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """
    def __init__(self, args):
        super(TransformerDecoder, self).__init__()
        self.layers_num = args.layers_num
        self.layernorm_positioning = args.layernorm_positioning
        self.transformer_decoder = nn.ModuleList([
            TransformerDecoderLayer(args) for _ in range(self.layers_num)
        ])
        if self.layernorm_positioning == "pre":
            self.layer_norm = LayerNorm(args.hidden_size)

    def forward(self, memory_bank, emb, additional_info):
        """
        Args:
            memory_bank: [batch_size x seq_length x emb_size]
            emb: [batch_size x seq_length x emb_size]
        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """
        _, src_seq_length, _ = memory_bank.size()
        batch_size, tgt_seq_length, _ = emb.size()

        mask_encoder = (additional_info[0] > 0). \
                unsqueeze(1). \
                repeat(1, tgt_seq_length, 1). \
                unsqueeze(1)
        mask_encoder = mask_encoder.float()
        mask_encoder = (1.0 - mask_encoder) * -10000.0

        mask_decoder = torch.ones(tgt_seq_length, tgt_seq_length, device=emb.device)
        mask_decoder = torch.tril(mask_decoder)
        mask_decoder = (1.0 - mask_decoder) * -10000
        mask_decoder = mask_decoder.repeat(batch_size, 1, 1, 1)

        hidden = emb

        for i in range(self.layers_num):
            hidden = self.transformer_decoder[i](hidden, memory_bank, mask_decoder, mask_encoder)

        if self.layernorm_positioning == "pre":
            return self.layer_norm(hidden)
        else:
            return hidden

