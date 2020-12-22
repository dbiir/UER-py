import torch
import torch.nn as nn
from uer.layers.transformer import TransformerLayer
from uer.layers.layer_norm import LayerNorm

class TransformerEncoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.mask = args.mask
        self.layers_num = args.layers_num
        self.parameter_sharing = args.parameter_sharing
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.layernorm_positioning = args.layernorm_positioning

        if self.factorized_embedding_parameterization:
            self.linear = nn.Linear(args.emb_size, args.hidden_size)

        if self.parameter_sharing:
            self.transformer = TransformerLayer(args)
        else:
            self.transformer = nn.ModuleList([
                TransformerLayer(args) for _ in range(self.layers_num)
            ])
        if self.layernorm_positioning == "pre":
            self.layer_norm = LayerNorm(args.hidden_size)
        
    def forward(self, emb, seg):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]
        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """
        if self.factorized_embedding_parameterization:
            emb = self.linear(emb)

        batch_size, seq_length, _ = emb.size()
        # Generate mask according to segment indicators.
        # mask: [batch_size x 1 x seq_length x seq_length]
        if self.mask == "fully_visible":
            mask = (seg > 0). \
                unsqueeze(1). \
                repeat(1, seq_length, 1). \
                unsqueeze(1)
            mask = mask.float()
            mask = (1.0 - mask) * -10000.0
        else:
            mask = torch.ones(seq_length, seq_length, device=emb.device)
            mask = torch.tril(mask)
            mask = (1.0 - mask) * -10000
            mask = mask.repeat(batch_size, 1, 1, 1)

        hidden = emb

        for i in range(self.layers_num):
            if self.parameter_sharing:
                hidden = self.transformer(hidden, mask)
            else:
                hidden = self.transformer[i](hidden, mask)

        if self.layernorm_positioning == "pre":
            return self.layer_norm(hidden)
        else:
            return hidden
