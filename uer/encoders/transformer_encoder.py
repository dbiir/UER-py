import torch
import torch.nn as nn
from uer.layers.transformer import TransformerLayer
from uer.layers.layer_norm import LayerNorm, T5LayerNorm
from uer.layers.relative_position_embedding import RelativePositionEmbedding

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
        self.relative_position_embedding = args.relative_position_embedding
        self.has_residual_attention = args.has_residual_attention
        if "deepspeed_checkpoint_activations" in args:
            self.deepspeed_checkpoint_activations = args.deepspeed_checkpoint_activations
            self.deepspeed_checkpoint_layers_num = args.deepspeed_checkpoint_layers_num
        else:
            self.deepspeed_checkpoint_activations = False

        has_bias = bool(1 - args.remove_transformer_bias)

        if self.factorized_embedding_parameterization:
            self.linear = nn.Linear(args.emb_size, args.hidden_size)

        if self.parameter_sharing:
            self.transformer = TransformerLayer(args)
        else:
            self.transformer = nn.ModuleList(
                [TransformerLayer(args) for _ in range(self.layers_num)]
            )
        if self.layernorm_positioning == "pre":
            if args.layernorm == "t5":
                self.layer_norm = T5LayerNorm(args.hidden_size)
            else:
                self.layer_norm = LayerNorm(args.hidden_size)

        if self.relative_position_embedding:
            self.relative_pos_emb = RelativePositionEmbedding(bidirectional=True, heads_num=args.heads_num,
                                                              num_buckets=args.relative_attention_buckets_num)


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
        elif self.mask == "causal":
            mask = torch.ones(seq_length, seq_length, device=emb.device)
            mask = torch.tril(mask)
            mask = (1.0 - mask) * -10000
            mask = mask.repeat(batch_size, 1, 1, 1)
        else:
            mask_a = (seg == 1). \
                unsqueeze(1). \
                repeat(1, seq_length, 1). \
                unsqueeze(1).float()

            mask_b = (seg > 0). \
                unsqueeze(1). \
                repeat(1, seq_length, 1). \
                unsqueeze(1).float()

            mask_tril = torch.ones(seq_length, seq_length, device=emb.device)
            mask_tril = torch.tril(mask_tril)
            mask_tril = mask_tril.repeat(batch_size, 1, 1, 1)

            mask = (mask_a + mask_b + mask_tril >= 2).float()
            mask = (1.0 - mask) * -10000.0

        hidden = emb

        if self.relative_position_embedding:
            position_bias = self.relative_pos_emb(hidden, hidden)
        else:
            position_bias = None

        prev_attn = None

        if self.deepspeed_checkpoint_activations:
            from deepspeed import checkpointing

            def custom(start, end):
                def custom_forward(*inputs):
                    x_, y_, position_bias_ = inputs
                    for index in range(start, end):
                        if self.parameter_sharing:
                            x_, y_ = self.transformer(x_, mask, position_bias=position_bias_,
                                                             has_residual_attention=self.has_residual_attention,
                                                             prev_attn=y_)
                        else:
                            x_, y_ = self.transformer[index](x_, mask, position_bias=position_bias_,
                                                             has_residual_attention=self.has_residual_attention,
                                                             prev_attn=y_)
                    return x_, y_

                return custom_forward
            l = 0
            while l < self.layers_num:
                hidden, prev_attn = checkpointing.checkpoint(custom(l, l + self.deepspeed_checkpoint_layers_num),
                                                             hidden, prev_attn, position_bias)
                l += self.deepspeed_checkpoint_layers_num
        else:
            for i in range(self.layers_num):
                if self.parameter_sharing:
                    hidden, prev_attn = self.transformer(hidden, mask, position_bias=position_bias,
                                                         has_residual_attention=self.has_residual_attention,
                                                         prev_attn=prev_attn)
                else:
                    hidden, prev_attn = self.transformer[i](hidden, mask, position_bias=position_bias,
                                                            has_residual_attention=self.has_residual_attention,
                                                            prev_attn=prev_attn)

        if self.layernorm_positioning == "pre":
            return self.layer_norm(hidden)
        else:
            return hidden
