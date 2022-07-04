import torch
import torch.nn as nn
from uer.layers import *
from uer.layers.transformer import TransformerDecoderLayer
from uer.layers.layer_norm import LayerNorm, T5LayerNorm
from uer.layers.relative_position_embedding import RelativePositionEmbedding


class TransformerDecoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """

    def __init__(self, args):
        super(TransformerDecoder, self).__init__()
        self.layers_num = args.layers_num
        self.layernorm_positioning = args.layernorm_positioning
        self.relative_position_embedding = args.relative_position_embedding
        self.transformer_decoder = nn.ModuleList(
            [TransformerDecoderLayer(args) for _ in range(self.layers_num)]
        )
        if "deepspeed_checkpoint_activations" in args:
            self.deepspeed_checkpoint_activations = args.deepspeed_checkpoint_activations
            self.deepspeed_checkpoint_layers_num = args.deepspeed_checkpoint_layers_num
        else:
            self.deepspeed_checkpoint_activations = False

        has_bias = bool(1 - args.remove_transformer_bias)

        if self.layernorm_positioning == "pre":
            if args.layernorm == "t5":
                self.layer_norm = T5LayerNorm(args.hidden_size)
            else:
                self.layer_norm = LayerNorm(args.hidden_size)

        if self.relative_position_embedding:
            self.self_pos_emb = RelativePositionEmbedding(bidirectional=False, heads_num=args.heads_num,
                                                          num_buckets=args.relative_attention_buckets_num)


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

        if self.relative_position_embedding:
            self_position_bias = self.self_pos_emb(hidden, hidden)
        else:
            self_position_bias = None

        if self.deepspeed_checkpoint_activations:
            from deepspeed import checkpointing

            def custom(start, end):
                def custom_forward(*inputs):
                    x_, memory_bank_, self_position_bias_ = inputs
                    for index in range(start, end):
                        x_ = self.transformer_decoder[index](x_, memory_bank_, mask_decoder, mask_encoder, self_position_bias_, None)
                    return x_

                return custom_forward
            l = 0
            while l < self.layers_num:
                hidden = checkpointing.checkpoint(custom(l, l + self.deepspeed_checkpoint_layers_num), hidden, memory_bank, self_position_bias)
                l += self.deepspeed_checkpoint_layers_num
        else:
            for i in range(self.layers_num):
                hidden = self.transformer_decoder[i](hidden, memory_bank, mask_decoder, mask_encoder, self_position_bias, None)

        if self.layernorm_positioning == "pre":
            return self.layer_norm(hidden)
        else:
            return hidden