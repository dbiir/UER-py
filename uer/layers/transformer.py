import torch.nn as nn
from uer.layers.layer_norm import LayerNorm
from uer.layers.position_ffn import PositionwiseFeedForward
from uer.layers.multi_headed_attn import MultiHeadedAttention


class TransformerLayer(nn.Module):
    """
    Transformer layer mainly consists of two parts:
    multi-headed self-attention and feed forward layer.
    """
    def __init__(self, args):
        super(TransformerLayer, self).__init__()

        self.layernorm_positioning = args.layernorm_positioning

        # Multi-headed self-attention.
        self.self_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, args.dropout
        )
        self.dropout_1 = nn.Dropout(args.dropout)
        self.layer_norm_1 = LayerNorm(args.hidden_size)
        # Feed forward layer.
        self.feed_forward = PositionwiseFeedForward(
            args.hidden_size, args.feedforward_size, args.hidden_act
        )
        self.dropout_2 = nn.Dropout(args.dropout)
        self.layer_norm_2 = LayerNorm(args.hidden_size)

    def forward(self, hidden, mask):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        if self.layernorm_positioning == "post":
            inter = self.dropout_1(self.self_attn(hidden, hidden, hidden, mask))
            inter = self.layer_norm_1(inter + hidden)
            output = self.dropout_2(self.feed_forward(inter))
            output = self.layer_norm_2(output + inter)
        else:
            inter = self.layer_norm_1(hidden)
            inter = self.dropout_1(self.self_attn(inter, inter, inter, mask))
            hidden = hidden + inter
            output = self.layer_norm_2(hidden)
            output = self.dropout_2(self.feed_forward(output)) + hidden
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, args):
        super(TransformerDecoderLayer, self).__init__()

        self.layernorm_positioning = args.layernorm_positioning

        # Multi-headed self-attention.
        self.self_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, args.dropout
        )
        self.dropout_1 = nn.Dropout(args.dropout)
        self.layer_norm_1 = LayerNorm(args.hidden_size)

        # Multi-headed context-attention.
        self.context_attn = MultiHeadedAttention(
            args.hidden_size, args.heads_num, args.dropout
        )
        self.dropout_2 = nn.Dropout(args.dropout)
        self.layer_norm_2 = LayerNorm(args.hidden_size)

        # Feed forward layer.
        self.feed_forward = PositionwiseFeedForward(
            args.hidden_size, args.feedforward_size, args.hidden_act
        )
        self.dropout_3 = nn.Dropout(args.dropout)
        self.layer_norm_3 = LayerNorm(args.hidden_size)

    def forward(self, hidden, encoder_hidden, mask_decoder, mask_encoder):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            hidden: [batch_size x seq_length x emb_size]
            mask_encoder: [batch_size x 1 x seq_length x seq_length]
            mask_decoder: [batch_size x 1 x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        if self.layernorm_positioning == "post":
            inter = self.dropout_1(self.self_attn(hidden, hidden, hidden, mask_decoder))
            inter_norm = self.layer_norm_1(inter + hidden)
            mid = self.dropout_2(self.context_attn(encoder_hidden, encoder_hidden, inter_norm, mask_encoder))
            mid_norm = self.layer_norm_2(mid + inter_norm)
            output = self.dropout_3(self.feed_forward(mid_norm))
            output = self.layer_norm_3(output + mid_norm)
        else:
            hidden_norm = self.layer_norm_1(hidden)
            inter = self.dropout_1(self.self_attn(hidden_norm, hidden_norm, hidden_norm, mask_decoder))
            inter = inter + hidden
            inter_norm = self.layer_norm_2(inter)
            mid = self.dropout_2(self.context_attn(encoder_hidden, encoder_hidden, inter_norm, mask_encoder))
            mid = mid + inter
            mid_norm = self.layer_norm_3(mid)
            output = self.dropout_3(self.feed_forward(mid_norm)) + mid
        return output


#class GptBlock(nn.Module):
#    def __init__(self, args):
#        super(GptBlock, self).__init__()

#        # Multi-headed self-attention.
#        self.self_attn = MultiHeadedAttention(
#            args.hidden_size, args.heads_num, args.dropout
#        )
#        self.layer_norm_1 = LayerNorm(args.hidden_size)
#        # Feed forward layer.
#        self.feed_forward = PositionwiseFeedForward(
#            args.hidden_size, args.feedforward_size, args.hidden_act
#        )
#        self.layer_norm_2 = LayerNorm(args.hidden_size)

#    def forward(self, hidden, mask):
#        """
#        Args:
#            hidden: [batch_size x seq_length x emb_size]
#            mask: [batch_size x 1 x seq_length x seq_length]
#        Returns:
#            output: [batch_size x seq_length x hidden_size]
#        """
#        inter = self.layer_norm_1(hidden)
#        inter = self.self_attn(inter, inter, inter, mask)
#        hidden = hidden + inter
#        output = self.layer_norm_2(hidden)
#        output = self.feed_forward(output)
        
#        return output + hidden
