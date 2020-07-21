# coding: utf-8
import torch
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm
from uer.layers.position_ffn import PositionwiseFeedForward


class DenseAttention(nn.Module):
    """
    Dense attention layer in DenseSynthesizer
    """

    def __init__(self, seq_length, hidden_size, dropout):
        super(DenseAttention, self).__init__()

        # Function F() in Dense format
        # Note: 根据论文无法知道是哪个一个Linear layer将hidden_size转换成seq_length，先随便试一试
        self.F_linear_1 = nn.Linear(hidden_size, seq_length)
        self.relu = nn.ReLU()
        self.F_linear_2 = nn.Linear(seq_length, seq_length) 
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # Function G() in Dense format
        self.G_linear = nn.Linear(hidden_size, hidden_size) 

        # Final linear
        self.final_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden, mask):
        """
        Args:
            hidden: [batch_size x seq_length x hidden_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, hidden_size = hidden.size()

        scores = self.relu(self.F_linear_1(hidden))
        scores = self.F_linear_2(scores)
        scores = scores + mask.squeeze(1)
        probs = self.softmax(scores)
        probs = self.dropout(probs)

        values = self.G_linear(hidden)

        output = torch.matmul(probs, values)
        output = self.final_linear(output)

        return output


class RandomAttention(nn.Module):
    
    def __init__(self, seq_length, hidden_size, dropout):
        super(RandomAttention, self).__init__()

        self.R = nn.Parameter(torch.randn(seq_length, seq_length, dtype=torch.float32))
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # Function G() in Dense format
        self.G_linear = nn.Linear(hidden_size, hidden_size) 

        # Final linear
        self.final_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden, mask):
        """
        Args:
            hidden: [batch_size x seq_length x hidden_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, hidden_size = hidden.size()

        scores = self.R + mask.squeeze(1)
        probs = self.softmax(scores)
        probs = self.dropout(probs)

        values = self.G_linear(hidden)

        output = torch.matmul(probs, values)
        output = self.final_linear(output)

        return output


class ISynthesizer(nn.Module):
    """
    Abstract class for Synthesizer
    """

    def __init__(self, args):
        super(ISynthesizer, self).__init__()

        self.att = None
        self.dropout_1 = nn.Dropout(args.dropout)
        self.layer_norm_1 = LayerNorm(args.hidden_size)
        # Feed forward layer.
        self.feed_forward = PositionwiseFeedForward(
            args.hidden_size, args.feedforward_size
        )
        self.dropout_2 = nn.Dropout(args.dropout)
        self.layer_norm_2 = LayerNorm(args.hidden_size)

        if self.__class__.__name__ == 'ISynthesizer':
            raise Exception("ISynthesizer cannot be instantiated.")

    def forward(self, hidden, mask):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        inter = self.dropout_1(self.att(hidden, mask))
        inter = self.layer_norm_1(inter + hidden)
        output = self.dropout_2(self.feed_forward(inter))
        output = self.layer_norm_2(output + inter)  
        return output


class DenseSynthesizer(ISynthesizer):
    """
    Dense Synthesizer layer.
    @ref: https://arxiv.org/abs/2005.00743
    """

    def __init__(self, args):
        super(DenseSynthesizer, self).__init__(args)

        # dense attention
        self.att = DenseAttention(
                args.seq_length, args.hidden_size, args.dropout
            )


class RandomSynthesizer(ISynthesizer):
    """
    Random Synthesizer layer.
    @ref: https://arxiv.org/abs/2005.00743
    """

    def __init__(self, args):
        super(RandomSynthesizer, self).__init__(args)

        # dense attention
        self.att = RandomAttention(
                args.seq_length, args.hidden_size, args.dropout
            )
        

SYNT_TYPE_MAP = {
        'dense': DenseSynthesizer,
        'random': RandomSynthesizer
    }



        
