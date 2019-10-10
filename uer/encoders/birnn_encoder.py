# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
from uer.utils.misc import *

class BilstmEncoder(nn.Module):
    def __init__(self, args):
        super(BilstmEncoder, self).__init__()

        assert args.hidden_size % 2 == 0 
        self.hidden_size= args.hidden_size // 2
        
        self.layers_num = args.layers_num

        self.rnn_forward = nn.LSTM(input_size=args.emb_size,
                           hidden_size=self.hidden_size,
                           num_layers=args.layers_num,
                           dropout=args.dropout,
                           batch_first=True)

        self.rnn_backward = nn.LSTM(input_size=args.emb_size,
                           hidden_size=self.hidden_size,
                           num_layers=args.layers_num,
                           dropout=args.dropout,
                           batch_first=True)

        self.drop = nn.Dropout(args.dropout)

    def forward(self, emb, seg):
        # Forward.
        emb_forward = emb
        hidden_forward = self.init_hidden(emb_forward.size(0), emb_forward.device)
        output_forward, hidden_forward = self.rnn_forward(emb_forward, hidden_forward) 
        output_forward = self.drop(output_forward)

        # Backward.
        emb_backward = flip(emb, 1)
        hidden_backward = self.init_hidden(emb_backward.size(0), emb_backward.device)
        output_backward, hidden_backward = self.rnn_backward(emb_backward, hidden_backward) 
        output_backward = self.drop(output_backward)
        output_backward = flip(output_backward, 1)

        return torch.cat([output_forward, output_backward], 2)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device),
                torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device))
