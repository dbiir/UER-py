# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn


class LstmEncoder(nn.Module):
    def __init__(self, args):
        super(LstmEncoder, self).__init__()

        self.bidirectional = args.bidirectional
        if self.bidirectional:
            assert args.hidden_size % 2 == 0 
            self.hidden_size= args.hidden_size // 2
        else:
            self.hidden_size= args.hidden_size
        
        self.layers_num = args.layers_num

        self.rnn = nn.LSTM(input_size=args.emb_size,
                           hidden_size=self.hidden_size,
                           num_layers=args.layers_num,
                           dropout=args.dropout,
                           batch_first=True,
                           bidirectional=self.bidirectional)

        self.drop = nn.Dropout(args.dropout)

    def forward(self, emb, seg):
        hidden = self.init_hidden(emb.size(0), emb.device)
        output, hidden = self.rnn(emb, hidden) 
        output = self.drop(output) 
        return output

    def init_hidden(self, batch_size, device):
        if self.bidirectional:
            return (torch.zeros(self.layers_num*2, batch_size, self.hidden_size, device=device),
                    torch.zeros(self.layers_num*2, batch_size, self.hidden_size, device=device))
        else:
            return (torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device),
                    torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device))


class GruEncoder(nn.Module):
    def __init__(self, args):
        super(GruEncoder, self).__init__()

        self.bidirectional = args.bidirectional
        if self.bidirectional:
            assert args.hidden_size % 2 == 0 
            self.hidden_size= args.hidden_size // 2
        else:
            self.hidden_size= args.hidden_size

        self.layers_num = args.layers_num

        self.rnn = nn.GRU(input_size=args.emb_size,
                           hidden_size=self.hidden_size,
                           num_layers=args.layers_num,
                           dropout=args.dropout,
                           batch_first=True,
                           bidirectional=self.bidirectional)

        self.drop = nn.Dropout(args.dropout)

    def forward(self, emb, seg):
        hidden = self.init_hidden(emb.size(0), emb.device)
        output, hidden = self.rnn(emb, hidden) 
        output = self.drop(output) 
        return output

    def init_hidden(self, batch_size, device):
        if self.bidirectional:
            return torch.zeros(self.layers_num*2, batch_size, self.hidden_size, device=device)
        else:
            return torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device)
