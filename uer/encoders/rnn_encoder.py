import torch
import torch.nn as nn
from uer.utils.misc import *


class RnnEncoder(nn.Module):
    def __init__(self, args):
        super(RnnEncoder, self).__init__()

        self.bidirectional = args.bidirectional
        if self.bidirectional:
            assert args.hidden_size % 2 == 0
            self.hidden_size = args.hidden_size // 2
        else:
            self.hidden_size = args.hidden_size
        self.layers_num = args.layers_num

        self.rnn = nn.RNN(input_size=args.emb_size,
                          hidden_size=self.hidden_size,
                          num_layers=args.layers_num,
                          dropout=args.dropout,
                          batch_first=True,
                          bidirectional=self.bidirectional)

        self.drop = nn.Dropout(args.dropout)

    def forward(self, emb, _):
        hidden = self.init_hidden(emb.size(0), emb.device)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        return output

    def init_hidden(self, batch_size, device):
        if self.bidirectional:
            return torch.zeros(self.layers_num*2, batch_size, self.hidden_size, device=device)
        else:
            return torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device)


class LstmEncoder(RnnEncoder):
    def __init__(self, args):
        super(LstmEncoder, self).__init__(args)

        self.rnn = nn.LSTM(input_size=args.emb_size,
                           hidden_size=self.hidden_size,
                           num_layers=args.layers_num,
                           dropout=args.dropout,
                           batch_first=True,
                           bidirectional=self.bidirectional)

    def init_hidden(self, batch_size, device):
        if self.bidirectional:
            return (torch.zeros(self.layers_num*2, batch_size, self.hidden_size, device=device),
                    torch.zeros(self.layers_num*2, batch_size, self.hidden_size, device=device))
        else:
            return (torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device),
                    torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device))


class GruEncoder(RnnEncoder):
    def __init__(self, args):
        super(GruEncoder, self).__init__(args)

        self.rnn = nn.GRU(input_size=args.emb_size,
                          hidden_size=self.hidden_size,
                          num_layers=args.layers_num,
                          dropout=args.dropout,
                          batch_first=True,
                          bidirectional=self.bidirectional)


class BirnnEncoder(nn.Module):
    def __init__(self, args):
        super(BirnnEncoder, self).__init__()

        assert args.hidden_size % 2 == 0
        self.hidden_size = args.hidden_size // 2
        self.layers_num = args.layers_num

        self.rnn_forward = nn.RNN(input_size=args.emb_size,
                           hidden_size=self.hidden_size,
                           num_layers=args.layers_num,
                           dropout=args.dropout,
                           batch_first=True)

        self.rnn_backward = nn.RNN(input_size=args.emb_size,
                           hidden_size=self.hidden_size,
                           num_layers=args.layers_num,
                           dropout=args.dropout,
                           batch_first=True)

        self.drop = nn.Dropout(args.dropout)
    
    def forward(self, emb, _):
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
        return torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device)


class BilstmEncoder(BirnnEncoder):
    def __init__(self, args):
        super(BilstmEncoder, self).__init__(args)

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

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device),
                torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device))


class BigruEncoder(BirnnEncoder):
    def __init__(self, args):
        super(BigruEncoder, self).__init__(args)

        self.rnn_forward = nn.GRU(input_size=args.emb_size,
                           hidden_size=self.hidden_size,
                           num_layers=args.layers_num,
                           dropout=args.dropout,
                           batch_first=True)

        self.rnn_backward = nn.GRU(input_size=args.emb_size,
                           hidden_size=self.hidden_size,
                           num_layers=args.layers_num,
                           dropout=args.dropout,
                           batch_first=True)
