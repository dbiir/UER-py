# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn


class RcnnEncoder(nn.Module):
    def __init__(self, args):
        super(RcnnEncoder, self).__init__()

        self.emb_size = args.emb_size
        self.hidden_size= args.hidden_size
        self.kernel_size = args.kernel_size
        self.layers_num = args.layers_num

        self.rnn = nn.LSTM(input_size=args.emb_size,
                           hidden_size=args.hidden_size,
                           num_layers=args.layers_num,
                           dropout=args.dropout,
                           batch_first=True)

        self.drop = nn.Dropout(args.dropout)

        self.conv_1 = nn.Conv2d(1, args.hidden_size, (args.kernel_size, args.emb_size))
        self.conv = nn.ModuleList([nn.Conv2d(args.hidden_size, args.hidden_size, (args.kernel_size, 1)) \
            for _ in range(args.layers_num-1)])

    def forward(self, emb, seg):
        batch_size, seq_len, _ = emb.size()

        hidden = self.init_hidden(batch_size, emb.device)
        output, hidden = self.rnn(emb, hidden) 
        output = self.drop(output)

        
        padding = torch.zeros([batch_size, self.kernel_size-1, self.emb_size]).to(emb.device)
        hidden = torch.cat([padding, output], dim=1).unsqueeze(1) # batch_size, 1, seq_length+width-1, emb_size
        hidden = self.conv_1(hidden)
        padding =  torch.zeros([batch_size, self.hidden_size, self.kernel_size-1, 1]).to(emb.device)
        hidden = torch.cat([padding, hidden], dim=2)
        for i, conv_i in enumerate(self.conv):
            hidden = conv_i(hidden)
            hidden = torch.cat([padding, hidden], dim=2)
        hidden = hidden[:,:,self.kernel_size-1:,:]
        output = hidden.transpose(1,2).contiguous().view(batch_size, seq_len, self.hidden_size)

        return output

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device),
                torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device))


class CrnnEncoder(nn.Module):
    def __init__(self, args):
        super(CrnnEncoder, self).__init__()

        self.emb_size = args.emb_size
        self.hidden_size= args.hidden_size
        self.kernel_size = args.kernel_size
        self.layers_num = args.layers_num

        self.conv_1 = nn.Conv2d(1, args.hidden_size, (args.kernel_size, args.emb_size))
        self.conv = nn.ModuleList([nn.Conv2d(args.hidden_size, args.hidden_size, (args.kernel_size, 1)) \
            for _ in range(args.layers_num-1)])


        self.rnn = nn.LSTM(input_size=args.emb_size,
                           hidden_size=args.hidden_size,
                           num_layers=args.layers_num,
                           dropout=args.dropout,
                           batch_first=True)
        self.drop = nn.Dropout(args.dropout)

    def forward(self, emb, seg):
        batch_size, seq_len, _ = emb.size()
        padding = torch.zeros([batch_size, self.kernel_size-1, self.emb_size]).to(emb.device)
        emb = torch.cat([padding, emb], dim=1).unsqueeze(1) # batch_size, 1, seq_length+width-1, emb_size
        hidden = self.conv_1(emb)
        padding =  torch.zeros([batch_size, self.hidden_size, self.kernel_size-1, 1]).to(emb.device)
        hidden = torch.cat([padding, hidden], dim=2)
        for i, conv_i in enumerate(self.conv):
            hidden = conv_i(hidden)
            hidden = torch.cat([padding, hidden], dim=2)
        hidden = hidden[:,:,self.kernel_size-1:,:]
        output = hidden.transpose(1,2).contiguous().view(batch_size, seq_len, self.hidden_size)

        hidden = self.init_hidden(batch_size, emb.device)
        output, hidden = self.rnn(output, hidden) 
        output = self.drop(output)

        return output

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device),
                torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device))

