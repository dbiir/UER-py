import torch
import torch.nn as nn


class GatedcnnEncoder(nn.Module):
    def __init__(self, args):
        super(GatedcnnEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.kernel_size = args.kernel_size
        self.block_size = args.block_size
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size

        self.conv_1 = nn.Conv2d(1, args.hidden_size, (args.kernel_size, args.emb_size))
        self.gate_1 = nn.Conv2d(1, args.hidden_size, (args.kernel_size, args.emb_size))

        self.conv_b1 = nn.Parameter(torch.randn(1, args.hidden_size, 1, 1))
        self.gate_b1 = nn.Parameter(torch.randn(1, args.hidden_size, 1, 1))

        self.conv = nn.ModuleList(
            [
                nn.Conv2d(args.hidden_size, args.hidden_size, (args.kernel_size, 1))
                for _ in range(args.layers_num - 1)
            ]
        )
        self.gate = nn.ModuleList(
            [
                nn.Conv2d(args.hidden_size, args.hidden_size, (args.kernel_size, 1))
                for _ in range(args.layers_num - 1)
            ]
        )
        
        self.conv_b = nn.ParameterList(
            nn.Parameter(torch.randn(1, args.hidden_size, 1, 1))
            for _ in range(args.layers_num - 1)
        )
        self.gate_b = nn.ParameterList(
            nn.Parameter(torch.randn(1, args.hidden_size, 1, 1))
            for _ in range(args.layers_num - 1)
        )

    def forward(self, emb, seg):
        batch_size, seq_length, _ = emb.size()

        padding = torch.zeros([batch_size, self.kernel_size-1, self.emb_size]).to(emb.device)
        emb = torch.cat([padding, emb], dim=1).unsqueeze(1)  # batch_size, 1, seq_length+width-1, emb_size

        hidden = self.conv_1(emb)
        hidden += self.conv_b1.repeat(1, 1, seq_length, 1)
        gate = self.gate_1(emb)
        gate += self.gate_b1.repeat(1, 1, seq_length, 1)
        hidden = hidden * torch.sigmoid(gate)

        res_input = hidden

        padding = torch.zeros([batch_size, self.hidden_size, self.kernel_size-1, 1]).to(emb.device)
        hidden = torch.cat([padding, hidden], dim=2)

        for i, (conv_i, gate_i) in enumerate(zip(self.conv, self.gate)):
            hidden, gate = conv_i(hidden), gate_i(hidden)
            hidden += self.conv_b[i].repeat(1, 1, seq_length, 1)
            gate += self.gate_b[i].repeat(1, 1, seq_length, 1)
            hidden = hidden * torch.sigmoid(gate)
            if (i + 1) % self.block_size == 0:
                hidden = hidden + res_input
                res_input = hidden
            hidden = torch.cat([padding, hidden], dim=2)

        hidden = hidden[:, :, self.kernel_size - 1:, :]
        output = hidden.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)

        return output
