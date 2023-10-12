import math
import torch
import torch.nn as nn

from uer.utils.constants import *


class SinusoidalposEmbedding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.
    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, args, _):
        super(SinusoidalposEmbedding, self).__init__()

        if "speech" in args.embedding:
            self.max_seq_length = max(args.max_seq_length, args.max_audio_frames)
            self.arrange_sincos_cross = False
        else:
            self.max_seq_length = args.max_seq_length
            self.arrange_sincos_cross = True
        self.emb_size = args.emb_size
        half_dim = self.emb_size // 2   
        value = math.log(10000) / (half_dim - 1)
        half_exp = torch.exp(torch.arange(half_dim, dtype=torch.float) * -value)
        half_mat = torch.arange(self.max_seq_length, dtype=torch.float).unsqueeze(
            1
        ) * half_exp.unsqueeze(0)
        if not self.arrange_sincos_cross: #Same as the implementation of huggingface/transformers, tensor2tensor
            self.emb = torch.cat([torch.sin(half_mat), torch.cos(half_mat)], dim=1).view(
                self.max_seq_length, -1
            )
        else: #Implementation based on "Attention Is All You Need"
            self.emb = torch.zeros(self.max_seq_length, args.emb_size)
            self.emb[:, 0::2] = torch.sin(half_mat)
            self.emb[:, 1::2] = torch.cos(half_mat)
        if self.emb_size % 2 == 1:
            # zero pad
            self.emb = torch.cat([self.emb, torch.zeros(self.max_seq_length, 1)], dim=1)

        self.emb[args.tokenizer.vocab.get(PAD_TOKEN), :] = 0

    def forward(self, src, seg):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(batch_size, seq_len, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """
        if seg is not None:
            batch_size, seq_length = seg.size()
            device = seg.device
            no_pad_num = seg.sum(dim=-1)
        else:
            batch_size, seq_length = src.size()
            device = src.device
            no_pad_num = (src != 0).sum(dim=-1)
        
        emb =  torch.zeros(batch_size, seq_length, self.emb_size)
        for i in range(batch_size):
            emb[i, :no_pad_num[i], :] = self.emb[2: no_pad_num[i]+2]

        return emb.to(device)
