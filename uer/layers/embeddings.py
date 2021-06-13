import torch
import math
import torch.nn as nn
import collections
from uer.layers.layer_norm import LayerNorm

def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)

class PatchPosEmbedding(nn.Module):
    """
    Image to Patch Embedding for Vision Transformer.
    """

    def __init__(self, args, vocab_size=None):
        super(PatchPosEmbedding, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args.emb_size))
        self.image_height = args.image_height
        self.image_width = args.image_width
        patch_size = to_2tuple(args.patch_size)
        num_channels = args.num_channels
        num_patches = (self.image_width // patch_size[1]) * (self.image_height // patch_size[0])

        self.projection = nn.Conv2d(num_channels, args.emb_size, kernel_size=patch_size, stride=patch_size)
        self.position_embedding = nn.Embedding(num_patches + 1, args.emb_size)
        self.dropout = nn.Dropout(args.dropout)
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, seg = None):
        batch_size, num_channels, height, width = src.shape
        if height != self.image_height or width != self.image_width:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_height}*{self.image_width})."
            )
        patch_emb = self.projection(src).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        patch_emb = torch.cat((cls_tokens, patch_emb), dim=1)
        pos_emb = self.position_embedding(
            torch.arange(0, patch_emb.size(1), device=patch_emb.device, dtype=torch.long)
                .unsqueeze(0)
                .repeat(patch_emb.size(0), 1)
        )
        emb = patch_emb + pos_emb
        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb


class PatchEmbedding(nn.Module):
    """
    Image to Patch Embedding for Vision Transformer.
    """

    def __init__(self, args, vocab_size=None):
        super(PatchEmbedding, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args.emb_size))
        patch_size = to_2tuple(args.patch_size)
        self.num_channels = args.num_channels

        self.projection = nn.Conv2d(self.num_channels, args.emb_size, kernel_size=patch_size, stride=patch_size)
        self.dropout = nn.Dropout(args.dropout)
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, seg = None):
        batch_size, num_channels, height, width = src.shape
        if num_channels != self.num_channels:
            raise ValueError(
                f"Input channel size ({num_channels}) doesn't match model ({self.num_channels})."
            )
        patch_emb = self.projection(src).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        patch_emb = torch.cat((cls_tokens, patch_emb), dim=1)

        if not self.remove_embedding_layernorm:
            patch_emb = self.layer_norm(patch_emb)
        patch_emb = self.dropout(patch_emb)
        return patch_emb


class ViLEmbedding(nn.Module):
    """
    """

    def __init__(self, args, vocab_size):
        super(ViLEmbedding, self).__init__()
        self.language_embedding = WordPosEmbedding(args, vocab_size)
        self.vision_embedding = PatchPosEmbedding(args)
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.emb_size)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, src, seg):
        l_emb = self.language_embedding(src[0], seg)
        v_emb = self.vision_embedding(src[1], seg)
        emb = torch.cat([l_emb,v_emb], dim=1)
        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb


class ClipEmbedding(nn.Module):
    """
    """

    def __init__(self, args, vocab_size):
        super(ClipEmbedding, self).__init__()
        self.language_embedding = WordPosEmbedding(args, vocab_size)
        self.vision_embedding = PatchPosEmbedding(args)
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.emb_size)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, src, seg):
        l_emb = self.language_embedding(src[0], seg)
        v_emb = self.vision_embedding(src[1])

        if not self.remove_embedding_layernorm:
            l_emb = self.layer_norm(l_emb)
            v_emb = self.layer_norm(v_emb)
        l_emb = self.dropout(l_emb)
        v_emb = self.dropout(v_emb)
        return (l_emb, v_emb)


class WordEmbedding(nn.Module):
    """
    """

    def __init__(self, args, vocab_size):
        super(WordEmbedding, self).__init__()
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        self.dropout = nn.Dropout(args.dropout)
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, _):
        emb = self.word_embedding(src)
        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb


class WordPosEmbedding(nn.Module):
    """
    GPT embedding consists of two parts:
    word embedding and position embedding.
    """

    def __init__(self, args, vocab_size):
        super(WordPosEmbedding, self).__init__()
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        self.dropout = nn.Dropout(args.dropout)
        self.max_seq_length = args.max_seq_length
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.position_embedding = nn.Embedding(self.max_seq_length, args.emb_size)
        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, _):
        word_emb = self.word_embedding(src)
        pos_emb = self.position_embedding(
            torch.arange(0, word_emb.size(1), device=word_emb.device, dtype=torch.long)
            .unsqueeze(0)
            .repeat(word_emb.size(0), 1)
        )

        emb = word_emb + pos_emb
        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb


class WordPosSegEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size):
        super(WordPosSegEmbedding, self).__init__()
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        self.dropout = nn.Dropout(args.dropout)
        self.max_seq_length = args.max_seq_length
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.position_embedding = nn.Embedding(self.max_seq_length, args.emb_size)
        self.segment_embedding = nn.Embedding(3, args.emb_size)
        if not self.remove_embedding_layernorm:
            self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, seg):
        word_emb = self.word_embedding(src)
        pos_emb = self.position_embedding(
            torch.arange(0, word_emb.size(1), device=word_emb.device, dtype=torch.long)
            .unsqueeze(0)
            .repeat(word_emb.size(0), 1)
        )
        seg_emb = self.segment_embedding(seg)

        emb = word_emb + pos_emb + seg_emb
        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb


class WordSinusoidalposEmbedding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.
    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, args, vocab_size):
        super(WordSinusoidalposEmbedding, self).__init__()
        if args.emb_size % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(args.emb_size))
        self.max_seq_length = args.max_seq_length
        pe = torch.zeros(self.max_seq_length, args.emb_size)
        position = torch.arange(0, self.max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            (
                torch.arange(0, args.emb_size, 2, dtype=torch.float)
                *- (math.log(10000.0) / args.emb_size)
            )
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, src, _):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(batch_size, seq_len, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """
        word_emb = self.word_embedding(src)
        emb = word_emb * math.sqrt(word_emb.size(-1))
        emb = emb + self.pe[: emb.size(1)].transpose(0, 1)
        emb = self.dropout(emb)
        return emb
