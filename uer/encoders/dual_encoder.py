from argparse import Namespace
import torch.nn as nn
import copy


class DualEncoder(nn.Module):
    """
    Dual Encoder which enables siamese models like SBER and CLIP.
    """
    def __init__(self, args):
        super(DualEncoder, self).__init__()
        from uer.encoders import str2encoder

        stream_0_args = copy.deepcopy(vars(args))
        stream_0_args.update(args.stream_0)
        stream_0_args = Namespace(**stream_0_args)
        self.encoder_0 = str2encoder[stream_0_args.encoder](stream_0_args)

        stream_1_args = copy.deepcopy(vars(args))
        stream_1_args.update(args.stream_1)
        stream_1_args = Namespace(**stream_1_args)
        self.encoder_1 = str2encoder[stream_1_args.encoder](stream_1_args)

        if args.tie_weights:
            self.encoder_1 = self.encoder_0

    def forward(self, emb, seg):
        """
        Args:
            emb: ([batch_size x seq_length x emb_size], [batch_size x seq_length x emb_size])
            seg: ([batch_size x seq_length], [batch_size x seq_length])
        Returns:
            features_0: [batch_size x seq_length x hidden_size]
            features_1: [batch_size x seq_length x hidden_size]
        """
        features_0 = self.get_encode_0(emb[0], seg[0])
        features_1 = self.get_encode_1(emb[1], seg[1])

        return features_0, features_1

    def get_encode_0(self, emb, seg):
        features = self.encoder_0(emb, seg)
        return features

    def get_encode_1(self, emb, seg):
        features = self.encoder_1(emb, seg)
        return features
