import torch
from uer.layers import *
from uer.encoders import *
from uer.targets import *
from uer.models.model import Model


def build_model(args):
    """
    Build universial encoder representations models.
    The combinations of different embedding, encoder, 
    and target layers yield pretrained models of different 
    properties. 
    We could select suitable one for downstream tasks.
    """

    embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
    encoder = str2encoder[args.encoder](args)
    if args.target == "seq2seq":
        target = str2target[args.target](args, len(args.tgt_tokenizer.vocab))
    else:
        target = str2target[args.target](args, len(args.tokenizer.vocab))
    model = Model(args, embedding, encoder, target)

    return model
