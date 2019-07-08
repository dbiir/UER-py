# -*- encoding:utf-8 -*-
import torch
from uer.layers.embeddings import BertEmbedding
from uer.encoders.bert_encoder import BertEncoder
from uer.encoders.rnn_encoder import LstmEncoder, GruEncoder
from uer.encoders.birnn_encoder import BilstmEncoder
from uer.encoders.cnn_encoder import CnnEncoder, GatedcnnEncoder
from uer.encoders.attn_encoder import AttnEncoder
from uer.encoders.gpt_encoder import GptEncoder
from uer.encoders.mixed_encoder import RcnnEncoder, CrnnEncoder
from uer.targets.bert_target import BertTarget
from uer.targets.lm_target import LmTarget
from uer.targets.cls_target import ClsTarget
from uer.targets.mlm_target import MlmTarget
from uer.targets.nsp_target import NspTarget
from uer.targets.s2s_target import S2sTarget
from uer.targets.bilm_target import BilmTarget
from uer.subencoders.avg_subencoder import AvgSubencoder
from uer.subencoders.rnn_subencoder import LstmSubencoder
from uer.subencoders.cnn_subencoder import CnnSubencoder
from uer.models.model import Model


def build_model(args):
    """
    Build universial encoder representations models.
    The combinations of different embedding, encoder, 
    and target layers yield pretrained models of different 
    properties. 
    We could select suitable one for downstream tasks.
    """

    if args.subword_type != "none":
        subencoder = globals()[args.subencoder.capitalize() + "Subencoder"](args, len(args.sub_vocab))
    else:
        subencoder = None

    embedding = BertEmbedding(args, len(args.vocab))
    encoder = globals()[args.encoder.capitalize() + "Encoder"](args)
    target = globals()[args.target.capitalize() + "Target"](args, len(args.vocab))
    model = Model(args, embedding, encoder, target, subencoder)

    return model
