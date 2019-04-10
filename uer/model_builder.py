# -*- encoding:utf-8 -*-
import torch
from uer.layers.embeddings import BertEmbedding
from uer.encoders.bert_encoder import BertEncoder
from uer.encoders.rnn_encoder import LstmEncoder, GruEncoder
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
#from uer.models.bert_model import BertModel
from uer.models.model import Model


def build_model(args, vocab_size):
    """
    Build universial encoder representations models.
    Only BERT is retained in this project.
    The combinations of different embedding, encoder, 
    and target layers yield pretrained models of different 
    properties. 
    We could select suitable one for downstream tasks.
    """

    # embedding = BertEmbedding(args, vocab_size)
    # encoder = BertEncoder(args)
    # target = BertTarget(args, vocab_size)
    # model = BertModel(args, embedding, encoder, target)

    embedding = BertEmbedding(args, vocab_size)
    encoder = globals()[args.encoder_type.capitalize() + "Encoder"](args)
    target = globals()[args.target.capitalize() + "Target"](args, vocab_size)
    model = Model(args, embedding, encoder, target)

    return model
