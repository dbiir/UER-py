from uer.embeddings import *
from uer.encoders import *
from uer.decoders import *
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

    embedding = Embedding(args)
    for embedding_name in args.embedding:
        tmp_emb = str2embedding[embedding_name](args, len(args.tokenizer.vocab))
        embedding.update(tmp_emb, embedding_name)

    encoder = str2encoder[args.encoder](args)

    if args.decoder is not None:
        if args.data_processor == "mt":
            tgt_vocab_size = len(args.tgt_tokenizer.vocab)
        else:
            tgt_vocab_size = len(args.tokenizer.vocab)

        tgt_embedding = Embedding(args)
        for embedding_name in args.tgt_embedding:
            tmp_emb = str2embedding[embedding_name](args, tgt_vocab_size)
            tgt_embedding.update(tmp_emb, embedding_name)

        decoder = str2decoder[args.decoder](args)
    else:
        tgt_embedding = None
        decoder = None

    target = Target()
    for target_name in args.target:
        if args.data_processor == "mt":
            tmp_target = str2target[target_name](args, len(args.tgt_tokenizer.vocab))
        else:
            tmp_target = str2target[target_name](args, len(args.tokenizer.vocab))
        target.update(tmp_target, target_name)
    model = Model(args, embedding, encoder, tgt_embedding, decoder, target)

    return model
