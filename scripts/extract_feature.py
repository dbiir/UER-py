# -*- encoding:utf-8 -*-
import sys
import os
import torch
import argparse
import torch.nn as nn

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils.config import load_hyperparam
from uer.utils.tokenizer import *
from uer.model_builder import build_model


class SequenceEncoder(torch.nn.Module):
    
    def __init__(self, model):
        super(SequenceEncoder, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        # Close dropout.
        self.eval()

    def forward(self, src, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        return output        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path of the input file.")
    parser.add_argument("--pretrained_model_path", type=str, required=True,
                        help="Path of the pretrained model.")
    parser.add_argument("--vocab_path", type=str, required=True,
                        help="Path of the vocabulary file.")
    parser.add_argument("--output_path", required=True,
                        help="Path of the output file.")
    parser.add_argument("--config_path", default="models/bert_base_config.json",
                        help="Path of the config file.")
    
    # Model options.
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--embedding", choices=["bert", "word"], default="bert",
                        help="Emebdding type.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                                   "cnn", "gatedcnn", "attn", \
                                                   "rcnn", "crnn", "gpt"], \
                                                   default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer." 
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )

    args = parser.parse_args()
    args = load_hyperparam(args)

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build and load modeli.
    # A pseudo target is added.
    args.target = "bert"
    model = build_model(args)
    pretrained_model = torch.load(args.pretrained_model_path)
    model.load_state_dict(pretrained_model, strict=False)

    seq_encoder = SequenceEncoder(model)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        seq_encoder = nn.DataParallel(seq_encoder)

    seq_encoder = seq_encoder.to(device)
    
    # Build tokenizer
    tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)

    dataset = []
    with open(args.input_path, mode="r", encoding="utf-8") as f:
        for line in f:
            tokens = [vocab.get(t) for t in tokenizer.tokenize(line)]
            if len(tokens) == 0:
                continue
            tokens = [CLS_ID] + tokens
            seg = [1] * len(tokens)

            if len(tokens) > args.seq_length:
                tokens = tokens[:args.seq_length]
                seg = seg[:args.seq_length]
            while len(tokens) < args.seq_length:
                tokens.append(PAD_ID)
                seg.append(PAD_ID)
            dataset.append((tokens, seg))
           
    src = torch.LongTensor([e[0] for e in dataset])
    seg = torch.LongTensor([e[1] for e in dataset])

    def batch_loader(batch_size, src, seg):
        instances_num = src.size(0)
        for i in range(instances_num // batch_size):
            src_batch = src[i*batch_size : (i+1)*batch_size]
            seg_batch = seg[i*batch_size : (i+1)*batch_size]
            yield src_batch, seg_batch
        if instances_num > instances_num // batch_size * batch_size:
            src_batch = src[instances_num//batch_size*batch_size:]
            seg_batch = seg[instances_num//batch_size*batch_size:]
            yield src_batch, seg_batch

    feature_vectors = []
    for i, (src_batch, seg_batch) in enumerate(batch_loader(args.batch_size, src, seg)):
        src_batch = src_batch.to(device)
        seg_batch = seg_batch.to(device)
        output = seq_encoder(src_batch, seg_batch)
        feature_vectors.append(output)

    feature_vectors = torch.cat(feature_vectors, 0)
    torch.save(feature_vectors,args.output_path)
    print("The number of sentences: {}".format(feature_vectors.size(0)))
