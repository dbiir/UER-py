# -*- encoding:utf-8 -*-
import sys
import os
import torch
import codecs
import argparse
import numpy as np
import torch.nn as nn

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils.config import load_hyperparam
from uer.utils.tokenizer import *
from uer.model_builder import build_model


class SequenceEncoder(torch.nn.Module):
    
    def __init__(self, bert_model):
        super(SequenceEncoder, self).__init__()
        self.embedding = bert_model.embedding
        self.encoder = bert_model.encoder
        # Close dropout.
        self.eval()

    def forward(self, input_ids, seg_ids):
        emb = self.embedding(input_ids, seg_ids)
        output = self.encoder(emb, seg_ids)
        return output        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path of the input file.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path of the pretrained model.")
    parser.add_argument("--vocab_path", type=str, required=True,
                        help="Path of the vocabulary file.")
    parser.add_argument("--output_path", required=True,
                        help="Path of the input file which is in npy format.")
    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")
    
    # Model options
    parser.add_argument("--seq_length", type=int, default=100, help="Sequence length.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--config_path", default="models/google_config.json", help="Model config file.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                                   "cnn", "gatedcnn", "attn", \
                                                   "rcnn", "crnn", "gpt"], \
                                                   default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")
    parser.add_argument("--target", choices=["bert", "lm", "cls", "mlm", "nsp", "s2s"], default="bert",
                        help="The training target of the pretraining model.")

    # Tokenizer options.

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["char", "word", "space", "mixed"], default="char",
                        help="Specify the tokenizer." 
                             "Original Google BERT uses char tokenizer on Chinese corpus."
                             "Word tokenizer supports online word segmentation based on jieba segmentor."
                             "Space tokenizer segments sentences into words according to space."
                             "Mixed tokenizer segments sentences into words according to space."
                             "If words are not in the vocabulary, the tokenizer splits words into characters."
                             )

    args = parser.parse_args()
    args = load_hyperparam(args)

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build and load model.
    bert_model = build_model(args)
    pretrained_model = torch.load(args.model_path)
    bert_model.load_state_dict(pretrained_model, strict=True)
    seq_encoder = SequenceEncoder(bert_model)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        seq_encoder = nn.DataParallel(seq_encoder)

    seq_encoder = seq_encoder.to(device)
    
    # Build tokenizer
    if args.tokenizer == "mixed":
        tokenizer = MixedTokenizer(vocab)
    else:
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
           
    input_ids = torch.LongTensor([e[0] for e in dataset])
    seg_ids = torch.LongTensor([e[1] for e in dataset])

    def batch_loader(batch_size, input_ids, seg_ids):
        instances_num = input_ids.size(0)
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i*batch_size : (i+1)*batch_size]
            seg_ids_batch = seg_ids[i*batch_size : (i+1)*batch_size]
            yield input_ids_batch, seg_ids_batch
        # Last data.
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num//batch_size*batch_size:]
            seg_ids_batch = seg_ids[instances_num//batch_size*batch_size:]
            yield input_ids_batch, seg_ids_batch

    sentence_vectors = []
    for i, (input_ids_batch, seg_ids_batch) in enumerate(batch_loader(args.batch_size, input_ids, seg_ids)):
        input_ids_batch = input_ids_batch.to(device)
        seg_ids_batch = seg_ids_batch.to(device)
        output = seq_encoder(input_ids_batch, seg_ids_batch)
        if args.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif args.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif args.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = output.cpu().data.numpy()
        sentence_vectors.append(output)

    sentence_vectors = np.concatenate(sentence_vectors, axis=0)
    np.save(args.output_path, sentence_vectors)
    print("The number of sentences: {}".format(sentence_vectors.shape[0]))