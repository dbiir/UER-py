# -*- encoding:utf-8 -*-
import sys
import os
import torch
import codecs
import argparse
import numpy as np

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.utils.vocab import Vocab
from uer.utils.config import load_hyperparam
from uer.layers.embeddings import BertEmbedding
from uer.encoders.bert_encoder import BertEncoder
from uer.utils.tokenizer import *
from uer.utils.constants import *


class SequenceEncoder(torch.nn.Module):
    
    def __init__(self, args, vocab):
        super(SequenceEncoder, self).__init__()
        self.embedding = BertEmbedding(args, len(vocab))
        self.encoder = BertEncoder(args)

    def forward(self, src, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        return output    


def sentence_encoding(src, seg):
    src = torch.LongTensor(src)
    seg = torch.LongTensor(seg)
    output = seq_encoder(src.to(device), seg.to(device))
    output = output.cpu().data.numpy()
    return output


def sentence_to_id(line):
    src = [vocab.get(w) for w in line]
    src = [CLS_ID] + src
    seg = [1] * len(src)
    if len(src) > args.seq_length:
        src = src[:args.seq_length]
        seg = seg[:args.seq_length]
    while len(src) < args.seq_length:
        src.append(PAD_ID)
        seg.append(PAD_ID)
    return src, seg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Path options.
    parser.add_argument("--sent_path", help=".")
    parser.add_argument("--vocab_path", help=".")
    parser.add_argument("--cand_vocab_path", help=".")
    parser.add_argument("--pretrained_model_path", help=".")
    # Model options.
    parser.add_argument("--seq_length", type=int)
    parser.add_argument("--layers_num", type=int, default=4)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--config_path", help=".")
    parser.add_argument("--topn", type=int, default=20)
    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="space",
                        help="Specify the tokenizer." 
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )

    args = parser.parse_args()
    layers_num = args.layers_num 
    args = load_hyperparam(args)
    args.layers_num = layers_num

    vocab = Vocab()
    vocab.load(args.vocab_path)

    seq_encoder = SequenceEncoder(args, vocab)    
 
    pretrained_model = torch.load(args.pretrained_model_path)
    seq_encoder.load_state_dict(pretrained_model, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        seq_encoder = torch.nn.DataParallel(seq_encoder)

    seq_encoder = seq_encoder.to(device)
    seq_encoder.eval()

    # Build tokenizer.
    tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)

    f_sent = open(args.sent_path, mode="r", encoding="utf-8")

    cand_vocab = Vocab()
    cand_vocab.load(args.cand_vocab_path)
    print("length of candidate vocab: "+str(len(cand_vocab)))

    for line in f_sent:
        # Sentence and word are splitted by "\t"
        line = line.strip().split("\t")
        if len(line) != 2:
            continue
        target_word = line[-1]
        print("Original sentence: ")
        print(line[0])
        sent = tokenizer.tokenize(line[0])
        print("Target word" + ": " + target_word)

        src, seg = sentence_to_id(sent)

        target_word_id = vocab.get(target_word)
        if target_word_id == UNK_ID:
            print("The candidate word is UNK in vocab.")
            continue
        
        # Search the position of the word in the sentence.
        position = -1
        if target_word_id in src:
            position = src.index(target_word_id)

        if position < 1:
            print("The target word is not in the sentence.")
            continue

        index_of_word_in_line = position 
        output = sentence_encoding([src], [seg])
        output = output.reshape([args.seq_length, -1])
        target_embedding = output[position,:]

        batch_size = args.batch_size
        cand_word_batch = []
        cand_embeddings = []
        for word_id, word in enumerate(cand_vocab.i2w):
            cand_word_batch.append(vocab.w2i.get(word))
            if len(cand_word_batch) == batch_size or word_id == (len(cand_vocab.i2w)-1):
                seg_batch = [seg]*len(cand_word_batch)
                src_batch = torch.LongTensor([src]*len(cand_word_batch))
                src_batch[:,position] = torch.LongTensor(cand_word_batch)
                output = sentence_encoding(src_batch, seg_batch)
                output = np.reshape(output, (len(output), args.seq_length, -1))
                cand_embeddings.extend(output[:, position, :].tolist())
                cand_word_batch = []

        target_embedding = np.array(target_embedding).reshape(1,-1).astype("float")

        sims = torch.nn.functional.cosine_similarity(torch.tensor(np.array(target_embedding), dtype=torch.float),\
                                                     torch.FloatTensor(cand_embeddings))
           
        sorted_id = torch.argsort(sims, descending=True)

        for j in sorted_id[1: args.topn+1]:
            print(cand_vocab.i2w[j].strip()+ "\t"+str(sims[j].item()))
            
    f_sent.close()
