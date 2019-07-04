# -*- encoding:utf-8 -*-
import sys
import os
import torch
import codecs
import argparse

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.utils.vocab import Vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Path options.
    parser.add_argument("--target_words_path", help=".")
    parser.add_argument("--vocab_path", help=".")
    parser.add_argument("--cand_vocab_path", help=".")
    parser.add_argument("--pretrained_model_path", help=".")

    # Output path.
    parser.add_argument("--topn", type=int, default=20)

    args = parser.parse_args()

    vocab = Vocab()
    vocab.load(args.vocab_path)

    pretrained_model = torch.load(args.pretrained_model_path)
    embedding = pretrained_model["embedding.word_embedding.weight"]

    cand_vocab = Vocab()
    cand_vocab.load(args.cand_vocab_path)
    cand_vocab_id = [vocab.get(w) for w in cand_vocab.i2w]
    cand_embedding = embedding[cand_vocab_id, :]

    f_word = open(args.target_words_path, mode="r", encoding="utf-8")

    for line in f_word:
        word = line.strip().split()[0]
        print("Target word: " + word)
        target_embedding = embedding[vocab.get(word), :]
        sims = torch.nn.functional.cosine_similarity(target_embedding.view(1, -1), cand_embedding)
        sorted_id = torch.argsort(sims, descending=True)
        for j in sorted_id[1: args.topn+1]:
            print(cand_vocab.i2w[j].strip()+ "\t"+str(sims[j].item()))    
        print()
