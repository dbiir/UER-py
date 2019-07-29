# -*- encoding:utf-8 -*-
#-*- coding : utf-8 -*-
# coding: utf-8
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
    parser.add_argument("--vocab_path", help=".")
    parser.add_argument("--pretrained_model_path", help=".")

    # Output path.
    parser.add_argument("--output_word_embedding_path", help=".")

    args = parser.parse_args()

    vocab = Vocab()
    vocab.load(args.vocab_path)

    pretrained_model = torch.load(args.pretrained_model_path)
    embedding = pretrained_model["embedding.word_embedding.weight"]

    f_out = open(args.output_word_embedding_path, mode="w", encoding="utf-8")

    head=str(list(embedding.size())[0])+" "+str(list(embedding.size())[1])+"\n"
    f_out.write(head)

    for i in range(len(vocab.i2w)):
        word = vocab.i2w[i]
        word_embedding = embedding[vocab.get(word), :]
        word_embedding = word_embedding.cpu().numpy().tolist()
        s = str(word)
        for i in range(len(word_embedding)):
            s = s + " " + str(word_embedding[i])
        s += "\n"
        f_out.write(s)
        # print(word)
