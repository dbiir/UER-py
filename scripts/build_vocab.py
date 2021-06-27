"""
Build vocabulary with given tokenizer
"""
import sys
import os
import argparse

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.utils import *
from uer.utils.vocab import Vocab
from uer.opts import tokenizer_opts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--workers_num", type=int, default=1, help="The number of processes to build vocabulary.")
    parser.add_argument("--min_count", type=int, default=1, help="The minimum count of words retained in the vocabulary.")

    tokenizer_opts(parser)

    args = parser.parse_args()

    vocab_path = args.vocab_path
    args.vocab_path, args.spm_model_path = "./models/reserved_vocab.txt", None

    # Build tokenizer.
    tokenizer = str2tokenizer[args.tokenizer](args)

    # Build and save vocabulary.
    vocab = Vocab()
    vocab.build(args.corpus_path, tokenizer, args.workers_num, args.min_count)
    vocab.save(vocab_path)
