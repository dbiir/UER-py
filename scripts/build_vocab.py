# -*- encoding:utf-8 -*-
"""
Build vocabulary with given tokenizer
"""

import sys
import os
import argparse

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.utils.tokenizer import *
from uer.utils.vocab import Vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--workers_num", type=int, default=1, help="The number of processes to build vocabulary.")
    parser.add_argument("--min_count", type=int, default=1, help="The minimum count of words retained in the vocabulary.")
    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["char", "space"], default="space",
                        help="Specify the tokenizer." 
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )

    args = parser.parse_args()

    # Build tokenizer.
    tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)

    # Build and save vocabulary.
    vocab = Vocab()
    vocab.build(args.corpus, tokenizer, args.workers_num, args.min_count)
    vocab.save(args.vocab_path)
