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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--delimiter", choices=["char", "space"], required=True,
                        help="Tokenizing the corpus in char-level or by the provided spaces.")
    parser.add_argument("--output_path", required=True,
                        help="The output path to save the vocabulary.")
    parser.add_argument("--workers_num", type=int, default=1,
                        help="The number of processes to build vocabulary.")
    parser.add_argument("--min_count", type=int, default=1,
                        help="The minimum count of words retained in the vocabulary.")
    
    args = parser.parse_args()

    # Build tokenizer only for char and space.
    args.vocab_path, args.spm_model_path = "./models/reserved_vocab.txt", None
    tokenizer = str2tokenizer[args.delimiter](args)

    # Build and save vocabulary using CharTokenizer or SpaceTokenizer.
    vocab = Vocab()
    vocab.build(args.corpus_path, tokenizer, args.workers_num, args.min_count)
    vocab.save(args.output_path)
