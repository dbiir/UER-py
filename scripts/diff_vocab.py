import sys
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--vocab_1", type=str)
    parser.add_argument("--vocab_2", type=str)
    args = parser.parse_args()

    vocab_set_1 = set()
    vocab_set_2 = set()

    with open(args.vocab_1, mode='r', encoding='utf-8') as f:
        for line in f:
            try:
                w = line.strip().split()[0]
                vocab_set_1.add(w)
            except:
                pass
    
    with open(args.vocab_2, mode='r', encoding='utf-8') as f:
        for line in f:
            try:
                w = line.strip().split()[0]
                vocab_set_2.add(w)
            except:
                pass

    print("vocab_1: " + args.vocab_1 + ", size: " + str(len(vocab_set_1)) )
    print("vocab_2: " + args.vocab_2 + ", size: " + str(len(vocab_set_2)) )

    print("vocab_1 - " + "vocab_2 = " + str(len(vocab_set_1 - vocab_set_2)))
    print("vocab_2 - " + "vocab_1 = " + str(len(vocab_set_2 - vocab_set_1)))
    print("vocab_1 & " + "vocab_2 = " + str(len(vocab_set_1 & vocab_set_2)))
