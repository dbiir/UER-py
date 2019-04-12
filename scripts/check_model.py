# -*- encoding:utf -*-
import os
import sys
import torch
import argparse

bert_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(bert_dir)

from bert.model_builder import build_model
from bert.utils.vocab import Vocab
from bert.utils.config import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_model_path", type=str)
    parser.add_argument("--vocab_path", type=str)
    parser.add_argument("--config_path", type=str, default="./model.config")
    args = parser.parse_args()
    args = load_hyperparam(args)

    input_model = torch.load(args.input_model_path)
    prefix = "module."
    for k, v in input_model.items():
        if prefix in k:
            print("Multi-GPU version.")
            break
        else:
            print("Single-GPU version.")
            break

    print("Check model loading operation.")
    vocab = Vocab()
    vocab.load(args.vocab_path)
    test_model = build_model(args, len(vocab))
    try:
        test_model.load_state_dict(input_model, strict=True)
    except RuntimeError:
        print("Model loading test Failed. Please check if the input model.")
    else:
        print("Pass the check. Test Done.")
