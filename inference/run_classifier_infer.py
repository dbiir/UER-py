"""
  This script provides an exmaple to wrap BERT-PyTorch for classification inference.
"""
import sys
import os
import torch
import json
import random
import argparse
import collections
import torch.nn as nn

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils.tokenizer import * 
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_loader import load_model
from run_classifier import Classifier


def batch_loader(batch_size, src, seg):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i*batch_size: (i+1)*batch_size, :]
        seg_batch = seg[i*batch_size: (i+1)*batch_size, :]
        yield src_batch, seg_batch
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num//batch_size*batch_size:, :]
        seg_batch = seg[instances_num//batch_size*batch_size:, :]
        yield src_batch, seg_batch


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                line = line.strip().split("\t")
                for i, column_name in enumerate(line):
                    columns[column_name] = i
                continue
            line = line.strip().split('\t')
            if "text_b" not in columns: # Sentence classification.
                text_a = line[columns["text_a"]]
                src = [args.vocab.get(t) for t in args.tokenizer.tokenize(text_a)]
                src = [CLS_ID] + src
                seg = [1] * len(src)
            else: # Sentence pair classification.
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                src_a = [args.vocab.get(t) for t in args.tokenizer.tokenize(text_a)]
                src_a = [CLS_ID] + src_a + [SEP_ID]
                src_b = [args.vocab.get(t) for t in args.tokenizer.tokenize(text_b)]
                src_b = src_b + [SEP_ID]
                src = src_a + src_b
                seg = [1] * len(src_a) + [2] * len(src_b)
            
            if len(src) > args.seq_length:
                src = src[:args.seq_length]
                seg = seg[:args.seq_length]
            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)
            dataset.append((src, seg))

    return dataset


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path of the classfier model.")
    parser.add_argument("--vocab_path", type=str, required=True,
                        help="Path of the vocabulary file.")
    parser.add_argument("--test_path", type=str,
                        help="Path of the testset.")
    parser.add_argument("--prediction_path", default=None, type=str,
                        help="Path of the prediction file.")
    parser.add_argument("--config_path", default="./models/bert_base_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--labels_num", type=int, required=True,
                        help="Number of prediction labels.")
    parser.add_argument("--embedding", choices=["bert", "word"], default="bert",
                        help="Emebdding type.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                              "cnn", "gatedcnn", "attn", "synt", \
                                              "rcnn", "crnn", "gpt", "bilstm"], \
                                              default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")
    parser.add_argument("--factorized_embedding_parameterization", action="store_true", help="Factorized embedding parameterization.")
    parser.add_argument("--parameter_sharing", action="store_true", help="Parameter sharing.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer." 
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )

    # Output options.
    parser.add_argument("--output_logits", action="store_true", help="Write logits to output file.")
    
    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build classification model and load parameters.
    args.soft_targets = False
    model = Classifier(args)
    model = load_model(model, args.load_model_path)
    
    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    # Build tokenizer.
    args.tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)
    
    dataset = read_dataset(args, args.test_path)

    src = torch.LongTensor([sample[0] for sample in dataset])
    seg = torch.LongTensor([sample[1] for sample in dataset])

    batch_size = args.batch_size
    instances_num = src.size()[0]

    print("The number of prediction instances: ", instances_num)

    model.eval()

    with open(args.prediction_path, mode="w", encoding="utf-8") as f:
        if args.output_logits:
            f.write("label" + "\t" + "logits"  + "\n")
        else:
            f.write("label" + "\n")
        for i, (src_batch, seg_batch) in enumerate(batch_loader(batch_size, src, seg)):
            src_batch = src_batch.to(device)
            seg_batch = seg_batch.to(device)
            with torch.no_grad():
                _, logits = model(src_batch, None, seg_batch)
            
            pred = torch.argmax(logits, dim=1)
            pred = pred.cpu().numpy().tolist()
            logits = logits.cpu().numpy().tolist()
            
            if args.output_logits:
                for j in range(len(pred)):
                    f.write(str(pred[j]) + "\t" + " ".join([str(v) for v in logits[j]]) + "\n")
            else:
                for j in range(len(pred)):
                    f.write(str(pred[j]) + "\n")
 

if __name__ == "__main__":
    main()
