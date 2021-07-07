"""
  This script provides an example to wrap UER-py for NER inference.
"""
import sys
import os
import argparse
import json
import torch
import torch.nn as nn

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.utils.config import load_hyperparam
from uer.utils.constants import *
from uer.utils.tokenizers import *
from uer.model_loader import load_model
from uer.opts import infer_opts
from finetune.run_ner import NerTagger


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().split('\t')
            text_a = line[columns["text_a"]]
            src = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_a))
            seg = [1] * len(src)

            if len(src) > args.seq_length:
                src = src[:args.seq_length]
                seg = seg[:args.seq_length]
            PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
            while len(src) < args.seq_length:
                src.append(PAD_ID)
                seg.append(0)
            dataset.append([src, seg])

    return dataset


def batch_loader(batch_size, src, seg):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        yield src_batch, seg_batch
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        yield src_batch, seg_batch


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    parser.add_argument("--label2id_path", type=str, required=True,
                        help="Path of the label2id file.")
    parser.add_argument("--crf_target", action="store_true",
                        help="Use CRF loss as the target function or not, default False.")
    
    args = parser.parse_args()

    # Load the hyperparameters of the config file.
    args = load_hyperparam(args)

    with open(args.label2id_path, mode="r", encoding="utf-8") as f:
        l2i = json.load(f)
        print("Labels: ", l2i)
        l2i["[PAD]"] = len(l2i)

    i2l = {}
    for key, value in l2i.items():
        i2l[value] = key

    args.l2i = l2i

    args.labels_num = len(l2i)

    # Load tokenizer.
    args.tokenizer = SpaceTokenizer(args)

    # Build sequence labeling model.
    model = NerTagger(args)
    model = load_model(model, args.load_model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    
    instances = read_dataset(args, args.test_path)

    src = torch.LongTensor([ins[0] for ins in instances])
    seg = torch.LongTensor([ins[1] for ins in instances])

    instances_num = src.size(0)
    batch_size = args.batch_size

    print("The number of prediction instances: ", instances_num)

    model.eval()

    with open(args.prediction_path, mode="w", encoding="utf-8") as f:
        f.write("pred_label" + "\n")
        for i, (src_batch, seg_batch) in enumerate(batch_loader(batch_size, src, seg)):
            src_batch = src_batch.to(device)
            seg_batch = seg_batch.to(device)
            with torch.no_grad():
                _, pred = model(src_batch, None, seg_batch)

            # Storing sequence length of instances in a batch.
            seq_length_batch = []
            for seg in seg_batch.cpu().numpy().tolist():
                for j in range(len(seg) - 1, -1, -1):
                    if seg[j] != 0:
                        break
                seq_length_batch.append(j+1)
            pred = pred.cpu().numpy().tolist()
            for j in range(0, len(pred), args.seq_length):
                for label_id in pred[j: j + seq_length_batch[j // args.seq_length]]:
                    f.write(i2l[label_id] + " ")
                f.write("\n")
        

if __name__ == "__main__":
    main()
