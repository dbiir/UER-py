"""
This script provides an example to wrap UER-py for text-to-text inference.
"""
import sys
import os
import random
import argparse
import torch

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.utils.vocab import Vocab
from uer.model_loader import load_model
from uer.opts import infer_opts, tokenizer_opts
from finetune.run_text2text import Text2text
from inference.run_classifier_infer import batch_loader


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.rstrip("\r\n").split("\t")):
                    columns[column_name] = i
                continue
            line = line.rstrip("\r\n").split("\t")

            if "text_b" in columns:
                text = line[columns["text_a"]] + SEP_TOKEN + line[columns["text_b"]]
            else:
                text = line[columns["text_a"]]

            src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text) + [SEP_TOKEN])
            seg = [1] * len(src)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
            while len(src) < args.seq_length:
                src.append(PAD_ID)
                seg.append(0)

            dataset.append((src, seg))

    return dataset


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    tokenizer_opts(parser)

    parser.add_argument("--tgt_seq_length", type=int, default=32,
                        help="Output sequence length.")
    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model.
    model = Text2text(args)
    model = load_model(model, args.load_model_path)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)


    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    dataset = read_dataset(args, args.test_path)

    src = torch.LongTensor([sample[0] for sample in dataset])
    seg = torch.LongTensor([sample[1] for sample in dataset])

    batch_size = args.batch_size
    instances_num = src.size()[0]

    print("The number of prediction instances: ", instances_num)

    model.eval()
    with open(args.prediction_path, mode="w", encoding="utf-8") as f:
        f.write("label")
        f.write("\n")
        for i, (src_batch, seg_batch) in enumerate(batch_loader(batch_size, src, seg)):
            src_batch = src_batch.to(args.device)
            seg_batch = seg_batch.to(args.device)
            tgt_in_batch = torch.zeros(src_batch.size()[0], 1, dtype = torch.long, device = args.device)
            tgt_seg_batch = torch.ones(tgt_in_batch.size()[0], 1, dtype = torch.long, device = args.device)
            current_batch_size = tgt_in_batch.size()[0]
            for j in range(current_batch_size):
                tgt_in_batch[j][-1] = args.tokenizer.vocab.get(CLS_TOKEN)

            with torch.no_grad():
                memory_bank = model(src_batch, None, seg_batch, tgt_seg_batch, only_use_encoder=True)
            for _ in range(args.tgt_seq_length):
                with torch.no_grad():
                    outputs = model(src_batch, (tgt_in_batch, None, src_batch), None, tgt_seg_batch, memory_bank=memory_bank)
                next_token_logits = outputs[:, -1]
                next_tokens = torch.argmax(next_token_logits, dim=1).unsqueeze(1)
                tgt_in_batch = torch.cat([tgt_in_batch, next_tokens], dim=1)
                tgt_seg_batch = torch.ones(tgt_in_batch.size()[0], tgt_in_batch.size()[1], dtype=torch.long, device=args.device)
            for j in range(len(outputs)):
                f.write("".join([args.tokenizer.inv_vocab[token_id.item()] for token_id in tgt_in_batch[j][1:]])
                        .split(SEP_TOKEN)[0])
                f.write("\n")


if __name__ == "__main__":
    main()
