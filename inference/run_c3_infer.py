"""
  This script provides an example to wrap UER-py for C3 (a multiple choice dataset) inference.
"""
import sys
import os
import argparse
import torch
import torch.nn as nn

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.model_loader import load_model
from uer.opts import infer_opts, tokenizer_opts
from finetune.run_classifier import batch_loader
from finetune.run_c3 import MultipleChoice, read_dataset


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    parser.add_argument("--max_choices_num", default=4, type=int,
                        help="The maximum number of cadicate answer, shorter than this will be padded.")

    tokenizer_opts(parser)

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model and load parameters.
    model = MultipleChoice(args)
    model = load_model(model, args.load_model_path)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    dataset = read_dataset(args, args.test_path)

    src = torch.LongTensor([example[0] for example in dataset])
    tgt = torch.LongTensor([example[1] for example in dataset])
    seg = torch.LongTensor([example[2] for example in dataset])

    batch_size = args.batch_size
    instances_num = src.size()[0]

    print("The number of prediction instances: ", instances_num)

    model.eval()

    with open(args.test_path) as f:
        data = json.load(f)

    question_ids = []
    for i in range(len(data)):
        questions = data[i][1]
        for question in questions:
            question_ids.append(question["id"])

    index = 0
    with open(args.prediction_path, "w") as f:
        for i, (src_batch, _, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):

            src_batch = src_batch.to(device)
            seg_batch = seg_batch.to(device)

            with torch.no_grad():
                _, logits = model(src_batch, None, seg_batch)

                pred = torch.argmax(logits, dim=1)
                pred = pred.cpu().numpy().tolist()
                for j in range(len(pred)):
                    output = {}
                    output["id"] = question_ids[index]
                    index += 1
                    output["label"] = int(pred[j])
                    f.write(json.dumps(output))
                    f.write("\n")


if __name__ == "__main__":
    main()
