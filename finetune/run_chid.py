"""
This script provides an example to wrap UER-py for ChID (a multiple choice dataset).
"""
import sys
import os
import argparse
import json
import random
import torch

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.utils.constants import *
from uer.utils.tokenizers import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.opts import finetune_opts
from finetune.run_c3 import MultipleChoice
from finetune.run_classifier import build_optimizer, load_or_initialize_parameters, train_model, batch_loader, evaluate


def tokenize_chid(text):
    output = []
    first_idiom = True
    while True:
        if first_idiom:
            idiom_index = text.find("#idiom")
            output.extend(text[:idiom_index])
            output.append(text[idiom_index : idiom_index + 13])
            pre_idiom_index = idiom_index
            first_idiom = False
        else:
            if text[idiom_index + 1 :].find("#idiom") == -1:
                output.extend(text[pre_idiom_index + 13 :])
                break
            else:
                idiom_index = idiom_index + 1 + text[idiom_index + 1 :].find("#idiom")
                output.extend(text[pre_idiom_index + 13 : idiom_index])
                output.append(text[idiom_index : idiom_index + 13])
                pre_idiom_index = idiom_index

    return output


def add_tokens_around(tokens, idiom_index, tokens_num):
    left_tokens_num = tokens_num // 2
    right_tokens_num = tokens_num - left_tokens_num

    if idiom_index >= left_tokens_num and (len(tokens) - 1 - idiom_index) >= right_tokens_num:
        left_tokens = tokens[idiom_index - left_tokens_num : idiom_index]
        right_tokens = tokens[idiom_index + 1 : idiom_index + 1 + right_tokens_num]
    elif idiom_index < left_tokens_num:
        left_tokens = tokens[:idiom_index]
        right_tokens = tokens[idiom_index + 1 : idiom_index + 1 + tokens_num - len(left_tokens)]
    elif (len(tokens) - 1 - idiom_index) < right_tokens_num:
        right_tokens = tokens[idiom_index + 1 :]
        left_tokens = tokens[idiom_index - (tokens_num - len(right_tokens)) : idiom_index]

    return left_tokens, right_tokens


def read_dataset(args, data_path, answer_path):
    if answer_path is not None:
        answers = json.load(open(answer_path))
    dataset = []
    max_tokens_for_doc = args.seq_length - 3
    group_index = 0

    for line in open(data_path, mode="r", encoding="utf-8"):
        example = json.loads(line)
        options = example["candidates"]
        for context in example["content"]:
            chid_tokens = tokenize_chid(context)
            tags = [token for token in chid_tokens if "#idiom" in token]
            for tag in tags:
                if answer_path is not None:
                    tgt = answers[tag]
                else:
                    tgt = -1
                tokens = []
                for i, token in enumerate(chid_tokens):
                    if "#idiom" in token:
                        sub_tokens = [str(token)]
                    else:
                        sub_tokens = args.tokenizer.tokenize(token)
                    for sub_token in sub_tokens:
                        tokens.append(sub_token)
                idiom_index = tokens.index(tag)
                left_tokens, right_tokens = add_tokens_around(tokens, idiom_index, max_tokens_for_doc - 1)

                for i in range(len(left_tokens)):
                    if "#idiom" in left_tokens[i] and left_tokens[i] != tag:
                        left_tokens[i] = MASK_TOKEN
                for i in range(len(right_tokens)):
                    if "#idiom" in right_tokens[i] and right_tokens[i] != tag:
                        right_tokens[i] = MASK_TOKEN

                dataset.append(([], tgt, [], tag, group_index))

                for option in options:
                    option_tokens = args.tokenizer.tokenize(option)
                    tokens = [CLS_TOKEN] + option_tokens + [SEP_TOKEN] + left_tokens + [SEP_TOKEN] + right_tokens + [SEP_TOKEN]

                    src = args.tokenizer.convert_tokens_to_ids(tokens)[: args.seq_length]
                    seg = [0] * len(src)

                    while len(src) < args.seq_length:
                        src.append(0)
                        seg.append(0)

                    dataset[-1][0].append(src)
                    dataset[-1][2].append(seg)

                while len(dataset[-1][0]) < args.max_choices_num:
                    dataset[-1][0].append([0] * args.seq_length)
                    dataset[-1][2].append([0] * args.seq_length)
        group_index += 1

    return dataset


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    parser.add_argument("--train_answer_path", type=str, required=True,
                        help="Path of the answers for trainset.")
    parser.add_argument("--dev_answer_path", type=str, required=True,
                        help="Path of the answers for devset.")

    parser.add_argument("--max_choices_num", default=10, type=int,
                        help="The maximum number of cadicate answer, shorter than this will be padded.")

    args = parser.parse_args()

    args.labels_num = args.max_choices_num

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Build tokenizer.
    args.tokenizer = CharTokenizer(args)

    # Build multiple choice model.
    model = MultipleChoice(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)

    # Training phase.
    trainset = read_dataset(args, args.train_path, args.train_answer_path)
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    src = torch.LongTensor([example[0] for example in trainset])
    tgt = torch.LongTensor([example[1] for example in trainset])
    seg = torch.LongTensor([example[2] for example in trainset])

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model

    total_loss, result, best_result = 0.0, 0.0, 0.0

    print("Start training.")

    for epoch in range(1, args.epochs_num + 1):
        model.train()
        for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):

            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch)
            total_loss += loss.item()

            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0

        result = evaluate(args, read_dataset(args, args.dev_path, args.dev_answer_path))
        if result[0] > best_result:
            best_result = result[0]
            save_model(model, args.output_model_path)


if __name__ == "__main__":
    main()
