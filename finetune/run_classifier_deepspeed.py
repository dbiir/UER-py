"""
This script provides an example to use DeepSpeed for classification.
"""
import sys
import os
import random
import argparse
import torch
import torch.nn as nn
import deepspeed
import torch.distributed as dist

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.opts import deepspeed_opts
from finetune.run_classifier import *


def read_dataset(args, path, split):
    dataset, columns = [], {}
    if split:
        for i in range(args.world_size):
            dataset.append([])
        index = 0
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.rstrip("\r\n").split("\t")):
                    columns[column_name] = i
                continue
            line = line.rstrip("\r\n").split("\t")
            tgt = int(line[columns["label"]])
            if args.soft_targets and "logits" in columns.keys():
                soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]
            if "text_b" not in columns:  # Sentence classification.
                text_a = line[columns["text_a"]]
                src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
                seg = [1] * len(src)
            else:  # Sentence-pair classification.
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
                src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
                src = src_a + src_b
                seg = [1] * len(src_a) + [2] * len(src_b)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
            while len(src) < args.seq_length:
                src.append(PAD_ID)
                seg.append(0)
            if split:
                if args.soft_targets and "logits" in columns.keys():
                    dataset[index].append((src, tgt, seg, soft_tgt))
                else:
                    dataset[index].append((src, tgt, seg))
                index += 1
                if index == args.world_size:
                    index = 0
            else:
                if args.soft_targets and "logits" in columns.keys():
                    dataset.append((src, tgt, seg, soft_tgt))
                else:
                    dataset.append((src, tgt, seg))
    if split:
        max_data_num_rank_index = 0
        max_data_num = len(dataset[0])
        for i in range(args.world_size):
            if len(dataset[i]) > max_data_num:
                max_data_num_rank_index = i
                max_data_num = len(dataset[i])
        for i in range(args.world_size):
            if len(dataset[i]) < max_data_num:
                dataset[i].append(dataset[max_data_num_rank_index][-1])

    return dataset


def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch=None):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)

    loss, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    model.backward(loss)

    model.step()

    return loss


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    parser.add_argument("--world_size", type=int, default=1,
                        help="Total number of processes (GPUs) for training.")

    tokenizer_opts(parser)

    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")

    deepspeed_opts(parser)

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    args.labels_num = count_labels_num(args.train_path)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model.
    model = Classifier(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    # Get logger.
    args.logger = init_logger(args)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    deepspeed.init_distributed()
    rank = dist.get_rank()
    args.rank = rank

    trainset = read_dataset(args, args.train_path, split=True)[args.rank]
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size
    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    custom_optimizer, custom_scheduler = build_optimizer(args, model)

    model, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=optimizer_grouped_parameters,
        args=args,
        optimizer=custom_optimizer,
        lr_scheduler=custom_scheduler,
        mpu=None,
        dist_init_required=False)

    src = torch.LongTensor([example[0] for example in trainset])
    tgt = torch.LongTensor([example[1] for example in trainset])
    seg = torch.LongTensor([example[2] for example in trainset])
    if args.soft_targets:
        soft_tgt = torch.FloatTensor([example[3] for example in trainset])
    else:
        soft_tgt = None

    args.model = model
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_loss, result, best_result, best_epoch = 0.0, 0.0, 0.0, 0

    result_tensor = torch.tensor(result).to(args.device)
    if args.rank == 0:
        args.logger.info("Batch size: {}".format(batch_size))
        args.logger.info("The number of training instances: {}".format(instances_num))
        args.logger.info("Start training.")

    for epoch in range(1, args.epochs_num + 1):
        model.train()
        for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, soft_tgt)):
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0 and args.rank == 0:
                args.logger.info("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0
        if args.rank == 0:
            result = evaluate(args, read_dataset(args, args.dev_path, split=False))
            result_tensor = torch.tensor(result[0]).to(args.device)
        dist.broadcast(result_tensor, 0, async_op=False)
        if result_tensor.float() >= best_result:
            best_result = result_tensor.float().item()
            best_epoch = epoch
        model.save_checkpoint(args.output_model_path, str(epoch))

    # Evaluation phase.
    if args.test_path is not None and args.rank == 0:
        args.logger.info("Test set evaluation.")
        model.load_checkpoint(args.output_model_path, str(best_epoch))
        evaluate(args, read_dataset(args, args.test_path, split=False), True)


if __name__ == "__main__":
    main()
