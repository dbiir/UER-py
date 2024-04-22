"""
This script provides an exmaple to wrap UER-py for document-based question answering.
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
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.utils.logging import init_logger
from uer.model_saver import save_model
from uer.opts import finetune_opts, tokenizer_opts, adv_opts
from finetune.run_classifier import Classifier, count_labels_num, build_optimizer, batch_loader, train_model, load_or_initialize_parameters


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.rstrip("\r\n").split("\t")):
                    columns[column_name] = i
                continue
            line = line.rstrip("\r\n").split("\t")
            qid = int(line[columns["qid"]])
            tgt = int(line[columns["label"]])
            text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
            src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
            src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
            src = src_a + src_b
            seg = [1] * len(src_a) + [2] * len(src_b)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            if len(src) < args.seq_length:
                PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
                src += [PAD_ID] * (args.seq_length - len(src))
                seg += [0] * (args.seq_length - len(seg))
            dataset.append((src, tgt, seg, qid))

    return dataset


def gen_dataset_groupby_qid(dataset, logits_all):
    dataset_groupby_qid, correct_answer_orders, scores = [], [], []
    for i in range(len(dataset)):
        label = dataset[i][1]
        if i == 0:
            qid = dataset[i][3]
            # Order of the current sentence in the document.
            current_order = 0
            scores.append(float(logits_all[i][1].item()))
            if label == 1:
                # Occasionally, more than one sentences in a document contain answers.
                correct_answer_orders.append(current_order)
            current_order += 1
            continue
        if qid == dataset[i][3]:
            scores.append(float(logits_all[i][1].item()))
            if label == 1:
                correct_answer_orders.append(current_order)
            current_order += 1
        else:
            # For each question, we record which sentences contain answers
            # and the scores of all sentences in the document.
            dataset_groupby_qid.append((qid, correct_answer_orders, scores))
            correct_answer_orders, scores, current_order = [], [], 0
            qid = dataset[i][3]
            scores.append(float(logits_all[i][1].item()))
            if label == 1:
                correct_answer_orders.append(current_order)
            current_order += 1
    dataset_groupby_qid.append((qid, correct_answer_orders, scores))
    return dataset_groupby_qid


def evaluate(args, dataset):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    batch_size = args.batch_size
    instances_num = src.size()[0]

    args.model.eval()

    for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        with torch.no_grad():
            loss, logits = args.model(src_batch, tgt_batch, seg_batch)
        if i == 0:
            logits_all = logits
        if i >= 1:
            logits_all = torch.cat((logits_all, logits), 0)

    # To calculate MRR, the results are grouped by qid.
    dataset_groupby_qid = gen_dataset_groupby_qid(dataset, logits_all)

    reciprocal_rank = []
    for _, correct_answer_orders, scores in dataset_groupby_qid:
        if len(correct_answer_orders) == 1:
            sorted_scores = sorted(scores, reverse=True)
            for j in range(len(sorted_scores)):
                if sorted_scores[j] == scores[correct_answer_orders[0]]:
                    reciprocal_rank.append(1 / (j + 1))
        else:
            current_rank = len(scores)
            sorted_scores = sorted(scores, reverse=True)
            for i in range(len(correct_answer_orders)):
                for j in range(len(scores)):
                    if sorted_scores[j] == scores[correct_answer_orders[i]] and j < current_rank:
                        current_rank = j
            reciprocal_rank.append(1 / (current_rank + 1))

    MRR = sum(reciprocal_rank) / len(reciprocal_rank)
    args.logger.info("Mean Reciprocal Rank: {:.4f}".format(MRR))
    return MRR


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    tokenizer_opts(parser)

    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")

    adv_opts(parser)

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

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)

    # Training phase.
    trainset = read_dataset(args, args.train_path)
    instances_num = len(trainset)
    batch_size = args.batch_size

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    args.logger.info("Batch size: {}".format(batch_size))
    args.logger.info("The number of training instances: {}".format(instances_num))

    optimizer, scheduler = build_optimizer(args, model)

    if torch.cuda.device_count() > 1:
        args.logger.info("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model

    if args.use_adv:
        args.adv_method = str2adv[args.adv_type](model)

    total_loss, result, best_result = 0.0, 0.0, 0.0

    args.logger.info("Start training.")

    for epoch in range(1, args.epochs_num + 1):
        random.shuffle(trainset)
        src = torch.LongTensor([example[0] for example in trainset])
        tgt = torch.LongTensor([example[1] for example in trainset])
        seg = torch.LongTensor([example[2] for example in trainset])

        model.train()
        for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                args.logger.info("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0

        result = evaluate(args, read_dataset(args, args.dev_path))
        if result > best_result:
            best_result = result
            save_model(model, args.output_model_path)

    # Evaluation phase.
    if args.test_path is not None:
        args.logger.info("Test set evaluation.")
        if torch.cuda.device_count() > 1:
            args.model.module.load_state_dict(torch.load(args.output_model_path))
        else:
            args.model.load_state_dict(torch.load(args.output_model_path))
        evaluate(args, read_dataset(args, args.test_path))


if __name__ == "__main__":
    main()
