"""
This script provides an example to wrap UER-py for multi-label classification.
"""
import sys
import os
import random
import argparse
import torch
import torch.nn as nn
import time
import datetime
import json

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.embeddings import *
from uer.encoders import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.utils.logging import init_logger
from uer.utils.misc import pooling
from uer.model_saver import save_model
from uer.opts import finetune_opts, tokenizer_opts, adv_opts
from finetune.run_classifier import load_or_initialize_parameters, build_optimizer, batch_loader


class MultilabelClassifier(nn.Module):
    def __init__(self, args):
        super(MultilabelClassifier, self).__init__()
        self.embedding = Embedding(args)
        for embedding_name in args.embedding:
            tmp_emb = str2embedding[embedding_name](args, len(args.tokenizer.vocab))
            self.embedding.update(tmp_emb, embedding_name)
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling_type = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def forward(self, src, tgt, seg):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output = self.encoder(emb, seg)
        # Target.
        output = pooling(output, seg, self.pooling_type)
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        if tgt is not None:
            probs_batch = nn.Sigmoid()(logits)
            loss = nn.BCELoss()(probs_batch, tgt)
            return loss, logits
        else:
            return None, logits


def count_labels_num(path):
    labels_set, columns = set(), {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.rstrip("\r\n").split("\t")):
                    columns[column_name] = i
                continue
            line = line.rstrip("\r\n").split("\t")
            label = set(line[columns["label"]].split(","))
            labels_set |= label
    return len(labels_set)


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.rstrip("\r\n").split("\t")):
                    columns[column_name] = i
                continue
            line = line.rstrip("\r\n").split("\t")
            tgt = [0] * args.labels_num
            for idx in [int(_) for _ in line[columns["label"]].split(",")]:
                tgt[idx] = 1
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
            if len(src) < args.seq_length:
                PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
                src += [PAD_ID] * (args.seq_length - len(src))
                seg += [0] * (args.seq_length - len(seg))

            dataset.append((src, tgt, seg))

    return dataset


def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)

    loss, _ = model(src_batch, tgt_batch, seg_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    loss.backward()

    if args.use_adv and args.adv_type == "fgm":
        args.adv_method.attack(epsilon=args.fgm_epsilon)
        loss_adv, _ = model(src_batch, tgt_batch, seg_batch)
        if torch.cuda.device_count() > 1:
            loss_adv = torch.mean(loss_adv)
        loss_adv.backward()
        args.adv_method.restore()

    if args.use_adv and args.adv_type == "pgd":
        K = args.pgd_k
        args.adv_method.backup_grad()
        for t in range(K):
            # apply the perturbation to embedding
            args.adv_method.attack(epsilon=args.pgd_epsilon, alpha=args.pgd_alpha,
                                   is_first_attack=(t == 0))
            if t != K - 1:
                model.zero_grad()
            else:
                args.adv_method.restore_grad()
            loss_adv, _ = model(src_batch, tgt_batch, seg_batch)
            if torch.cuda.device_count() > 1:
                loss_adv = torch.mean(loss_adv)
            loss_adv.backward()
        args.adv_method.restore()

    optimizer.step()
    scheduler.step()

    return loss


def evaluate(args, dataset):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.tensor([sample[1] for sample in dataset], dtype=torch.float)
    seg = torch.LongTensor([sample[2] for sample in dataset])

    batch_size = args.batch_size

    correct = 0

    args.model.eval()

    for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        with torch.no_grad():
            _, logits = args.model(src_batch, tgt_batch, seg_batch)        
        probs_batch = nn.Sigmoid()(logits)
        predict_label_batch = (probs_batch > 0.5).float()
        gold = tgt_batch

        for k in range(len(predict_label_batch)):
            correct += predict_label_batch[k].equal(gold[k]) 

    args.logger.info("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))
    return correct / len(dataset)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    tokenizer_opts(parser)

    adv_opts(parser)

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)
    # Count the number of labels.
    args.labels_num = count_labels_num(args.train_path)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)
    set_seed(args.seed)
    
    # Build classification model.
    model = MultilabelClassifier(args)

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
        tgt = torch.tensor([sample[1] for sample in trainset], dtype=torch.float)
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
