"""
This script provides an example to wrap UER-py for SimCSE.
"""
import sys
import os
import random
import argparse
import math
import scipy.stats
import torch
import torch.nn as nn
import numpy as np

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.embeddings import *
from uer.encoders import *
from uer.targets import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.utils.logging import init_logger
from uer.model_saver import save_model
from uer.opts import finetune_opts, tokenizer_opts
from finetune.run_classifier import count_labels_num, build_optimizer, load_or_initialize_parameters
from finetune.run_classifier_siamese import batch_loader


class SimCSE(nn.Module):
    def __init__(self, args):
        super(SimCSE, self).__init__()
        self.embedding = Embedding(args)
        for embedding_name in args.embedding:
            tmp_emb = str2embedding[embedding_name](args, len(args.tokenizer.vocab))
            self.embedding.update(tmp_emb, embedding_name)
        self.encoder = str2encoder[args.encoder](args)

        self.pooling_type = args.pooling

    def forward(self, src, seg):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        # Embedding.
        emb_0 = self.embedding(src[0], seg[0])
        emb_1 = self.embedding(src[1], seg[1])
        # Encoder.
        output_0 = self.encoder(emb_0, seg[0])
        output_1 = self.encoder(emb_1, seg[1])
        # Target.
        features_0 = self.pooling(output_0, seg[0], self.pooling_type)
        features_1 = self.pooling(output_1, seg[1], self.pooling_type)

        return features_0, features_1

    def pooling(self, memory_bank, seg, pooling_type):
        seg = torch.unsqueeze(seg, dim=-1).type(torch.float)
        memory_bank = memory_bank * seg
        if pooling_type == "mean":
            features = torch.sum(memory_bank, dim=1)
            features = torch.div(features, torch.sum(seg, dim=1))
        elif pooling_type == "last":
            features = memory_bank[torch.arange(memory_bank.shape[0]), torch.squeeze(torch.sum(seg, dim=1).type(torch.int64) - 1), :]
        elif pooling_type == "max":
            features = torch.max(memory_bank + (seg - 1) * sys.maxsize, dim=1)[0]
        else:
            features = memory_bank[:, 0, :]
        return features


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
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
            else:
                text_a = line[columns["text_a"]]
                text_b = text_a
            src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
            src_b = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
            seg_a = [1] * len(src_a)
            seg_b = [1] * len(src_b)
            PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]

            if len(src_a) >= args.seq_length:
                src_a = src_a[:args.seq_length]
                seg_a = seg_a[:args.seq_length]
            while len(src_a) < args.seq_length:
                src_a.append(PAD_ID)
                seg_a.append(0)

            if len(src_b) >= args.seq_length:
                src_b = src_b[:args.seq_length]
                seg_b = seg_b[:args.seq_length]
            while len(src_b) < args.seq_length:
                src_b.append(PAD_ID)
                seg_b.append(0)

            if "label" in columns:
                tgt = float(line[columns["label"]])
                dataset.append(((src_a, src_b), tgt, (seg_a, seg_b)))
            else:
                dataset.append(((src_a, src_a), -1, (seg_a, seg_a)))
    return dataset


def evaluate(args, dataset):
    src_a = torch.LongTensor([example[0][0] for example in dataset])
    src_b = torch.LongTensor([example[0][1] for example in dataset])
    tgt = torch.FloatTensor([example[1] for example in dataset])
    seg_a = torch.LongTensor([example[2][0] for example in dataset])
    seg_b = torch.LongTensor([example[2][1] for example in dataset])

    all_similarities = []
    batch_size = args.batch_size
    args.model.eval()

    for i, (src_batch, tgt_batch, seg_batch) in enumerate(batch_loader(batch_size, (src_a, src_b), tgt, (seg_a, seg_b))):

        src_a_batch, src_b_batch = src_batch
        seg_a_batch, seg_b_batch = seg_batch

        src_a_batch = src_a_batch.to(args.device)
        src_b_batch = src_b_batch.to(args.device)

        seg_a_batch = seg_a_batch.to(args.device)
        seg_b_batch = seg_b_batch.to(args.device)

        with torch.no_grad():
            features_0, features_1 = args.model((src_a_batch, src_b_batch), (seg_a_batch, seg_b_batch))
        similarity_matrix = similarity(features_0, features_1, 1)

        for j in range(similarity_matrix.size(0)):
            all_similarities.append(similarity_matrix[j][j].item())

    corrcoef = scipy.stats.spearmanr(tgt, all_similarities).correlation
    args.logger.info("Spearman's correlation: {:.4f}".format(corrcoef))
    return corrcoef


def similarity(x, y, temperature):
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    return torch.matmul(x, y.transpose(-2, -1)) / temperature


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    tokenizer_opts(parser)

    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluate frequency.")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model.
    model = SimCSE(args)

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

    total_loss, result, best_result = 0.0, 0.0, 0.0

    args.logger.info("Start training.")
    for epoch in range(1, args.epochs_num + 1):
        random.shuffle(trainset)
        src_a = torch.LongTensor([example[0][0] for example in trainset])
        src_b = torch.LongTensor([example[0][1] for example in trainset])
        tgt = torch.FloatTensor([example[1] for example in trainset])
        seg_a = torch.LongTensor([example[2][0] for example in trainset])
        seg_b = torch.LongTensor([example[2][1] for example in trainset])

        model.train()
        for i, (src_batch, tgt_batch, seg_batch) in enumerate(batch_loader(batch_size, (src_a, src_b), tgt, (seg_a, seg_b))):
            model.zero_grad()

            src_a_batch, src_b_batch = src_batch
            seg_a_batch, seg_b_batch = seg_batch

            src_a_batch = src_a_batch.to(args.device)
            src_b_batch = src_b_batch.to(args.device)

            seg_a_batch = seg_a_batch.to(args.device)
            seg_b_batch = seg_b_batch.to(args.device)

            features_0, features_1 = model((src_a_batch, src_b_batch), (seg_a_batch, seg_b_batch))

            similarity_matrix = similarity(features_0, features_1, args.temperature)
            tgt_batch = torch.arange(similarity_matrix.size(0), device=similarity_matrix.device, dtype=torch.long)
            loss = nn.CrossEntropyLoss()(similarity_matrix, tgt_batch)

            loss.backward()

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                args.logger.info("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}"
                                 .format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0

            if (i + 1) % args.eval_steps == 0 or (i + 1) == math.ceil(instances_num / batch_size):
                result = evaluate(args, read_dataset(args, args.dev_path))
                args.logger.info("Epoch id: {}, Training steps: {}, Evaluate result: {}, Best result: {}"
                                 .format(epoch, i + 1, result, best_result))
                if result > best_result:
                    best_result = result
                    save_model(model, args.output_model_path)
                    args.logger.info("It is the best model until now. Save it to {}".format(args.output_model_path))


if __name__ == "__main__":
    main()
