"""
  This script provides an example to wrap UER-py for NER.
"""
import sys
import os
import random
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.layers import *
from uer.encoders import *
from uer.utils.config import load_hyperparam
from uer.utils.optimizers import *
from uer.utils.constants import *
from uer.utils.vocab import Vocab
from uer.utils.seed import set_seed
from uer.utils.tokenizers import *
from uer.model_saver import save_model
from uer.opts import finetune_opts
from finetune.run_classifier import build_optimizer, load_or_initialize_parameters


class NerTagger(nn.Module):
    def __init__(self, args):
        super(NerTagger, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.output_layer = nn.Linear(args.hidden_size, self.labels_num)
        self.crf_target = args.crf_target
        if args.crf_target:
            from torchcrf import CRF
            self.crf = CRF(self.labels_num, batch_first=True)
            self.seq_length = args.seq_length

    def forward(self, src, tgt, seg):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size x seq_length]
            seg: [batch_size x seq_length]
        Returns:
            loss: Sequence labeling loss.
            logits: Output logits.
        """
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output = self.encoder(emb, seg)

        # Target.
        logits = self.output_layer(output)
        if self.crf_target:
            tgt_mask = seg.type(torch.uint8)
            pred = self.crf.decode(logits, mask=tgt_mask)
            for j in range(len(pred)):
                while len(pred[j]) < self.seq_length:
                    pred[j].append(self.labels_num - 1)
            pred = torch.tensor(pred).contiguous().view(-1)
            if tgt is not None:
                loss = -self.crf(F.log_softmax(logits, 2), tgt, mask=tgt_mask, reduction='mean')
                return loss, pred
            else:
                return None, pred
        else:
            tgt_mask = seg.contiguous().view(-1).float()
            logits = logits.contiguous().view(-1, self.labels_num)
            pred = logits.argmax(dim=-1)
            if tgt is not None:
                tgt = tgt.contiguous().view(-1, 1)
                one_hot = torch.zeros(tgt.size(0), self.labels_num). \
                    to(torch.device(tgt.device)). \
                    scatter_(1, tgt, 1.0)
                numerator = -torch.sum(nn.LogSoftmax(dim=-1)(logits) * one_hot, 1)
                numerator = torch.sum(tgt_mask * numerator)
                denominator = torch.sum(tgt_mask) + 1e-6
                loss = numerator / denominator
                return loss, pred
            else:
                return None, pred


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().split("\t")
            labels = line[columns["label"]]
            tgt = [args.l2i[l] for l in labels.split(" ")]

            text_a = line[columns["text_a"]]
            src = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_a))
            seg = [1] * len(src)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                tgt = tgt[: args.seq_length]
                seg = seg[: args.seq_length]
            PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
            while len(src) < args.seq_length:
                src.append(PAD_ID)
                tgt.append(args.labels_num - 1)
                seg.append(0)
            dataset.append([src, tgt, seg])

    return dataset


def batch_loader(batch_size, src, tgt, seg):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size, :]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        yield src_batch, tgt_batch, seg_batch
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :, :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        yield src_batch, tgt_batch, seg_batch


def train(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)

    loss, _ = model(src_batch, tgt_batch, seg_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    if args.fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()
    scheduler.step()

    return loss


def evaluate(args, dataset):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    instances_num = src.size(0)
    batch_size = args.batch_size

    correct, gold_entities_num, pred_entities_num = 0, 0, 0

    args.model.eval()

    for i, (src_batch, tgt_batch, seg_batch) in enumerate(batch_loader(batch_size, src, tgt, seg)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        loss, pred = args.model(src_batch, tgt_batch, seg_batch)

        gold = tgt_batch.contiguous().view(-1, 1)

        for j in range(gold.size()[0]):
            if gold[j].item() in args.begin_ids:
                gold_entities_num += 1

        for j in range(pred.size()[0]):
            if pred[j].item() in args.begin_ids and gold[j].item() != args.l2i["[PAD]"]:
                pred_entities_num += 1

        pred_entities_pos, gold_entities_pos = set(), set()

        for j in range(gold.size()[0]):
            if gold[j].item() in args.begin_ids:
                start = j
                for k in range(j + 1, gold.size()[0]):
                    if gold[k].item() == args.l2i["[PAD]"] or gold[k].item() == args.l2i["O"] or gold[k].item() in args.begin_ids:
                        end = k - 1
                        break
                else:
                    end = gold.size()[0] - 1
                gold_entities_pos.add((start, end))

        for j in range(pred.size()[0]):
            if pred[j].item() in args.begin_ids and gold[j].item() != args.l2i["[PAD]"]:
                start = j
                for k in range(j + 1, pred.size()[0]):
                    if pred[k].item() == args.l2i["[PAD]"] or pred[k].item() == args.l2i["O"] or pred[k].item() in args.begin_ids:
                        end = k - 1
                        break
                else:
                    end = pred.size()[0] - 1
                pred_entities_pos.add((start, end))

        for entity in pred_entities_pos:
            if entity not in gold_entities_pos:
                continue
            for j in range(entity[0], entity[1] + 1):
                if gold[j].item() != pred[j].item():
                    break
            else:
                correct += 1

    print("Report precision, recall, and f1:")
    eps = 1e-9
    p = correct / (pred_entities_num + eps)
    r = correct / (gold_entities_num + eps)
    f1 = 2 * p * r / (p + r + eps)
    print("{:.3f}, {:.3f}, {:.3f}".format(p, r, f1))

    return f1


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    parser.add_argument("--label2id_path", type=str, required=True,
                        help="Path of the label2id file.")
    parser.add_argument("--crf_target", action="store_true",
                        help="Use CRF loss as the target function or not, default False.")

    args = parser.parse_args()

    # Load the hyperparameters of the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    args.begin_ids = []

    with open(args.label2id_path, mode="r", encoding="utf-8") as f:
        l2i = json.load(f)
        print("Labels: ", l2i)
        l2i["[PAD]"] = len(l2i)
        for label in l2i:
            if label.startswith("B"):
                args.begin_ids.append(l2i[label])

    args.l2i = l2i

    args.labels_num = len(l2i)

    args.tokenizer = SpaceTokenizer(args)

    # Build sequence labeling model.
    model = NerTagger(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)

    # Training phase.
    instances = read_dataset(args, args.train_path)

    src = torch.LongTensor([ins[0] for ins in instances])
    tgt = torch.LongTensor([ins[1] for ins in instances])
    seg = torch.LongTensor([ins[2] for ins in instances])

    instances_num = src.size(0)
    batch_size = args.batch_size
    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level = args.fp16_opt_level)

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model

    total_loss, f1, best_f1 = 0.0, 0.0, 0.0

    print("Start training.")

    for epoch in range(1, args.epochs_num + 1):
        model.train()
        for i, (src_batch, tgt_batch, seg_batch) in enumerate(batch_loader(batch_size, src, tgt, seg)):
            loss = train(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0

        f1 = evaluate(args, read_dataset(args, args.dev_path))
        if f1 > best_f1:
            best_f1 = f1
            save_model(model, args.output_model_path)
        else:
            continue

    # Evaluation phase.
    if args.test_path is not None:
        print("Test set evaluation.")
        if torch.cuda.device_count() > 1:
            args.model.module.load_state_dict(torch.load(args.output_model_path))
        else:
            args.model.load_state_dict(torch.load(args.output_model_path))
        evaluate(args, read_dataset(args, args.test_path))


if __name__ == "__main__":
    main()
