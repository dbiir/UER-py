"""
  This script provides an example to wrap UER-py for NER.
"""
import sys
import os
import argparse
import json
import torch
import torch.nn as nn
from torchcrf import CRF
from tqdm import trange

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

import conlleval
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
        self.output_dim = args.hidden_size
        self.use_birnn = args.use_birnn
        self.use_crf = args.use_crf
        if args.use_birnn:
            self.birnn = nn.LSTM(
                args.hidden_size,
                args.rnn_dim,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )
            self.output_dim = args.rnn_dim * 2
        self.crf = CRF(args.labels_num, batch_first=True)
        args.crf = self.crf
        self.output_layer = nn.Linear(self.output_dim, self.labels_num)

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
        # output = output[0]
        if self.use_birnn:
            output, _ = self.birnn(output)

        logits = self.output_layer(output)
        if tgt is not None:
            if self.use_crf:
                tgt_mask = (tgt < self.labels_num - 1).float().to(tgt.device)
                loss = -1 * self.crf(logits, tgt, mask=tgt_mask.byte())
            else:
                tgt = tgt.contiguous().view(-1, 1)
                one_hot = (
                    torch.zeros(tgt.size(0), self.labels_num)
                    .to(torch.device(tgt.device))
                    .scatter_(1, tgt, 1.0)
                )
                _logits = logits.contiguous().view(-1, self.labels_num)
                numerator = -torch.sum(nn.LogSoftmax(dim=-1)(_logits) * one_hot, 1)
                tgt = tgt.contiguous().view(-1)
                tgt_mask = ((tgt < self.labels_num - 1).float().to(torch.device(tgt.device)))

                numerator = torch.sum(tgt_mask * numerator)
                denominator = torch.sum(tgt_mask) + 1e-6
                loss = numerator / denominator
            return loss, logits
        else:
            return None, logits


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
            text_a = text_a.lower()  # 因为用的是 uncased vocab
            tokens = args.tokenizer.tokenize(text_a)
            src = args.tokenizer.convert_tokens_to_ids(tokens)
            seg = [1] * len(src)

            assert len(src) == len(tgt) == len(seg)

            if args.encoder == 'transformer':
                # special [CLS] + tokens + [SEP] for input
                if len(src) > args.seq_length - 2:
                    src = src[: args.seq_length - 2]
                    tgt = tgt[: args.seq_length - 2]
                    seg = seg[: args.seq_length - 2]
                    tokens = tokens[: args.seq_length - 2]
                PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
                CLS_ID = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN])[0]
                SEP_ID = args.tokenizer.convert_tokens_to_ids([SEP_TOKEN])[0]
                tokens = [CLS_TOKEN] + tokens + [SEP_TOKEN]
                src = [CLS_ID] + src + [SEP_ID]
                tgt = [args.l2i["O"]] + tgt + [args.l2i["O"]]
                seg = [1] + seg + [1]

            while len(src) < args.seq_length:
                src.append(PAD_ID)
                tgt.append(args.labels_num - 1)
                seg.append(0)
            dataset.append([src, tgt, seg, tokens])

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
    all_ori_tokens_eval = [sample[3] for sample in dataset]

    batch_size = args.batch_size

    args.model.eval()

    print("***** Running eval *****")
    pred_labels = []
    ori_labels = []
    for (src_batch, tgt_batch, seg_batch) in batch_loader(batch_size, src, tgt, seg):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        _, logits = args.model(src_batch, tgt_batch, seg_batch)
        
        if args.use_crf:
            pred = args.crf.decode(logits, seg_batch.byte())
        else:
            pred = logits.argmax(dim=-1)
            pred = pred.detach().cpu().numpy()
        
        tgt_batch = tgt_batch.detach().cpu().numpy()
        for predict_idx, gold_idx in zip(pred, tgt_batch):
            pred_label = []
            gold_label = []
            for _pred_idx, _glod_idx in zip(predict_idx, gold_idx):
                pred_label.append(args.i2l[_pred_idx])
                gold_label.append(args.i2l[_glod_idx])
            pred_labels.append(pred_label)
            ori_labels.append(gold_label)

    eval_list = []
    for ori_tokens, oril, prel in zip(all_ori_tokens_eval, ori_labels, pred_labels):
        for ot, ol, pl in zip(ori_tokens, oril, prel):
            if ot in [CLS_TOKEN, SEP_TOKEN]:
                continue
            eval_list.append(f"{ot} {ol} {pl}\n")
        
        eval_list.append("\n")

    # eval the model
    print(''.join(eval_list))
    counts = conlleval.evaluate(eval_list)
    conlleval.report(counts)

    # namedtuple('Metrics', 'tp fp fn prec rec fscore')
    overall, _ = conlleval.metrics(counts)
    return overall.fscore


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    finetune_opts(parser)

    parser.add_argument("--label2id_path", type=str, required=True, 
                        help="Path of the label2id file.")
    parser.add_argument("--use_crf", action="store_true", 
                        help="Use CRF layer or not, default False.")
    parser.add_argument("--use_birnn", action="store_true",
                        help="Use Bidirecrional RNN layer or not, default False.")
    parser.add_argument("--rnn_dim", type=int, default=256, 
                        help="RNN hidden size, default 256.")

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
    args.i2l = {i: l for l, i in l2i.items()}

    args.labels_num = len(l2i)

    args.tokenizer = SpaceTokenizer(args)

    # Build sequence labeling model.
    model = NerTagger(args)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

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
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model

    total_loss, f1, best_f1 = 0.0, 0.0, 0.0

    print("Start training.")

    dev_dataset = read_dataset(args, args.dev_path)
    for epoch in trange(args.epochs_num, desc="Epoch"):
        model.train()
        for i, (src_batch, tgt_batch, seg_batch) in enumerate(batch_loader(batch_size, src, tgt, seg)):
            loss = train(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch + 1, i + 1, total_loss / args.report_steps))
                total_loss = 0.0
        
        f1 = evaluate(args, dev_dataset)
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
