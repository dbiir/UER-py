"""
This script provides an example to wrap UER-py for classification with siamese network.
"""
import sys
import os
import random
import argparse
import collections
import torch
import torch.nn as nn

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
from uer.utils.misc import pooling
from uer.model_saver import save_model
from uer.opts import finetune_opts, tokenizer_opts
from finetune.run_classifier import count_labels_num, build_optimizer


class SiameseClassifier(nn.Module):
    def __init__(self, args):
        super(SiameseClassifier, self).__init__()
        self.embedding = Embedding(args)
        for embedding_name in args.embedding:
            tmp_emb = str2embedding[embedding_name](args, len(args.tokenizer.vocab))
            self.embedding.update(tmp_emb, embedding_name)
        self.encoder = DualEncoder(args)

        self.classifier = nn.Linear(4 * args.stream_0["hidden_size"], args.labels_num)
        self.pooling_type = args.pooling

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
        features_0, features_1 = output
        features_0 = pooling(features_0, seg[0], self.pooling_type)
        features_1 = pooling(features_1, seg[1], self.pooling_type)

        vectors_concat = []

        # concatenation
        vectors_concat.append(features_0)
        vectors_concat.append(features_1)
        # difference:
        vectors_concat.append(torch.abs(features_0 - features_1))
        # multiplication:
        vectors_concat.append(features_0 * features_1)

        features = torch.cat(vectors_concat, 1)

        logits = self.classifier(features)

        if tgt is not None:
            loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits
        else:
            return None, logits


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        state_dict = torch.load(args.pretrained_model_path, map_location="cpu")
        load_siamese_weights = False
        for key in state_dict.keys():
            if key.find("embedding_0") != -1:
                load_siamese_weights = True
                break
        if not load_siamese_weights:
            siamese_state_dict = collections.OrderedDict()
            for key in state_dict.keys():
                if key.split('.')[0] == "embedding":
                    siamese_state_dict["embedding.embedding_0." + ".".join(key.split('.')[1:])] = state_dict[key]
                    siamese_state_dict["embedding.embedding_1." + ".".join(key.split('.')[1:])] = state_dict[key]
                if key.split('.')[0] == "encoder":
                    siamese_state_dict["encoder.encoder_0." + ".".join(key.split('.')[1:])] = state_dict[key]
                    siamese_state_dict["encoder.encoder_1." + ".".join(key.split('.')[1:])] = state_dict[key]
            model.load_state_dict(siamese_state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def batch_loader(batch_size, src, tgt, seg):
    instances_num = tgt.size()[0]
    src_a, src_b = src
    seg_a, seg_b = seg
    for i in range(instances_num // batch_size):
        src_a_batch = src_a[i * batch_size : (i + 1) * batch_size, :]
        src_b_batch = src_b[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size]
        seg_a_batch = seg_a[i * batch_size : (i + 1) * batch_size, :]
        seg_b_batch = seg_b[i * batch_size : (i + 1) * batch_size, :]
        yield (src_a_batch, src_b_batch), tgt_batch, (seg_a_batch, seg_b_batch)
    if instances_num > instances_num // batch_size * batch_size:
        src_a_batch = src_a[instances_num // batch_size * batch_size :, :]
        src_b_batch = src_b[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :]
        seg_a_batch = seg_a[instances_num // batch_size * batch_size :, :]
        seg_b_batch = seg_b[instances_num // batch_size * batch_size :, :]
        yield (src_a_batch, src_b_batch), tgt_batch, (seg_a_batch, seg_b_batch)


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.rstrip("\r\n").split("\t")):
                    columns[column_name] = i
                continue
            line = line.rstrip("\r\n").split("\t")
            tgt = int(line[columns["label"]])

            text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
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

            dataset.append(((src_a, src_b), tgt, (seg_a, seg_b)))

    return dataset


def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch):
    model.zero_grad()

    src_a_batch, src_b_batch = src_batch
    seg_a_batch, seg_b_batch = seg_batch

    src_a_batch = src_a_batch.to(args.device)
    src_b_batch = src_b_batch.to(args.device)

    tgt_batch = tgt_batch.to(args.device)

    seg_a_batch = seg_a_batch.to(args.device)
    seg_b_batch = seg_b_batch.to(args.device)

    loss, _ = model((src_a_batch, src_b_batch), tgt_batch, (seg_a_batch, seg_b_batch))

    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    loss.backward()

    optimizer.step()
    scheduler.step()

    return loss


def evaluate(args, dataset):
    src_a = torch.LongTensor([example[0][0] for example in dataset])
    src_b = torch.LongTensor([example[0][1] for example in dataset])
    tgt = torch.LongTensor([example[1] for example in dataset])
    seg_a = torch.LongTensor([example[2][0] for example in dataset])
    seg_b = torch.LongTensor([example[2][1] for example in dataset])

    batch_size = args.batch_size

    correct = 0
    # Confusion matrix.
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    args.model.eval()

    for i, (src_batch, tgt_batch, seg_batch) in enumerate(batch_loader(batch_size, (src_a, src_b), tgt, (seg_a, seg_b))):

        src_a_batch, src_b_batch = src_batch
        seg_a_batch, seg_b_batch = seg_batch

        src_a_batch = src_a_batch.to(args.device)
        src_b_batch = src_b_batch.to(args.device)

        tgt_batch = tgt_batch.to(args.device)

        seg_a_batch = seg_a_batch.to(args.device)
        seg_b_batch = seg_b_batch.to(args.device)

        with torch.no_grad():
            _, logits = args.model((src_a_batch, src_b_batch), None, (seg_a_batch, seg_b_batch))
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        gold = tgt_batch
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()

    args.logger.debug("Confusion matrix:")
    args.logger.debug(confusion)
    args.logger.debug("Report precision, recall, and f1:")

    eps = 1e-9
    for i in range(confusion.size()[0]):
        p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
        r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
        f1 = 2 * p * r / (p + r + eps)
        args.logger.debug("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))

    args.logger.info("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))
    return correct / len(dataset), confusion


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    tokenizer_opts(parser)

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    args.labels_num = count_labels_num(args.train_path)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model.
    model = SiameseClassifier(args)

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
        tgt = torch.LongTensor([example[1] for example in trainset])
        seg_a = torch.LongTensor([example[2][0] for example in trainset])
        seg_b = torch.LongTensor([example[2][1] for example in trainset])

        model.train()
        for i, (src_batch, tgt_batch, seg_batch) in enumerate(batch_loader(batch_size, (src_a, src_b), tgt, (seg_a, seg_b))):
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                args.logger.info("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0

        result = evaluate(args, read_dataset(args, args.dev_path))
        if result[0] > best_result:
            best_result = result[0]
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
