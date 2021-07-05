"""
This script provides an example to wrap UER-py for multi-task classification.
"""
import sys
import os
import random
import argparse
import torch
import torch.nn as nn

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.layers import *
from uer.encoders import *
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.opts import *
from finetune.run_classifier import count_labels_num, batch_loader, build_optimizer, load_or_initialize_parameters, train_model, read_dataset, evaluate


class MultitaskClassifier(nn.Module):
    def __init__(self, args):
        super(MultitaskClassifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.pooling = args.pooling
        self.output_layers_1 = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size) for _ in args.labels_num_list])
        self.output_layers_2 = nn.ModuleList([nn.Linear(args.hidden_size, labels_num) for labels_num in args.labels_num_list])

        self.dataset_id = 0

    def forward(self, src, tgt, seg, soft_tgt=None):
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
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layers_1[self.dataset_id](output))
        logits = self.output_layers_2[self.dataset_id](output)
        if tgt is not None:
            loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits
        else:
            return None, logits

    def change_dataset(self, dataset_id):
        self.dataset_id = dataset_id


def pack_dataset(dataset, dataset_id, batch_size):
    packed_dataset = []
    src_batch, tgt_batch, seg_batch = [], [], []
    for i, sample in enumerate(dataset):
        src_batch.append(sample[0])
        tgt_batch.append(sample[1])
        seg_batch.append(sample[2])
        if (i + 1) % batch_size == 0:
            packed_dataset.append((dataset_id, torch.LongTensor(src_batch), torch.LongTensor(tgt_batch), torch.LongTensor(seg_batch)))
            src_batch, tgt_batch, seg_batch = [], [], []
            continue
    if len(src_batch) > 0:
        packed_dataset.append((dataset_id, torch.LongTensor(src_batch), torch.LongTensor(tgt_batch), torch.LongTensor(seg_batch)))

    return packed_dataset


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--dataset_path_list", default=[], nargs='+', type=str, help="Dataset path list.")
    parser.add_argument("--output_model_path", default="models/multitask_classifier_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default=None, type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--spm_model_path", default=None, type=str,
                        help="Path of the sentence piece model.")    
    parser.add_argument("--config_path", default="models/bert/base_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    model_opts(parser)
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",                           
                        help="Pooling type.")

    # Tokenizer options.
    tokenizer_opts(parser)

    # Optimizer options.
    optimization_opts(parser)

    # Training options.
    training_opts(parser)

    args = parser.parse_args()

    args.soft_targets = False

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    args.labels_num_list = [count_labels_num(os.path.join(path, "train.tsv")) for path in args.dataset_path_list]

    args.datasets_num = len(args.dataset_path_list)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build multi-task classification model.
    model = MultitaskClassifier(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)
    args.model = model

    # Training phase.
    dataset_list = [read_dataset(args, os.path.join(path, "train.tsv")) for path in args.dataset_path_list]
    packed_dataset_list = [pack_dataset(dataset, i, args.batch_size) for i, dataset in enumerate(dataset_list)]

    packed_dataset_all = []
    for packed_dataset in packed_dataset_list:
        packed_dataset_all += packed_dataset

    random.shuffle(packed_dataset_all)
    instances_num = sum([len(dataset) for dataset in dataset_list])
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
        args.amp = amp

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    total_loss, result, best_result = 0.0, 0.0, 0.0

    print("Start training.")

    for epoch in range(1, args.epochs_num + 1):
        model.train()
        for i, (dataset_id, src_batch, tgt_batch, seg_batch) in enumerate(packed_dataset_all):
            if hasattr(model, "module"):
                model.module.change_dataset(dataset_id)
            else:
                model.change_dataset(dataset_id)
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, None)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0

        for dataset_id, path in enumerate(args.dataset_path_list):
            args.labels_num = args.labels_num_list[dataset_id]
            if hasattr(model, "module"):
                model.module.change_dataset(dataset_id)
            else:
                model.change_dataset(dataset_id)
            result = evaluate(args, read_dataset(args, os.path.join(path, "dev.tsv")))

    save_model(model, args.output_model_path)


if __name__ == "__main__":
    main()
