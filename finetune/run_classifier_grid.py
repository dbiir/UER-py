"""
This script provides an example to wrap UER-py for classification with grid search.
"""
import sys
import os
import torch
import random
import argparse
from itertools import product

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.opts import *
from finetune.run_classifier import *


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    tokenizer_opts(parser)

    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")

    parser.add_argument("--batch_size_list", default=[32, 64], nargs='+', type=int,
                        help="A list of batch sizes for grid search.")
    parser.add_argument("--learning_rate_list", type=float, default=[3e-5, 1e-4, 3e-4], nargs='+',
                        help="A list of learning rate for grid search.")
    parser.add_argument("--epochs_num_list", type=int, default=[3, 5, 8], nargs='+',
                        help="A list of number of epochs for grid search.")

    adv_opts(parser)

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    args.labels_num = count_labels_num(args.train_path)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Get logger.
    args.logger = init_logger(args)

    best_acc = 0
    config = {}

    # Build dataset
    trainset = read_dataset(args, args.train_path)
    instances_num = len(trainset)

    for batch_size, learning_rate, epochs_num in product(args.batch_size_list, args.learning_rate_list, args.epochs_num_list):

        args.learning_rate = learning_rate
        args.batch_size = batch_size
        args.epochs_num = epochs_num

        args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

        # Build classification model.
        model = Classifier(args)

        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(args.device)
        load_or_initialize_parameters(args, model)
        optimizer, scheduler = build_optimizer(args, model)

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        args.model = model

        if args.use_adv:
            args.adv_method = str2adv[args.adv_type](model)

        # Training phase.

        total_loss, _, _ = 0., 0., 0.

        for _ in range(1, args.epochs_num + 1):
            random.shuffle(trainset)
            src = torch.LongTensor([example[0] for example in trainset])
            tgt = torch.LongTensor([example[1] for example in trainset])
            seg = torch.LongTensor([example[2] for example in trainset])
            if args.soft_targets:
                soft_tgt = torch.FloatTensor([example[3] for example in trainset])
            else:
                soft_tgt = None

            model.train()
            for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, soft_tgt)):
                _ = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)

        acc, _ = evaluate(args, read_dataset(args, args.dev_path))

        if acc > best_acc:
            best_acc = acc
            config = {"learning_rate": learning_rate, "batch_size": batch_size, "epochs_num": epochs_num}
        args.logger.info("On configuration: {}.\n".format(config))

    args.logger.info("Best Acc. is: {:.4f}, on configuration {}.".format(best_acc, config))


if __name__ == "__main__":
    main()
