"""
This script provides an example to wrap UER-py for classification with cross validation.
"""
import sys
import os
import random
import argparse
import torch.nn as nn
import numpy as np

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
from finetune.run_classifier import *


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="models/classifier_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default=None, type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--spm_model_path", default=None, type=str,
                        help="Path of the sentence piece model.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")
    parser.add_argument("--config_path", default="models/bert/base_config.json", type=str,
                        help="Path of the config file.")
    parser.add_argument("--train_features_path", type=str, required=True,
                        help="Path of the train features for stacking.")

    # Model options.
    model_opts(parser)
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    # Tokenizer options.
    tokenizer_opts(parser)

    # Optimization options.
    optimization_opts(parser) 
    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")

    # Training options.
    training_opts(parser)

    # Cross validation options.
    parser.add_argument("--folds_num", type=int, default=5,
                        help="The number of folds for cross validation.")
    
    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    args.labels_num = count_labels_num(args.train_path)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Training phase.
    dataset = read_dataset(args, args.train_path)
    instances_num = len(dataset)
    batch_size = args.batch_size
    instances_num_per_fold = instances_num // args.folds_num + 1

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    train_features = []

    total_loss, result = 0.0, 0.0
    acc, marco_f1 = 0.0, 0.0

    for fold_id in range(args.folds_num):
        torch.cuda.empty_cache()

        # Build classification model.
        model = Classifier(args)

        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(args.device)
        load_or_initialize_parameters(args, model)
        optimizer, scheduler = build_optimizer(args, model)
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
            args.amp = amp
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        args.model = model

        trainset = dataset[0 : fold_id * instances_num_per_fold] + dataset[(fold_id + 1) * instances_num_per_fold :]
        random.shuffle(trainset)

        train_src = torch.LongTensor([example[0] for example in trainset])
        train_tgt = torch.LongTensor([example[1] for example in trainset])
        train_seg = torch.LongTensor([example[2] for example in trainset])

        if args.soft_targets:
            train_soft_tgt = torch.FloatTensor([example[3] for example in trainset])
        else:
            train_soft_tgt = None

        devset = dataset[fold_id * instances_num_per_fold : (fold_id + 1) * instances_num_per_fold]

        dev_src = torch.LongTensor([example[0] for example in devset])
        dev_tgt = torch.LongTensor([example[1] for example in devset])
        dev_seg = torch.LongTensor([example[2] for example in devset])
        dev_soft_tgt = None

        for epoch in range(1, args.epochs_num + 1):
            model.train()
            for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, train_src, train_tgt, train_seg, train_soft_tgt)):
                loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
                total_loss += loss.item()
                if (i + 1) % args.report_steps == 0:
                    print("Fold id: {}, Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(fold_id, epoch, i + 1, total_loss / args.report_steps))
                    total_loss = 0.0

        model.eval()
        for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, dev_src, dev_tgt, dev_seg, dev_soft_tgt)): 
            src_batch = src_batch.to(args.device)
            seg_batch = seg_batch.to(args.device)
            with torch.no_grad():
                _, logits = model(src_batch, None, seg_batch)
            prob = nn.Softmax(dim=1)(logits)
            prob = prob.cpu().numpy().tolist()
            train_features.extend(prob)

        output_model_name = ".".join(args.output_model_path.split(".")[:-1])
        output_model_suffix = args.output_model_path.split(".")[-1]
        save_model(model, output_model_name + "-fold_" + str(fold_id) + "." + output_model_suffix)
        result = evaluate(args, devset)
        acc += result[0] / args.folds_num
        f1 = []
        confusion = result[1]
        eps = 1e-9
        for i in range(confusion.size()[0]):
            p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
            r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
            f1.append(2 * p * r / (p + r + eps))

        marco_f1 += sum(f1) / len(f1) / args.folds_num

    train_features = np.array(train_features)
    np.save(args.train_features_path, train_features)
    print("Acc. : {:.4f}".format(acc))
    print("Marco F1 : {:.4f}".format(marco_f1))


if __name__ == "__main__":
    main()
