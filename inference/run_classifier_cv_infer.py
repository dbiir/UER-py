"""
  This script provides an example to wrap UER-py for classification inference (cross validation).
"""
import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import * 
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_loader import load_model
from uer.opts import *
from finetune.run_classifier import Classifier
from inference.run_classifier_infer import *


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path of the classfier model.")
    parser.add_argument("--vocab_path", default=None, type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--spm_model_path", default=None, type=str,
                        help="Path of the sentence piece model.")
    parser.add_argument("--test_path", type=str,
                        help="Path of the testset.")
    parser.add_argument("--test_features_path", default=None, type=str,
                        help="Path of the test features for stacking.")
    parser.add_argument("--config_path", default="models/bert/base_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    model_opts(parser)
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    # Inference options.
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--labels_num", type=int, required=True,
                        help="Number of prediction labels.")

    # Tokenizer options.
    tokenizer_opts(parser)

    # Output options.
    parser.add_argument("--output_logits", action="store_true", help="Write logits to output file.")
    parser.add_argument("--output_prob", action="store_true", help="Write probabilities to output file.")

    # Cross validation options.
    parser.add_argument("--folds_num", type=int, default=5,
                        help="The number of folds for cross validation.")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model and load parameters.
    args.soft_targets, args.soft_alpha = False, False

    dataset = read_dataset(args, args.test_path)

    src = torch.LongTensor([sample[0] for sample in dataset])
    seg = torch.LongTensor([sample[1] for sample in dataset])

    batch_size = args.batch_size
    instances_num = src.size()[0]

    print("The number of prediction instances: ", instances_num)

    test_features = [[] for _ in range(args.folds_num)]
    for fold_id in range(args.folds_num):
        load_model_name = ".".join(args.load_model_path.split(".")[:-1])
        load_model_suffix = args.load_model_path.split(".")[-1]

        model = Classifier(args)
        model = load_model(model, load_model_name+"-fold_"+str(fold_id)+"."+load_model_suffix)

        # For simplicity, we use DataParallel wrapper to use multiple GPUs.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model)

        model.eval()
        for i, (src_batch, seg_batch) in enumerate(batch_loader(batch_size, src, seg)):
            src_batch = src_batch.to(device)
            seg_batch = seg_batch.to(device)
            with torch.no_grad():
                _, logits = model(src_batch, None, seg_batch)

            prob = nn.Softmax(dim=1)(logits)
            prob = prob.cpu().numpy().tolist()
            test_features[fold_id].extend(prob)

    test_features = np.array(test_features)
    test_features = np.mean(test_features, axis=0)
    np.save(args.test_features_path, test_features)


if __name__ == "__main__":
    main()
