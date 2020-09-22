"""
  This script provides an example to wrap UER-py for Chinese machine reading comprehension inference.
"""
import sys
import os
import random
import argparse
import json
import torch
import torch.nn as nn

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.utils.config import load_hyperparam
from uer.utils.constants import *
from uer.utils.tokenizer import * 
from uer.utils.vocab import Vocab
from uer.model_loader import load_model
from run_cmrc import *


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
    parser.add_argument("--prediction_path", default=None, type=str,
                        help="Path of the prediction file.")
    parser.add_argument("--config_path", default="./models/bert_base_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=512,
                        help="Sequence length.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--embedding", choices=["bert", "word"], default="bert",
                        help="Emebdding type.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                              "cnn", "gatedcnn", "attn", "synt", \
                                              "rcnn", "crnn", "gpt", "bilstm"], \
                                              default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--factorized_embedding_parameterization", action="store_true", help="Factorized embedding parameterization.")
    parser.add_argument("--parameter_sharing", action="store_true", help="Parameter sharing.")


    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # Build tokenizer.
    args.tokenizer = CharTokenizer(args)

    # Build model and load parameters.
    model = MachineReadingComprehension(args)
    model = load_model(model, args.load_model_path)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    dataset, examples = read_dataset(args, args.test_path)

    src = torch.LongTensor([sample[0] for sample in dataset])
    seg = torch.LongTensor([sample[1] for sample in dataset])
    start_position = torch.LongTensor([sample[2] for sample in dataset])
    end_position = torch.LongTensor([sample[3] for sample in dataset])

    batch_size = args.batch_size
    instances_num = len(dataset)

    print("The number of prediction instances: ", instances_num)

    model.eval()

    with open(args.prediction_path, mode="w", encoding="utf-8") as f:

        start_prob_all, end_prob_all = [], []

        for i, (src_batch, seg_batch, start_position_batch, end_position_batch) in enumerate(batch_loader(batch_size, src, seg, start_position, end_position)):
            src_batch = src_batch.to(device)
            seg_batch = seg_batch.to(device)
            start_position_batch = start_position_batch.to(device)
            end_position_batch = end_position_batch.to(device)

            with torch.no_grad():
                loss, start_logits, end_logits = model(src_batch, seg_batch, start_position_batch, end_position_batch)

            start_prob = nn.Softmax(dim=1)(start_logits)
            end_prob = nn.Softmax(dim=1)(end_logits)

            for j in range(start_prob.size()[0]):
                start_prob_all.append(start_prob[j])
                end_prob_all.append(end_prob[j])

        pred_answers = get_answers(dataset, start_prob_all, end_prob_all)

        output = {}
        for i in range(len(examples)):
            question_id = examples[i][2]
            start_pred_pos = pred_answers[i][1]
            end_pred_pos = pred_answers[i][2]

            prediction = examples[i][0][start_pred_pos:end_pred_pos]
            output[question_id] = prediction

        f.write(json.dumps(output, indent=4, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
