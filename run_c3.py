"""
This script provides an exmaple to wrap UER-py for C3 (multiple choice dataset).
"""
import argparse
import json
import torch
import random
import torch.nn as nn
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils.tokenizer import *
from uer.layers.embeddings import *
from uer.encoders.bert_encoder import *
from uer.encoders.rnn_encoder import *
from uer.encoders.birnn_encoder import *
from uer.encoders.cnn_encoder import *
from uer.encoders.attn_encoder import *
from uer.encoders.gpt_encoder import *
from uer.encoders.mixed_encoder import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from run_classifier import build_optimizer, load_or_initialize_parameters, train_model, batch_loader, evaluate


class MultipleChoice(nn.Module):
    def __init__(self, args):
        super(MultipleChoice, self).__init__()
        self.embedding = globals()[args.embedding.capitalize() + "Embedding"](args, len(args.tokenizer.vocab))
        self.encoder = globals()[args.encoder.capitalize() + "Encoder"](args)
        self.dropout = nn.Dropout(args.dropout)
        self.output_layer = nn.Linear(args.hidden_size, 1)

    def forward(self, src, tgt, seg, soft_tgt=None):
        """
        Args:
            src: [batch_size x choices_num x seq_length]
            tgt: [batch_size]
            seg: [batch_size x choices_num x seq_length]
        """

        choices_num = src.shape[1]

        src = src.view(-1, src.size(-1))
        seg = seg.view(-1, seg.size(-1))

        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output = self.encoder(emb, seg)
        output = self.dropout(output)
        logits = self.output_layer(output[:, 0, :])
        reshaped_logits = logits.view(-1, choices_num)

        if tgt is not None:
            loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(reshaped_logits), tgt.view(-1))
            return loss, reshaped_logits
        else:
            return None, reshaped_logits


def read_dataset(args, path):

    with open(path, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for i in range(len(data)):
        for j in range(len(data[i][1])):
            example = ['\n'.join(data[i][0]).lower(), data[i][1][j]["question"].lower()]
            for k in range(len(data[i][1][j]["choice"])):
                example += [data[i][1][j]["choice"][k].lower()]
            for k in range(len(data[i][1][j]["choice"]), args.max_choices_num):
                example += ["No Answer"]

            example += [data[i][1][j].get("answer", "").lower()]

            examples += [example]

    dataset = []
    for i, example in enumerate(examples):
        tgt = 0
        for k in range(args.max_choices_num):
            if example[2 + k] == example[6]:
                tgt = k
        dataset.append(([], tgt, []))

        for k in range(args.max_choices_num):

            src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(example[k+2]) + [SEP_TOKEN])
            src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(example[1]) + [SEP_TOKEN])
            src_c = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(example[0]) + [SEP_TOKEN])

            src = src_a + src_b + src_c
            seg = [1] * (len(src_a) + len(src_b)) + [2] * len(src_c)

            if len(src) > args.seq_length:
                src = src[:args.seq_length]
                seg = seg[:args.seq_length]
            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)

            dataset[-1][0].append(src)
            dataset[-1][2].append(seg)

    return dataset


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="./models/multichoice_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default=None, type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--spm_model_path", default=None, type=str,
                        help="Path of the sentence piece model.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str, required=True,
                        help="Path of the devset.")
    parser.add_argument("--test_path", type=str,
                        help="Path of the testset.")
    parser.add_argument("--config_path", default="./models/bert_base_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=512,
                        help="Sequence length.")
    parser.add_argument("--embedding", choices=["bert", "word"], default="bert",
                        help="Emebdding type.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                              "cnn", "gatedcnn", "attn", "synt", \
                                              "rcnn", "crnn", "gpt", "bilstm"], \
                                              default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--factorized_embedding_parameterization", action="store_true", help="Factorized embedding parameterization.")
    parser.add_argument("--parameter_sharing", action="store_true", help="Parameter sharing.")
    parser.add_argument("--max_choices_num", default=4, type=int,
                        help="The maximum number of cadicate answer, shorter than this will be padded.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                        )

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")
    parser.add_argument("--fp16", action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit.")
    parser.add_argument("--fp16_opt_level", choices=["O0", "O1", "O2", "O3" ], default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=8,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    args = parser.parse_args()
    args.labels_num = args.max_choices_num

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Build tokenizer.
    args.tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)

    # Build multiple choice model.
    model = MultipleChoice(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)

    # Training phase.
    trainset = read_dataset(args, args.train_path)
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    src = torch.LongTensor([example[0] for example in trainset])
    tgt = torch.LongTensor([example[1] for example in trainset])
    seg = torch.LongTensor([example[2] for example in trainset])

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
    args.model = model

    total_loss, result, best_result = 0., 0., 0.

    print("Start training.")

    for epoch in range(1, args.epochs_num+1):
        model.train()
        for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):

            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch)
            total_loss += loss.item()

            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i+1, total_loss / args.report_steps))
                total_loss = 0.

        result = evaluate(args, read_dataset(args, args.dev_path))
        if result[0] > best_result:
            best_result = result[0]
            save_model(model, args.output_model_path)

    # Evaluation phase.
    if args.test_path is not None:
        print("Test set evaluation.")
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(args.output_model_path))
        else:
            model.load_state_dict(torch.load(args.output_model_path))
        evaluate(args, read_dataset(args, args.test_path))


if __name__ == "__main__":
    main()
