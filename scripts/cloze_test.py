# -*- encoding:utf-8 -*-
"""
  This script provides an exmaple to wrap UER-py for cloze test.
  We randomly mask some characters and use BERT to predict.
"""
import os
import sys
import torch
import argparse
import random

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.utils.act_fun import gelu
from uer.utils.constants import *
from uer.utils.tokenizer import *
from uer.layers.layer_norm import LayerNorm
from uer.utils.config import load_hyperparam
from uer.utils.vocab import Vocab
from uer.model_builder import build_model


class ClozeModel(torch.nn.Module):
    def __init__(self, args, model):
        super(ClozeModel, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.target = model.target
        # Open eval mode.
        self.eval()

    def forward(self, src, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        output = gelu(self.target.mlm_linear_1(output))
        output = self.target.layer_norm(output)
        output = self.target.mlm_linear_2(output)
        prob = torch.nn.Softmax(dim=-1)(output)
        return prob


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default="models/google_model.bin", type=str, 
                        help="Path of the pretrained model.")
    parser.add_argument("--vocab_path", type=str, required=True,
                        help="Path of the vocabulary file.")
    parser.add_argument("--input_path", type=str, default="datasets/cloze_input.txt", 
                        help="Path of the input file for cloze test. One sentence per line.")
    parser.add_argument("--output_path", type=str, default="datasets/cloze_output.txt", 
                        help="Path of the output file for cloze test.")
    parser.add_argument("--config_path", default="models/bert_base_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=100,
                        help="Sequence length.")
    parser.add_argument("--embedding", choices=["bert", "word"], default="bert",
                        help="Emebdding type.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                                   "cnn", "gatedcnn", "attn", \
                                                   "rcnn", "crnn", "gpt"], \
                                                   default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--target", choices=["bert", "mlm"], default="bert",
                        help="The training target of the pretraining model.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder_type", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )

    # Output options.
    parser.add_argument("--topn", type=int, default=10,
                        help="Print top n nearest neighbours.")
    
    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # Load Vocabulary
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build bert model.
    model = build_model(args)

    # Load pretrained model.
    pretrained_model = torch.load(args.pretrained_model_path)
    model.load_state_dict(pretrained_model, strict=False)

    model = ClozeModel(args, model)

    # Build tokenizer
    tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)

    # Construct input datasets.
    def mask_token(tokens):
        """
        Mask a random token for prediction.
        """
        start = 1
        end = len(tokens) if len(tokens) < args.seq_length else args.seq_length
        mask_pos = random.randint(start, end-1)
        token = tokens[mask_pos]
        tokens[mask_pos] = MASK_ID
        return (tokens, mask_pos, token)

    input_ids = []
    seg_ids = []
    mask_positions = [] # The position of the masked word.
    label_ids = [] # The id of the masked word.

    with open(args.input_path, mode="r", encoding="utf-8") as f:
        for line in f:        
            tokens = [vocab.get(t) for t in tokenizer.tokenize(line.strip())]
            if len(tokens) == 0:
                continue
            tokens = [CLS_ID] + tokens
            tokens, mask_pos, label = mask_token(tokens)

            seg = [1] * len(tokens)
            if len(tokens) > args.seq_length:
                tokens = tokens[:args.seq_length]
                seg = seg[:args.seq_length]
            while len(tokens) < args.seq_length:
                tokens.append(PAD_ID)
                seg.append(PAD_ID)
            input_ids.append(tokens)
            seg_ids.append(seg)

            mask_positions.append(mask_pos)
            label_ids.append(label)

    input_ids = torch.LongTensor(input_ids)
    seg_ids = torch.LongTensor(seg_ids)

    def batch_loader(batch_size, input_ids, seg_ids, mask_positions, label_ids):
        instances_num = input_ids.size(0)
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i*batch_size : (i+1)*batch_size]
            seg_ids_batch = seg_ids[i*batch_size : (i+1)*batch_size]
            mask_positions_batch = mask_positions[i*batch_size : (i+1)*batch_size]
            label_ids_batch = label_ids[i*batch_size : (i+1)*batch_size]
            yield input_ids_batch, seg_ids_batch, mask_positions_batch, label_ids_batch

        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num//batch_size*batch_size:]
            seg_ids_batch = seg_ids[instances_num//batch_size*batch_size:]
            mask_positions_batch = mask_positions[instances_num//batch_size*batch_size:]
            label_ids_batch = label_ids[instances_num//batch_size*batch_size:]
            yield input_ids_batch, seg_ids_batch, mask_positions_batch, label_ids_batch

    f_output = open(args.output_path, mode="w", encoding="utf-8")
               
    for i, (input_ids_batch, seg_ids_batch, mask_positions_batch, label_ids_batch) in \
        enumerate(batch_loader(args.batch_size, input_ids, seg_ids, mask_positions, label_ids)):
        prob = model(input_ids_batch, seg_ids_batch)

        for j, p in enumerate(mask_positions_batch):
            topn_tokens = (-prob[j][p]).argsort()[:args.topn]

            sentence = "".join([vocab.i2w[token_id] for token_id in input_ids_batch[j] if token_id != 0])
            pred_tokens = " ".join(vocab.i2w[token_id] for token_id in topn_tokens)
            label_token = vocab.i2w[label_ids_batch[j]]
            f_output.write(sentence + '\n')
            f_output.write("Predicted answer: " + pred_tokens + '\n')
            f_output.write("Correct answer: " + label_token + '\n')
            f_output.write("\n")
    
    f_output.close()
