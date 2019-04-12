# -*- encoding:utf-8 -*-
"""
  This script provides an exmaple to wrap bert-pytorch for cloze test.
  We randomly mask some characters and use BERT to predict.
"""
import sys
import torch
import argparse
import random
from bert.utils.act_fun import gelu
from bert.utils.constants import *
from bert.utils.tokenizer import *
from bert.layers.layer_norm import LayerNorm
from bert.utils.config import load_hyperparam
from bert.utils.vocab import Vocab
from bert.model_builder import build_model


class ClozeModel(torch.nn.Module):
    def __init__(self, args, bert_model):
        super(ClozeModel, self).__init__()
        self.embedding = bert_model.embedding
        self.encoder = bert_model.encoder
        self.target = bert_model.target
        # open eval mode
        self.eval()

    def forward(self, input_ids, seg_ids):
        emb = self.embedding(input_ids, seg_ids)
        # seq_length = emb.size(1)
        # # 
        # mask = (seg_ids>0).\
        #         unsqueeze(1).\
        #         repeat(1, seq_length, 1).\
        #         unsqueeze(1)

        # mask = mask.float()
        # mask = (1.0 - mask) * -10000.0
        output = self.encoder(emb, seg_ids)
        # mlm loss
        output = gelu(self.target.transform(output))
        output = self.target.transform_norm(output)
        # computing word prediction probability based on mlm task
        output = self.target.output_mlm(output)
        prob = torch.nn.Softmax(dim=-1)(output)
        return prob


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--model_path", default="./models/google_model.bin", type=str, 
                        help="Path of the pretrained model.")
    parser.add_argument("--vocab_path", default="./models/google_vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--input_path", type=str, default="datasets/cloze_input.txt", 
                        help="Path of the input file for cloze test. One sentence per line.")
    parser.add_argument("--output_path", type=str, default="./datasets/cloze_output.txt", 
                        help="Path of the output file for cloze test.")
    parser.add_argument("--config_path", default="./model.config", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=100,
                        help="Sequence length.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["char", "word", "space", "mixed"], default="char",
                        help="Specify the tokenizer." 
                             "Original Google BERT uses char tokenizer on Chinese corpus."
                             "Word tokenizer supports online word segmentation based on jieba segmentor."
                             "Space tokenizer segments sentences into words according to space."
                             "Mixed tokenizer segments sentences into words according to space."
                             "If words are not in the vocabulary, the tokenizer splits words into characters."
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

    # Build bert model.
    bert_model = build_model(args, len(vocab))

    # Load pretrained model.
    pretrained_model = torch.load(args.model_path)
    bert_model.load_state_dict(pretrained_model, strict=True)

    model = ClozeModel(args, bert_model)

    # Build tokenizer
    if args.tokenizer == "mixed":
        tokenizer = MixedTokenizer(vocab)
    else:
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
            topn_tokens = (-prob[i][p]).argsort()[:args.topn]

            sentence = "".join([vocab.i2w[token_id] for token_id in input_ids[j] if token_id != 0])
            pred_tokens = " ".join(vocab.i2w[token_id] for token_id in topn_tokens)
            label_token = vocab.i2w[label_ids_batch[j]]
            f_output.write(sentence + '\n')
            f_output.write("Predicted answer: " + pred_tokens + '\n')
            f_output.write("Correct answer: " + label_token + '\n')
            f_output.write("\n")
    
    f_output.close()
