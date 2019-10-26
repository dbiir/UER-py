# -*- encoding:utf-8 -*-
"""
  This script provides an exmaple to wrap UER-py for generate.
  We randomly give the beginning of story and use GPT to generate the full of story.
"""
import sys
import os
import torch
import torch.nn.functional as F
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


class GenerateModel(torch.nn.Module):
    def __init__(self, args, model):
        super(GenerateModel, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.target = model.target
        # Open eval mode.
        self.eval()

    def forward(self, src, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        output = gelu(self.target.output_layer(output))
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", type=str, required=True, 
                        help="Path of the pretrained model.")
    parser.add_argument("--vocab_path", default="models/google_vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--input_path", type=str, required=True, 
                        help="Path of the input file, containing the beginning of a story.")
    parser.add_argument("--output_path", type=str, required=True, 
                        help="Path of the output file, containing the entire story.")
    parser.add_argument("--config_path", default="models/bert_base_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--embedding", choices=["bert", "word"], default="bert",
                        help="Emebdding type.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                              "cnn", "gatedcnn", "attn", \
                                              "rcnn", "crnn", "gpt", "bilstm"], \
                                     default="bert", help="Encoder type.")
    parser.add_argument("--target", choices=["lm"], default="lm",
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
    pretrained_model_dict = torch.load(args.pretrained_model_path)
    model.load_state_dict(pretrained_model_dict, strict=False)

    model = GenerateModel(args, model)

    # Build tokenizer.
    tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)

    def top_k_top_p_filtering(logits, top_k, top_p):
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float("Inf")

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float("Inf")
        return logits

    with open(args.input_path, mode="r", encoding="utf-8") as f:
        line = f.readline().strip()
        src = [vocab.get(t) for t in tokenizer.tokenize(line.strip())]
        seg = [1] * len(src)
        start_length = len(src)
        if len(src) > args.seq_length:
            src = src[:args.seq_length]
            seg = seg[:args.seq_length]
    src = [src]
    seg = [seg]    
    src_tensor = torch.LongTensor(src)
    seg_tensor = torch.LongTensor(seg)

    f_output = open(args.output_path, mode="w", encoding="utf-8")

    for i in range(args.seq_length-start_length):
        outputs = model(src_tensor, seg_tensor)
        next_token_logits = outputs[0][-1] / args.temperature
        filtered_logits = top_k_top_p_filtering(next_token_logits, args.top_k, args.top_p)
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
        
        src_tensor = torch.cat([src_tensor, next_token.view(1,1)], dim=1)
        seg_tensor = torch.cat([seg_tensor, torch.tensor([[1]])], dim=1)

    f_output.write(line+"\n")
    generated_sentence = "".join([vocab.i2w[token_id] for token_id in src_tensor[0]])
    f_output.write(generated_sentence)
    
    f_output.close()
