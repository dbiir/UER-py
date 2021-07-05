"""
  This script provides an exmaple to wrap UER-py for generation.
  Given the beginning of a text, language model generates the rest.
"""
import sys
import os
import torch
import torch.nn.functional as F
import argparse
import random

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 
sys.path.append(uer_dir)

from uer.layers import *
from uer.encoders import *
from uer.targets import *
from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.model_loader import load_model
from uer.opts import infer_opts, tokenizer_opts


class GenerateLm(torch.nn.Module):
    def __init__(self, args):
        super(GenerateLm, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.target = str2target[args.target](args, len(args.tokenizer.vocab))

    def forward(self, src, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        output = self.target.output_layer(output)
        return output


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    parser.add_argument("--target", choices=["lm"], default="lm",
                        help="The training target of the pretraining model.")
    parser.add_argument("--has_lmtarget_bias", action="store_true",
                        help="Add bias on output_layer for lm target.")
    parser.add_argument("--tie_weights", action="store_true",
                        help="Tie the word embedding and softmax weights.")
    parser.add_argument("--top_k", type=int, default=70)
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
 
    tokenizer_opts(parser)
    
    args = parser.parse_args()

    args.batch_size = 1

    args = load_hyperparam(args)

    args.tokenizer = str2tokenizer[args.tokenizer](args)

    model = GenerateLm(args)
    model = load_model(model, args.load_model_path)
    model.eval()

    with open(args.test_path, mode="r", encoding="utf-8") as f:
        line = f.readline().strip()
        src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(line))
        seg = [1] * len(src)
        beginning_length = len(src)
        if len(src) > args.seq_length:
            src = src[:args.seq_length]
            seg = seg[:args.seq_length]
    src_tensor, seg_tensor = torch.LongTensor([src]), torch.LongTensor([seg])

    with open(args.prediction_path, mode="w", encoding="utf-8") as f:
        for i in range(args.seq_length - beginning_length):
            output = model(src_tensor, seg_tensor)
            next_token_logits = output[0][-1] / args.temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, args.top_k, args.top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
        
            src_tensor = torch.cat([src_tensor, next_token.view(1, 1)], dim=1)
            seg_tensor = torch.cat([seg_tensor, torch.tensor([[1]])], dim=1)

        f.write(line + "\n")
        generated_sentence = "".join(args.tokenizer.convert_ids_to_tokens([token_id.item() for token_id in src_tensor[0]]))
        f.write(generated_sentence)
