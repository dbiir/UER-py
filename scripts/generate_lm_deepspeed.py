"""
  This script provides an example to use DeepSpeed for generation.
  Given the beginning of a text, language model generates the rest.
"""
import sys
import os
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
import deepspeed

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.opts import deepspeed_opts
from scripts.generate_lm import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    parser.add_argument("--top_k", type=int, default=70)
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)

    tokenizer_opts(parser)

    deepspeed_opts(parser)
    parser.add_argument("--mp_size", type=int, default=1, help="Model parallel size.")

    args = parser.parse_args()

    args.target = "lm"
    args.batch_size = 1

    args = load_hyperparam(args)

    args.tokenizer = str2tokenizer[args.tokenizer](args)

    model = GenerateLm(args)
    model = load_model(model, args.load_model_path)
    deepspeed.init_distributed()
    model = deepspeed.init_inference(model=model, mp_size=args.mp_size, replace_method=None)

    rank = dist.get_rank()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if rank == 0:
        model.eval()

        with open(args.test_path, mode="r", encoding="utf-8") as f:
            line = f.readline().strip()
            src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(line))
            seg = [1] * len(src)
            beginning_length = len(src)
            if len(src) > args.seq_length:
                src = src[:args.seq_length]
                seg = seg[:args.seq_length]
        src_tensor, seg_tensor = torch.LongTensor([src]).to(device), torch.LongTensor([seg]).to(device)

        with open(args.prediction_path, mode="w", encoding="utf-8") as f:
            for i in range(args.seq_length - beginning_length):
                output = model(src_tensor, seg_tensor)
                next_token_logits = output[0][-1] / args.temperature
                filtered_logits = top_k_top_p_filtering(next_token_logits, args.top_k, args.top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

                src_tensor = torch.cat([src_tensor, next_token.view(1, 1).to(device)], dim=1)
                seg_tensor = torch.cat([seg_tensor, torch.tensor([[1]]).to(device)], dim=1)

            f.write(line + "\n")
            generated_sentence = "".join(
                args.tokenizer.convert_ids_to_tokens([token_id.item() for token_id in src_tensor[0]])
            )
            f.write(generated_sentence)
