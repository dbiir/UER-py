import sys
import os
import torch
import torch.nn.functional as F
import argparse
import random

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, uer_dir)

from uer.layers import *
from uer.encoders import *
from uer.targets import *
from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.model_loader import load_model
from uer.opts import infer_opts, tokenizer_opts
from scripts.generate_lm import top_k_top_p_filtering


class GenerateSeq2seq(torch.nn.Module):
    def __init__(self, args):
        super(GenerateSeq2seq, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.target = str2target[args.target](args, len(args.tgt_tokenizer.vocab))

    def forward(self, src, seg, tgt):
        emb = self.embedding(src, seg)
        memory_bank = self.encoder(emb, seg)
        emb = self.target.embedding(tgt, None)
        hidden = self.target.decoder(memory_bank, emb, (src,))
        output = self.target.output_layer(hidden)
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    parser.add_argument("--target", choices=["seq2seq", "t5"], default="t5",
                        help="The training target of the pretraining model.")
    parser.add_argument("--share_relative_position_embedding", action="store_true",
                        help="Add bias on output_layer for lm target.")
    parser.add_argument("--has_lmtarget_bias", action="store_true",
                        help="Add bias on output_layer for lm target.")
    parser.add_argument("--tie_weights", action="store_true",
                        help="Tie the word embedding and softmax weights.")
    parser.add_argument("--top_k", type=int, default=70)
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tgt_vocab_path", type=str,
                        help="Path of the vocabulary file.")
    tokenizer_opts(parser)
    parser.add_argument("--tgt_tokenizer", choices=["bert", "char", "space", "xlmroberta"], default="bert",
                        help="Specify the tokenizer for target side.")
    parser.add_argument("--tgt_seq_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--tgt_embedding", choices=["word", "word_pos", "word_pos_seg", "word_sinusoidalpos"], default="word",
                        help="Target embedding type.")
    parser.add_argument("--decoder", choices=["transformer"], \
                                              default="transformer", help="Decoder type.")
    args = parser.parse_args()

    args.batch_size = 1

    args = load_hyperparam(args)

    args.tokenizer = str2tokenizer[args.tokenizer](args)

    if args.target == "seq2seq":
        args.vocab_path = args.tgt_vocab_path
        args.tgt_tokenizer = str2tokenizer[args.tgt_tokenizer](args)
        args.tgt_vocab = args.tgt_tokenizer.vocab
    else:
        args.tgt_tokenizer = args.tokenizer

    model = GenerateSeq2seq(args)
    model = load_model(model, args.load_model_path)
    model.eval()

    with open(args.test_path, mode="r", encoding="utf-8") as f:
        line = f.readline().strip()
        src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(line) + [SEP_TOKEN])
        seg = [1] * len(src)
        tgt = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN])
        beginning_length = len(src)
        if len(src) > args.seq_length:
            src = src[:args.seq_length]
            seg = seg[:args.seq_length]
    src_tensor, seg_tensor, tgt_tensor = torch.LongTensor([src]), torch.LongTensor([seg]), torch.LongTensor([tgt])

    with open(args.prediction_path, mode="w", encoding="utf-8") as f:
        for i in range(args.tgt_seq_length-1):
            output = model(src_tensor, seg_tensor, tgt_tensor)
            next_token_logits = output[0][-1] / args.temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, args.top_k, args.top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            tgt_tensor = torch.cat([tgt_tensor, next_token.view(1, 1)], dim=1)

        f.write(line + "\n")
        generated_sentence = "".join(args.tgt_tokenizer.convert_ids_to_tokens([token_id.item() for token_id in tgt_tensor[0]]))
        f.write(generated_sentence)
