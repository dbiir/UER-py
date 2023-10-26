import sys
import os
import argparse
import collections
import torch
import math

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, uer_dir)

from scripts.convert_bart_from_uer_to_huggingface import \
    convert_encoder_decoder_transformer_from_uer_to_huggingface

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="models/input_model.bin",
                    help=".")
parser.add_argument("--output_model_path", type=str, default="models/output_model.bin",
                    help=".")
parser.add_argument("--layers_num", type=int, default=12, help=".")
parser.add_argument("--decoder_layers_num", type=int, default=12, help=".")
parser.add_argument("--max_seq_length", type=int, default=1024, help=".")

args = parser.parse_args()

input_model = torch.load(args.input_model_path)

output_model = collections.OrderedDict()

output_model["model.shared.weight"] = input_model["embedding.word.embedding.weight"]

emb_size = input_model["embedding.word.embedding.weight"].shape[1]
pe = torch.zeros(args.max_seq_length, emb_size)
position = torch.arange(0, args.max_seq_length).unsqueeze(1)
div_term = torch.exp(
    (
        torch.arange(0, emb_size, 2, dtype=torch.float)
        *- (math.log(10000.0) / emb_size)
    )
)
pe[:, 0::2] = torch.sin(position.float() * div_term)
pe[:, 1::2] = torch.cos(position.float() * div_term)

output_model["model.encoder.embed_positions.weight"] = pe
output_model["model.decoder.embed_positions.weight"] = pe
output_model["model.encoder.embed_tokens.weight"] = input_model["embedding.word.embedding.weight"]
output_model["model.decoder.embed_tokens.weight"] = input_model["tgt_embedding.word.embedding.weight"]
output_model["lm_head.weight"] = input_model["target.lm.output_layer.weight"]
output_model["final_logits_bias"] = input_model["target.lm.output_layer.bias"].unsqueeze(0)

convert_encoder_decoder_transformer_from_uer_to_huggingface(input_model, output_model, args.layers_num, args.decoder_layers_num)

output_model["model.encoder.layer_norm.weight"] = input_model["encoder.layer_norm.gamma"]
output_model["model.encoder.layer_norm.bias"] = input_model["encoder.layer_norm.beta"]
output_model["model.decoder.layer_norm.weight"] = input_model["decoder.layer_norm.gamma"]
output_model["model.decoder.layer_norm.bias"] = input_model["decoder.layer_norm.beta"]

torch.save(output_model, args.output_model_path)
