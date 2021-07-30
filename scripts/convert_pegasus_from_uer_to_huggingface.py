import sys
import os
import argparse
import collections
import torch

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, uer_dir)

from scripts.convert_bart_from_uer_to_huggingface import convert_encoder_decoder_transformer_from_uer_to_huggingface

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="models/input_model.bin",
                    help=".")
parser.add_argument("--output_model_path", type=str, default="models/output_model.bin",
                    help=".")
parser.add_argument("--layers_num", type=int, default=12, help=".")

args = parser.parse_args()

input_model = torch.load(args.input_model_path)

output_model = collections.OrderedDict()

output_model["model.shared.weight"] = input_model["embedding.word_embedding.weight"]
output_model["model.encoder.embed_positions.weight"] = input_model["embedding.pe"].squeeze(1)
output_model["model.decoder.embed_positions.weight"] = input_model["target.embedding.pe"].squeeze(1)
output_model["model.encoder.embed_tokens.weight"] = input_model["embedding.word_embedding.weight"]
output_model["model.decoder.embed_tokens.weight"] = input_model["embedding.word_embedding.weight"]
output_model["lm_head.weight"] = input_model["target.output_layer.weight"]
output_model["final_logits_bias"] = input_model["target.output_layer.bias"].unsqueeze(0)

convert_encoder_decoder_transformer_from_uer_to_huggingface(input_model, output_model, args.layers_num)

output_model["model.encoder.layer_norm.weight"] = input_model["encoder.layer_norm.gamma"]
output_model["model.encoder.layer_norm.bias"] = input_model["encoder.layer_norm.beta"]
output_model["model.decoder.layer_norm.weight"] = input_model["target.decoder.layer_norm.gamma"]
output_model["model.decoder.layer_norm.bias"] = input_model["target.decoder.layer_norm.beta"]

torch.save(output_model, args.output_model_path)
