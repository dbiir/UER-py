import sys
import os
import argparse
import collections
import torch


uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, uer_dir)

from scripts.convert_bart_from_huggingface_to_uer import convert_encoder_decoder_transformer_from_huggingface_to_uer

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="models/input_model.bin",
                    help=".")
parser.add_argument("--output_model_path", type=str, default="models/output_model.bin",
                    help=".")
parser.add_argument("--layers_num", type=int, default=12, help=".")

args = parser.parse_args()

input_model = torch.load(args.input_model_path, map_location="cpu")

output_model = collections.OrderedDict()

output_model["embedding.pe"] = input_model["model.encoder.embed_positions.weight"].unsqueeze(1)
output_model["target.embedding.pe"] = input_model["model.decoder.embed_positions.weight"].unsqueeze(1)
output_model["embedding.word_embedding.weight"] = input_model["model.encoder.embed_tokens.weight"]
output_model["target.embedding.word_embedding.weight"] = input_model["model.decoder.embed_tokens.weight"]
output_model["target.output_layer.weight"] = input_model["lm_head.weight"]
output_model["target.output_layer.bias"] = input_model["final_logits_bias"].squeeze(0)

convert_encoder_decoder_transformer_from_huggingface_to_uer(input_model, output_model, args.layers_num)

output_model["encoder.layer_norm.gamma"] = input_model["model.encoder.layer_norm.weight"]
output_model["encoder.layer_norm.beta"] = input_model["model.encoder.layer_norm.bias"]
output_model["target.decoder.layer_norm.gamma"] = input_model["model.decoder.layer_norm.weight"]
output_model["target.decoder.layer_norm.beta"] = input_model["model.decoder.layer_norm.bias"]

torch.save(output_model, args.output_model_path)
