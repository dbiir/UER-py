import sys
import os
import argparse
import collections
import torch

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, uer_dir)

from scripts.convert_bert_from_uer_to_huggingface import convert_bert_transformer_encoder_from_uer_to_huggingface

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="models/input_model.bin",
                    help=".")
parser.add_argument("--output_model_path", type=str, default="models/output_model.bin",
                    help=".")
parser.add_argument("--layers_num", type=int, default=12, help=".")

args = parser.parse_args()

input_model = torch.load(args.input_model_path)

output_model = collections.OrderedDict()

output_model["bert.embeddings.word_embeddings.weight"] = input_model["embedding.word_embedding.weight"]
output_model["bert.embeddings.position_embeddings.weight"] = input_model["embedding.position_embedding.weight"]
output_model["bert.embeddings.token_type_embeddings.weight"] = input_model["embedding.segment_embedding.weight"][1:, :]
output_model["bert.embeddings.LayerNorm.weight"] = input_model["embedding.layer_norm.gamma"]
output_model["bert.embeddings.LayerNorm.bias"] = input_model["embedding.layer_norm.beta"]

convert_bert_transformer_encoder_from_uer_to_huggingface(input_model, output_model, args.layers_num)

output_model["qa_outputs.weight"] = input_model["output_layer.weight"]
output_model["qa_outputs.bias"] = input_model["output_layer.bias"]

torch.save(output_model, args.output_model_path)
