import sys
import os
import torch
import argparse
import collections

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, uer_dir)

from scripts.convert_bert_from_huggingface_to_uer import convert_bert_transformer_encoder_from_huggingface_to_uer

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="input_model.bin",
                    help=".")
parser.add_argument("--output_model_path", type=str, default="output_model.bin",
                    help=".")
parser.add_argument("--layers_num", type=int, default=12, help=".")

args = parser.parse_args()

input_model = torch.load(args.input_model_path, map_location="cpu")

output_model = collections.OrderedDict()

output_model["embedding.word_embedding.weight"] = input_model["bert.embeddings.word_embeddings.weight"]
output_model["embedding.position_embedding.weight"] = input_model["bert.embeddings.position_embeddings.weight"]
output_model["embedding.segment_embedding.weight"] = torch.cat((torch.Tensor([[0]*input_model["bert.embeddings.token_type_embeddings.weight"].size()[1]]), input_model["bert.embeddings.token_type_embeddings.weight"]), dim=0)
output_model["embedding.layer_norm.gamma"] = input_model["bert.embeddings.LayerNorm.weight"]
output_model["embedding.layer_norm.beta"] = input_model["bert.embeddings.LayerNorm.bias"]

convert_bert_transformer_encoder_from_huggingface_to_uer(input_model, output_model, args.layers_num)

output_model["output_layer_1.weight"] = input_model["bert.pooler.dense.weight"]
output_model["output_layer_1.bias"] = input_model["bert.pooler.dense.bias"]
output_model["output_layer_2.weight"] = input_model["classifier.weight"]
output_model["output_layer_2.bias"] = input_model["classifier.bias"]

torch.save(output_model, args.output_model_path)
