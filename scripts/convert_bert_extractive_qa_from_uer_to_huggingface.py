import torch
import argparse
import collections

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="robert_extractive_qa_model.bin",
                    help=".")
parser.add_argument("--output_model_path", type=str, default="pytorch_model.bin",
                    help=".")
parser.add_argument("--layers_num", type=int, default=12, help=".")

args = parser.parse_args()
path = args.input_model_path

input_model = torch.load(args.input_model_path)

output_model = collections.OrderedDict()

output_model["bert.embeddings.word_embeddings.weight"] = input_model["embedding.word_embedding.weight"]
output_model["bert.embeddings.position_embeddings.weight"] = input_model["embedding.position_embedding.weight"]
output_model["bert.embeddings.token_type_embeddings.weight"] = input_model["embedding.segment_embedding.weight"][1:, :]
output_model["bert.embeddings.LayerNorm.weight"] = input_model["embedding.layer_norm.gamma"]
output_model["bert.embeddings.LayerNorm.bias"] = input_model["embedding.layer_norm.beta"]

for i in range(args.layers_num):
    output_model["bert.encoder.layer." + str(i) + ".attention.self.query.weight"] = input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.weight"]
    output_model["bert.encoder.layer." + str(i) + ".attention.self.query.bias"] = input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.bias"]
    output_model["bert.encoder.layer." + str(i) + ".attention.self.key.weight"] = input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.weight"]
    output_model["bert.encoder.layer." + str(i) + ".attention.self.key.bias"] = input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.bias"]
    output_model["bert.encoder.layer." + str(i) + ".attention.self.value.weight"] = input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.weight"]
    output_model["bert.encoder.layer." + str(i) + ".attention.self.value.bias"] = input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.bias"]
    output_model["bert.encoder.layer." + str(i) + ".attention.output.dense.weight"] = input_model["encoder.transformer." + str(i) + ".self_attn.final_linear.weight"]
    output_model["bert.encoder.layer." + str(i) + ".attention.output.dense.bias"] = input_model["encoder.transformer." + str(i) + ".self_attn.final_linear.bias"]
    output_model["bert.encoder.layer." + str(i) + ".attention.output.LayerNorm.weight"] = input_model["encoder.transformer." + str(i) + ".layer_norm_1.gamma"]
    output_model["bert.encoder.layer." + str(i) + ".attention.output.LayerNorm.bias"] = input_model["encoder.transformer." + str(i) + ".layer_norm_1.beta"]
    output_model["bert.encoder.layer." + str(i) + ".intermediate.dense.weight"] = input_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"]
    output_model["bert.encoder.layer." + str(i) + ".intermediate.dense.bias"] = input_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.bias"]
    output_model["bert.encoder.layer." + str(i) + ".output.dense.weight"] = input_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"]
    output_model["bert.encoder.layer." + str(i) + ".output.dense.bias"] = input_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.bias"]
    output_model["bert.encoder.layer." + str(i) + ".output.LayerNorm.weight"] = input_model["encoder.transformer." + str(i) + ".layer_norm_2.gamma"]
    output_model["bert.encoder.layer." + str(i) + ".output.LayerNorm.bias"] = input_model["encoder.transformer." + str(i) + ".layer_norm_2.beta"]

output_model["qa_outputs.weight"] = input_model["output_layer.weight"]
output_model["qa_outputs.bias"] = input_model["output_layer.bias"]


torch.save(output_model, args.output_model_path)
