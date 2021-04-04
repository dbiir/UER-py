import torch
import argparse
import collections

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="pytorch_model.bin",
                    help=".")
parser.add_argument("--output_model_path", type=str, default="robert_extractive_qa_model.bin",
                    help=".")
parser.add_argument("--layers_num", type=int, default=12, help=".")

args = parser.parse_args()
path = args.input_model_path

input_model = torch.load(args.input_model_path, map_location='cpu')

output_model = collections.OrderedDict()

output_model["embedding.word_embedding.weight"] = input_model["bert.embeddings.word_embeddings.weight"]
output_model["embedding.position_embedding.weight"] = input_model["bert.embeddings.position_embeddings.weight"]
output_model["embedding.segment_embedding.weight"] = torch.cat((torch.Tensor([[0]*input_model["bert.embeddings.token_type_embeddings.weight"].size()[1]]), input_model["bert.embeddings.token_type_embeddings.weight"]), dim=0)
output_model["embedding.layer_norm.gamma"] = input_model["bert.embeddings.LayerNorm.weight"]
output_model["embedding.layer_norm.beta"] = input_model["bert.embeddings.LayerNorm.bias"]

for i in range(args.layers_num):
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.weight"] = input_model["bert.encoder.layer." + str(i) + ".attention.self.query.weight"]
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.bias"] = input_model["bert.encoder.layer." + str(i) + ".attention.self.query.bias"]
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.weight"] = input_model["bert.encoder.layer." + str(i) + ".attention.self.key.weight"]
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.bias"] = input_model["bert.encoder.layer." + str(i) + ".attention.self.key.bias"]
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.weight"] = input_model["bert.encoder.layer." + str(i) + ".attention.self.value.weight"]
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.bias"] = input_model["bert.encoder.layer." + str(i) + ".attention.self.value.bias"]
    output_model["encoder.transformer." + str(i) + ".self_attn.final_linear.weight"] = input_model["bert.encoder.layer." + str(i) + ".attention.output.dense.weight"]
    output_model["encoder.transformer." + str(i) + ".self_attn.final_linear.bias"] = input_model["bert.encoder.layer." + str(i) + ".attention.output.dense.bias"]
    output_model["encoder.transformer." + str(i) + ".layer_norm_1.gamma"] = input_model["bert.encoder.layer." + str(i) + ".attention.output.LayerNorm.weight"]
    output_model["encoder.transformer." + str(i) + ".layer_norm_1.beta"] = input_model["bert.encoder.layer." + str(i) + ".attention.output.LayerNorm.bias"]
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"] = input_model["bert.encoder.layer." + str(i) + ".intermediate.dense.weight"]
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.bias"] = input_model["bert.encoder.layer." + str(i) + ".intermediate.dense.bias"]
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"] = input_model["bert.encoder.layer." + str(i) + ".output.dense.weight"]
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.bias"] = input_model["bert.encoder.layer." + str(i) + ".output.dense.bias"]
    output_model["encoder.transformer." + str(i) + ".layer_norm_2.gamma"] = input_model["bert.encoder.layer." + str(i) + ".output.LayerNorm.weight"]
    output_model["encoder.transformer." + str(i) + ".layer_norm_2.beta"] = input_model["bert.encoder.layer." + str(i) + ".output.LayerNorm.bias"]

output_model["output_layer.weight"] = input_model["qa_outputs.weight"]
output_model["output_layer.bias"] = input_model["qa_outputs.bias"]

torch.save(output_model, args.output_model_path)
