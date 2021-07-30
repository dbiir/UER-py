import argparse
import collections
import torch


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="models/input_model.bin",
                        help=".")
parser.add_argument("--output_model_path", type=str, default="models/output_model.bin",
                        help=".")
parser.add_argument("--layers_num", type=int, default=12, help=".")

args = parser.parse_args()

input_model = torch.load(args.input_model_path)

output_model = collections.OrderedDict()
emb_size = input_model["embedding.word_embedding.weight"].shape[1]

output_model["roberta.embeddings.word_embeddings.weight"] = input_model["embedding.word_embedding.weight"]
output_model["roberta.embeddings.position_embeddings.weight"] = torch.cat((torch.zeros(2, emb_size), input_model["embedding.position_embedding.weight"][:-2]),0)
output_model["roberta.embeddings.token_type_embeddings.weight"] = input_model["embedding.segment_embedding.weight"][2:, :]
output_model["roberta.embeddings.LayerNorm.weight"] = input_model["embedding.layer_norm.gamma"]
output_model["roberta.embeddings.LayerNorm.bias"] = input_model["embedding.layer_norm.beta"]

for i in range(args.layers_num):
    output_model["roberta.encoder.layer." + str(i) + ".attention.self.query.weight"] = input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.weight"]
    output_model["roberta.encoder.layer." + str(i) + ".attention.self.query.bias"] = input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.bias"]
    output_model["roberta.encoder.layer." + str(i) + ".attention.self.key.weight"] = input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.weight"]
    output_model["roberta.encoder.layer." + str(i) + ".attention.self.key.bias"] = input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.bias"]
    output_model["roberta.encoder.layer." + str(i) + ".attention.self.value.weight"] = input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.weight"]
    output_model["roberta.encoder.layer." + str(i) + ".attention.self.value.bias"] = input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.bias"]
    output_model["roberta.encoder.layer." + str(i) + ".attention.output.dense.weight"] = input_model["encoder.transformer." + str(i) + ".self_attn.final_linear.weight"]
    output_model["roberta.encoder.layer." + str(i) + ".attention.output.dense.bias"] = input_model["encoder.transformer." + str(i) + ".self_attn.final_linear.bias"]
    output_model["roberta.encoder.layer." + str(i) + ".attention.output.LayerNorm.weight"] = input_model["encoder.transformer." + str(i) + ".layer_norm_1.gamma"]
    output_model["roberta.encoder.layer." + str(i) + ".attention.output.LayerNorm.bias"] = input_model["encoder.transformer." + str(i) + ".layer_norm_1.beta"]
    output_model["roberta.encoder.layer." + str(i) + ".intermediate.dense.weight"] = input_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"]
    output_model["roberta.encoder.layer." + str(i) + ".intermediate.dense.bias"] = input_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.bias"]
    output_model["roberta.encoder.layer." + str(i) + ".output.dense.weight"] = input_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"]
    output_model["roberta.encoder.layer." + str(i) + ".output.dense.bias"] = input_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.bias"]
    output_model["roberta.encoder.layer." + str(i) + ".output.LayerNorm.weight"] = input_model["encoder.transformer." + str(i) + ".layer_norm_2.gamma"]
    output_model["roberta.encoder.layer." + str(i) + ".output.LayerNorm.bias"] = input_model["encoder.transformer." + str(i) + ".layer_norm_2.beta"]

output_model["lm_head.dense.weight"] = input_model["target.mlm_linear_1.weight"]
output_model["lm_head.dense.bias"] = input_model["target.mlm_linear_1.bias"]
output_model["lm_head.layer_norm.weight"] = input_model["target.layer_norm.gamma"]
output_model["lm_head.layer_norm.bias"] = input_model["target.layer_norm.beta"]
output_model["lm_head.decoder.weight"] = input_model["target.mlm_linear_2.weight"]
output_model["lm_head.decoder.bias"] = input_model["target.mlm_linear_2.bias"]
output_model["lm_head.bias"] = input_model["target.mlm_linear_2.bias"]

torch.save(output_model, args.output_model_path)
