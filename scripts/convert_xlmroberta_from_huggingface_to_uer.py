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

input_model = torch.load(args.input_model_path, map_location='cpu')

output_model = collections.OrderedDict()
emb_size = input_model["roberta.embeddings.word_embeddings.weight"].shape[1]

output_model["embedding.word_embedding.weight"] = input_model["roberta.embeddings.word_embeddings.weight"]
output_model["embedding.position_embedding.weight"] = torch.cat((input_model["roberta.embeddings.position_embeddings.weight"][2:], torch.zeros(2, emb_size)), 0)
output_model["embedding.segment_embedding.weight"] = torch.cat((torch.Tensor(torch.zeros(2, emb_size)), input_model["roberta.embeddings.token_type_embeddings.weight"]), dim=0)
output_model["embedding.layer_norm.gamma"] = input_model["roberta.embeddings.LayerNorm.weight"]
output_model["embedding.layer_norm.beta"] = input_model["roberta.embeddings.LayerNorm.bias"]

for i in range(args.layers_num):
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.weight"] = input_model["roberta.encoder.layer." + str(i) + ".attention.self.query.weight"]
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.bias"] = input_model["roberta.encoder.layer." + str(i) + ".attention.self.query.bias"]
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.weight"] = input_model["roberta.encoder.layer." + str(i) + ".attention.self.key.weight"]
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.bias"] = input_model["roberta.encoder.layer." + str(i) + ".attention.self.key.bias"]
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.weight"] = input_model["roberta.encoder.layer." + str(i) + ".attention.self.value.weight"]
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.bias"] = input_model["roberta.encoder.layer." + str(i) + ".attention.self.value.bias"]
    output_model["encoder.transformer." + str(i) + ".self_attn.final_linear.weight"] = input_model["roberta.encoder.layer." + str(i) + ".attention.output.dense.weight"]
    output_model["encoder.transformer." + str(i) + ".self_attn.final_linear.bias"] = input_model["roberta.encoder.layer." + str(i) + ".attention.output.dense.bias"]
    output_model["encoder.transformer." + str(i) + ".layer_norm_1.gamma"] = input_model["roberta.encoder.layer." + str(i) + ".attention.output.LayerNorm.weight"]
    output_model["encoder.transformer." + str(i) + ".layer_norm_1.beta"] = input_model["roberta.encoder.layer." + str(i) + ".attention.output.LayerNorm.bias"]
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"] = input_model["roberta.encoder.layer." + str(i) + ".intermediate.dense.weight"]
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.bias"] = input_model["roberta.encoder.layer." + str(i) + ".intermediate.dense.bias"]
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"] = input_model["roberta.encoder.layer." + str(i) + ".output.dense.weight"]
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.bias"] = input_model["roberta.encoder.layer." + str(i) + ".output.dense.bias"]
    output_model["encoder.transformer." + str(i) + ".layer_norm_2.gamma"] = input_model["roberta.encoder.layer." + str(i) + ".output.LayerNorm.weight"]
    output_model["encoder.transformer." + str(i) + ".layer_norm_2.beta"] = input_model["roberta.encoder.layer." + str(i) + ".output.LayerNorm.bias"]

output_model["target.mlm_linear_1.weight"] = input_model["lm_head.dense.weight"]
output_model["target.mlm_linear_1.bias"] = input_model["lm_head.dense.bias"]
output_model["target.layer_norm.gamma"] = input_model["lm_head.layer_norm.weight"]
output_model["target.layer_norm.beta"] = input_model["lm_head.layer_norm.bias"]
output_model["target.mlm_linear_2.weight"] = input_model["lm_head.decoder.weight"]
output_model["target.mlm_linear_2.bias"] = input_model["lm_head.bias"]

torch.save(output_model, args.output_model_path)
