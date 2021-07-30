import argparse
import collections
import torch


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="models/input_model.bin",
                    help=".")
parser.add_argument("--output_model_path", type=str, default="models/output_model.bin",
                    help=".")

args = parser.parse_args()

input_model = torch.load(args.input_model_path, map_location="cpu")

output_model = collections.OrderedDict()

output_model["embedding.word_embedding.weight"] = input_model["albert.embeddings.word_embeddings.weight"]
output_model["embedding.position_embedding.weight"] = input_model["albert.embeddings.position_embeddings.weight"]
output_model["embedding.segment_embedding.weight"] = torch.cat((torch.Tensor([[0]*input_model["albert.embeddings.token_type_embeddings.weight"].size()[1]]), input_model["albert.embeddings.token_type_embeddings.weight"]), dim=0)
output_model["embedding.layer_norm.gamma"] = input_model["albert.embeddings.LayerNorm.weight"]
output_model["embedding.layer_norm.beta"] = input_model["albert.embeddings.LayerNorm.bias"]

output_model["encoder.linear.weight"] = input_model["albert.encoder.embedding_hidden_mapping_in.weight"]
output_model["encoder.linear.bias"] = input_model["albert.encoder.embedding_hidden_mapping_in.bias"]
output_model["encoder.transformer.layer_norm_2.gamma"] = input_model["albert.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.weight"]
output_model["encoder.transformer.layer_norm_2.beta"] = input_model["albert.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.bias"]
output_model["encoder.transformer.self_attn.linear_layers.0.weight"] = input_model["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight"]
output_model["encoder.transformer.self_attn.linear_layers.0.bias"] = input_model["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.query.bias"]
output_model["encoder.transformer.self_attn.linear_layers.1.weight"] = input_model["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.key.weight"]
output_model["encoder.transformer.self_attn.linear_layers.1.bias"] = input_model["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.key.bias"]
output_model["encoder.transformer.self_attn.linear_layers.2.weight"] = input_model["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.value.weight"]
output_model["encoder.transformer.self_attn.linear_layers.2.bias"] = input_model["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.value.bias"]
output_model["encoder.transformer.self_attn.final_linear.weight"] = input_model["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.dense.weight"]
output_model["encoder.transformer.self_attn.final_linear.bias"] = input_model["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.dense.bias"]
output_model["encoder.transformer.layer_norm_1.gamma"] = input_model["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.weight"]
output_model["encoder.transformer.layer_norm_1.beta"] = input_model["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.bias"]
output_model["encoder.transformer.feed_forward.linear_1.weight"] = input_model["albert.encoder.albert_layer_groups.0.albert_layers.0.ffn.weight"]
output_model["encoder.transformer.feed_forward.linear_1.bias"] = input_model["albert.encoder.albert_layer_groups.0.albert_layers.0.ffn.bias"]
output_model["encoder.transformer.feed_forward.linear_2.weight"] = input_model["albert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.weight"]
output_model["encoder.transformer.feed_forward.linear_2.bias"] = input_model["albert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.bias"]

output_model["target.sop_linear_1.weight"] = input_model["albert.pooler.weight"]
output_model["target.sop_linear_1.bias"] = input_model["albert.pooler.bias"]
output_model["target.sop_linear_2.weight"] = input_model["sop_classifier.classifier.weight"]
output_model["target.sop_linear_2.bias"] = input_model["sop_classifier.classifier.bias"]
output_model["target.mlm_linear_1.weight"] = input_model["predictions.dense.weight"]
output_model["target.mlm_linear_1.bias"] = input_model["predictions.dense.bias"]
output_model["target.mlm_linear_2.weight"] = input_model["predictions.decoder.weight"]
output_model["target.mlm_linear_2.bias"] = input_model["predictions.bias"]
output_model["target.layer_norm.gamma"] = input_model["predictions.LayerNorm.weight"]
output_model["target.layer_norm.beta"] = input_model["predictions.LayerNorm.bias"]

torch.save(output_model, args.output_model_path)
