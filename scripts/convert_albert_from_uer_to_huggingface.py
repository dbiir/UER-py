import argparse
import collections
import torch


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="models/input_model.bin",
                    help=".")
parser.add_argument("--output_model_path", type=str, default="models/output_model.bin",
                    help=".")

args = parser.parse_args()

input_model = torch.load(args.input_model_path)

output_model = collections.OrderedDict()

output_model["albert.embeddings.word_embeddings.weight"] = input_model["embedding.word_embedding.weight"]
output_model["albert.embeddings.position_embeddings.weight"] = input_model["embedding.position_embedding.weight"]
output_model["albert.embeddings.token_type_embeddings.weight"] = input_model["embedding.segment_embedding.weight"][1:, :]
output_model["albert.embeddings.LayerNorm.weight"] = input_model["embedding.layer_norm.gamma"]
output_model["albert.embeddings.LayerNorm.bias"] = input_model["embedding.layer_norm.beta"]

output_model["albert.encoder.embedding_hidden_mapping_in.weight"] = input_model["encoder.linear.weight"]
output_model["albert.encoder.embedding_hidden_mapping_in.bias"] = input_model["encoder.linear.bias"]
output_model["albert.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.weight"] = input_model["encoder.transformer.layer_norm_2.gamma"]
output_model["albert.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.bias"] = input_model["encoder.transformer.layer_norm_2.beta"]
output_model["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight"] = input_model["encoder.transformer.self_attn.linear_layers.0.weight"]
output_model["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.query.bias"] = input_model["encoder.transformer.self_attn.linear_layers.0.bias"]
output_model["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.key.weight"] = input_model["encoder.transformer.self_attn.linear_layers.1.weight"]
output_model["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.key.bias"] = input_model["encoder.transformer.self_attn.linear_layers.1.bias"]
output_model["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.value.weight"] = input_model["encoder.transformer.self_attn.linear_layers.2.weight"]
output_model["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.value.bias"] = input_model["encoder.transformer.self_attn.linear_layers.2.bias"]
output_model["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.dense.weight"] = input_model["encoder.transformer.self_attn.final_linear.weight"]
output_model["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.dense.bias"] = input_model["encoder.transformer.self_attn.final_linear.bias"]
output_model["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.weight"] = input_model["encoder.transformer.layer_norm_1.gamma"]
output_model["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.bias"] = input_model["encoder.transformer.layer_norm_1.beta"]
output_model["albert.encoder.albert_layer_groups.0.albert_layers.0.ffn.weight"] = input_model["encoder.transformer.feed_forward.linear_1.weight"]
output_model["albert.encoder.albert_layer_groups.0.albert_layers.0.ffn.bias"] = input_model["encoder.transformer.feed_forward.linear_1.bias"]
output_model["albert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.weight"] = input_model["encoder.transformer.feed_forward.linear_2.weight"]
output_model["albert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.bias"] = input_model["encoder.transformer.feed_forward.linear_2.bias"]

output_model["albert.pooler.weight"] = input_model["target.sop_linear_1.weight"]
output_model["albert.pooler.bias"] = input_model["target.sop_linear_1.bias"]
output_model["sop_classifier.classifier.weight"] = input_model["target.sop_linear_2.weight"]
output_model["sop_classifier.classifier.bias"] = input_model["target.sop_linear_2.bias"]
output_model["predictions.dense.weight"] = input_model["target.mlm_linear_1.weight"]
output_model["predictions.dense.bias"] = input_model["target.mlm_linear_1.bias"]
output_model["predictions.LayerNorm.weight"] = input_model["target.layer_norm.gamma"]
output_model["predictions.LayerNorm.bias"] = input_model["target.layer_norm.beta"]
output_model["predictions.decoder.weight"] = input_model["target.mlm_linear_2.weight"]
output_model["predictions.decoder.bias"] = input_model["target.mlm_linear_2.bias"]
output_model["predictions.bias"] = input_model["target.mlm_linear_2.bias"]

torch.save(output_model, args.output_model_path)
