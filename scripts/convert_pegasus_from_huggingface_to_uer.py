import torch
import argparse
import collections

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="pytorch_model.bin",
                    help=".")
parser.add_argument("--output_model_path", type=str, default="huggingface_model.bin",
                    help=".")
parser.add_argument("--layers_num", type=int, default=6, help=".")


args = parser.parse_args()
path = args.input_model_path

input_model = torch.load(args.input_model_path, map_location="cpu")

output_model = collections.OrderedDict()

output_model["embedding.pe"] = input_model["model.encoder.embed_positions.weight"].unsqueeze(1)
output_model["target.embedding.pe"] = input_model["model.decoder.embed_positions.weight"].unsqueeze(1)
output_model["embedding.word_embedding.weight"] = input_model["model.encoder.embed_tokens.weight"]
output_model["target.embedding.word_embedding.weight"] = input_model["model.decoder.embed_tokens.weight"]
output_model["target.output_layer.weight"] = input_model["lm_head.weight"]
output_model["target.output_layer.bias"] = input_model["final_logits_bias"].squeeze(0)
for i in range(args.layers_num):
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.weight"] = input_model["model.encoder.layers." + str(i) + ".self_attn.q_proj.weight"]
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.bias"] = input_model["model.encoder.layers." + str(i) + ".self_attn.q_proj.bias"]
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.weight"] = input_model["model.encoder.layers." + str(i) + ".self_attn.k_proj.weight"]
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.bias"] = input_model["model.encoder.layers." + str(i) + ".self_attn.k_proj.bias"]
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.weight"] = input_model["model.encoder.layers." + str(i) + ".self_attn.v_proj.weight"]
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.bias"] = input_model["model.encoder.layers." + str(i) + ".self_attn.v_proj.bias"]
    output_model["encoder.transformer." + str(i) + ".self_attn.final_linear.weight"] = input_model["model.encoder.layers." + str(i) + ".self_attn.out_proj.weight"]
    output_model["encoder.transformer." + str(i) + ".self_attn.final_linear.bias"] = input_model["model.encoder.layers." + str(i) + ".self_attn.out_proj.bias"]
    output_model["encoder.transformer." + str(i) + ".layer_norm_1.gamma"] = input_model["model.encoder.layers." + str(i) + ".self_attn_layer_norm.weight"]
    output_model["encoder.transformer." + str(i) + ".layer_norm_1.beta"] = input_model["model.encoder.layers." + str(i) + ".self_attn_layer_norm.bias"]
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"] = input_model["model.encoder.layers." + str(i) + ".fc1.weight"]
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.bias"] = input_model["model.encoder.layers." + str(i) + ".fc1.bias"]
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"] = input_model["model.encoder.layers." + str(i) + ".fc2.weight"]
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.bias"] = input_model["model.encoder.layers." + str(i) + ".fc2.bias"]
    output_model["encoder.transformer." + str(i) + ".layer_norm_2.gamma"] = input_model["model.encoder.layers." + str(i) + ".final_layer_norm.weight"]
    output_model["encoder.transformer." + str(i) + ".layer_norm_2.beta"] = input_model["model.encoder.layers." + str(i) + ".final_layer_norm.bias"]

    output_model["target.decoder.transformer_decoder." + str(i) + ".self_attn.linear_layers.0.weight"] = input_model["model.decoder.layers." + str(i) + ".self_attn.q_proj.weight"]
    output_model["target.decoder.transformer_decoder." + str(i) + ".self_attn.linear_layers.0.bias"] = input_model["model.decoder.layers." + str(i) + ".self_attn.q_proj.bias"]
    output_model["target.decoder.transformer_decoder." + str(i) + ".self_attn.linear_layers.1.weight"] = input_model["model.decoder.layers." + str(i) + ".self_attn.k_proj.weight"]
    output_model["target.decoder.transformer_decoder." + str(i) + ".self_attn.linear_layers.1.bias"] = input_model["model.decoder.layers." + str(i) + ".self_attn.k_proj.bias"]
    output_model["target.decoder.transformer_decoder." + str(i) + ".self_attn.linear_layers.2.weight"] = input_model["model.decoder.layers." + str(i) + ".self_attn.v_proj.weight"]
    output_model["target.decoder.transformer_decoder." + str(i) + ".self_attn.linear_layers.2.bias"] = input_model["model.decoder.layers." + str(i) + ".self_attn.v_proj.bias"]
    output_model["target.decoder.transformer_decoder." + str(i) + ".self_attn.final_linear.weight"] = input_model["model.decoder.layers." + str(i) + ".self_attn.out_proj.weight"]
    output_model["target.decoder.transformer_decoder." + str(i) + ".self_attn.final_linear.bias"] = input_model["model.decoder.layers." + str(i) + ".self_attn.out_proj.bias"]
    output_model["target.decoder.transformer_decoder." + str(i) + ".layer_norm_1.gamma"] = input_model["model.decoder.layers." + str(i) + ".self_attn_layer_norm.weight"]
    output_model["target.decoder.transformer_decoder." + str(i) + ".layer_norm_1.beta"] = input_model["model.decoder.layers." + str(i) + ".self_attn_layer_norm.bias"]

    output_model["target.decoder.transformer_decoder." + str(i) + ".context_attn.linear_layers.0.weight"] = input_model["model.decoder.layers." + str(i) + ".encoder_attn.q_proj.weight"]
    output_model["target.decoder.transformer_decoder." + str(i) + ".context_attn.linear_layers.0.bias"] = input_model["model.decoder.layers." + str(i) + ".encoder_attn.q_proj.bias"]
    output_model["target.decoder.transformer_decoder." + str(i) + ".context_attn.linear_layers.1.weight"] = input_model["model.decoder.layers." + str(i) + ".encoder_attn.k_proj.weight"]
    output_model["target.decoder.transformer_decoder." + str(i) + ".context_attn.linear_layers.1.bias"] = input_model["model.decoder.layers." + str(i) + ".encoder_attn.k_proj.bias"]
    output_model["target.decoder.transformer_decoder." + str(i) + ".context_attn.linear_layers.2.weight"] = input_model["model.decoder.layers." + str(i) + ".encoder_attn.v_proj.weight"]
    output_model["target.decoder.transformer_decoder." + str(i) + ".context_attn.linear_layers.2.bias"] = input_model["model.decoder.layers." + str(i) + ".encoder_attn.v_proj.bias"]
    output_model["target.decoder.transformer_decoder." + str(i) + ".context_attn.final_linear.weight"] = input_model["model.decoder.layers." + str(i) + ".encoder_attn.out_proj.weight"]
    output_model["target.decoder.transformer_decoder." + str(i) + ".context_attn.final_linear.bias"] = input_model["model.decoder.layers." + str(i) + ".encoder_attn.out_proj.bias"]
    output_model["target.decoder.transformer_decoder." + str(i) + ".layer_norm_2.gamma"] = input_model["model.decoder.layers." + str(i) + ".encoder_attn_layer_norm.weight"]
    output_model["target.decoder.transformer_decoder." + str(i) + ".layer_norm_2.beta"] = input_model["model.decoder.layers." + str(i) + ".encoder_attn_layer_norm.bias"]

    output_model["target.decoder.transformer_decoder." + str(i) + ".feed_forward.linear_1.weight"] = input_model["model.decoder.layers." + str(i) + ".fc1.weight"]
    output_model["target.decoder.transformer_decoder." + str(i) + ".feed_forward.linear_1.bias"] = input_model["model.decoder.layers." + str(i) + ".fc1.bias"]
    output_model["target.decoder.transformer_decoder." + str(i) + ".feed_forward.linear_2.weight"] = input_model["model.decoder.layers." + str(i) + ".fc2.weight"]
    output_model["target.decoder.transformer_decoder." + str(i) + ".feed_forward.linear_2.bias"] = input_model["model.decoder.layers." + str(i) + ".fc2.bias"]
    output_model["target.decoder.transformer_decoder." + str(i) + ".layer_norm_3.gamma"] = input_model["model.decoder.layers." + str(i) + ".final_layer_norm.weight"]
    output_model["target.decoder.transformer_decoder." + str(i) + ".layer_norm_3.beta"] = input_model["model.decoder.layers." + str(i) + ".final_layer_norm.bias"]


output_model["encoder.layer_norm.gamma"] = input_model["model.encoder.layer_norm.weight"]
output_model["encoder.layer_norm.beta"] = input_model["model.encoder.layer_norm.bias"]
output_model["target.decoder.layer_norm.gamma"] = input_model["model.decoder.layer_norm.weight"]
output_model["target.decoder.layer_norm.beta"] = input_model["model.decoder.layer_norm.bias"]

torch.save(output_model, args.output_model_path)
