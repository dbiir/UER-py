import argparse
import collections
import torch


def convert_encoder_decoder_transformer_from_huggingface_to_uer(input_model, output_model, layers_num):
    for i in range(layers_num):
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


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_model_path", type=str, default="models/input_model.bin",
                        help=".")
    parser.add_argument("--output_model_path", type=str, default="models/output_model.bin",
                        help=".")
    parser.add_argument("--layers_num", type=int, default=6, help=".")


    args = parser.parse_args()

    input_model = torch.load(args.input_model_path, map_location="cpu")

    output_model = collections.OrderedDict()

    output_model["embedding.position_embedding.weight"] = input_model["model.encoder.embed_positions.weight"][2:]
    output_model["target.embedding.position_embedding.weight"] = input_model["model.decoder.embed_positions.weight"][2:]
    output_model["embedding.word_embedding.weight"] = input_model["model.encoder.embed_tokens.weight"]
    output_model["target.embedding.word_embedding.weight"] = input_model["model.decoder.embed_tokens.weight"]
    output_model["target.output_layer.weight"] = input_model["lm_head.weight"]
    output_model["target.output_layer.bias"] = input_model["final_logits_bias"].squeeze(0)

    convert_encoder_decoder_transformer_from_huggingface_to_uer(input_model, output_model, args.layers_num)

    output_model["embedding.layer_norm.gamma"] = input_model["model.encoder.layernorm_embedding.weight"]
    output_model["embedding.layer_norm.beta"] = input_model["model.encoder.layernorm_embedding.bias"]
    output_model["target.embedding.layer_norm.gamma"] = input_model["model.decoder.layernorm_embedding.weight"]
    output_model["target.embedding.layer_norm.beta"] = input_model["model.decoder.layernorm_embedding.bias"]

    torch.save(output_model, args.output_model_path)


if __name__ == "__main__":
    main()
