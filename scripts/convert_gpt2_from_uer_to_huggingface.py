import torch
import argparse
import collections
import re

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="pytorch_model.bin",
                        help=".")
parser.add_argument("--output_model_path", type=str, default="gpt_model.bin",
                        help=".")
parser.add_argument("--layers_num", type=int, default=12)

args = parser.parse_args()
path = args.input_model_path

input_model = torch.load(args.input_model_path)

output_model = collections.OrderedDict()


output_model["transformer.wte.weight"] = input_model["embedding.word_embedding.weight"]
output_model["transformer.wpe.weight"] = input_model["embedding.position_embedding.weight"]

for i in range(args.layers_num):
    output_model["transformer.h." + str(i) + ".attn.bias"] = torch.tril(torch.ones(1024, 1024)).view(1, 1, 1024, 1024)
    weight = []
    bias = []
    for j in range(3):
        weight.append(input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers." + str(j) + ".weight"])
        bias.append(input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers." + str(j) + ".bias"])
    output_model["transformer.h." + str(i) + ".attn.c_attn.weight"] = torch.cat(weight, 0).t()
    output_model["transformer.h." + str(i) + ".attn.c_attn.bias"] = torch.cat(bias, 0)

    output_model["transformer.h." + str(i) + ".attn.c_proj.weight"] = input_model["encoder.transformer." + str(i) + ".self_attn.final_linear.weight"].t()
    output_model["transformer.h." + str(i) + ".attn.c_proj.bias"] = input_model["encoder.transformer." + str(i) + ".self_attn.final_linear.bias"]

    output_model["transformer.h." + str(i) + ".ln_1.weight"] = input_model["encoder.transformer." + str(i) + ".layer_norm_1.gamma"]
    output_model["transformer.h." + str(i) + ".ln_1.bias"] = input_model["encoder.transformer." + str(i) + ".layer_norm_1.beta"]

    output_model["transformer.h." + str(i) + ".mlp.c_fc.weight"] = input_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"].t()
    output_model["transformer.h." + str(i) + ".mlp.c_fc.bias"] = input_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.bias"]
    output_model["transformer.h." + str(i) + ".mlp.c_proj.weight"] = input_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"].t()
    output_model["transformer.h." + str(i) + ".mlp.c_proj.bias"] = input_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.bias"]

    output_model["transformer.h." + str(i) + ".ln_2.weight"] = input_model["encoder.transformer." + str(i) + ".layer_norm_2.gamma"]
    output_model["transformer.h." + str(i) + ".ln_2.bias"] = input_model["encoder.transformer." + str(i) + ".layer_norm_2.beta"]


output_model["transformer.ln_f.weight"] = input_model["encoder.layer_norm.gamma"]
output_model["transformer.ln_f.bias"] = input_model["encoder.layer_norm.beta"]
output_model["lm_head.weight"] = input_model["embedding.word_embedding.weight"]

torch.save(output_model, args.output_model_path)
