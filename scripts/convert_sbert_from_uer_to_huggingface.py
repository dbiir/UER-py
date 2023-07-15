import argparse
import collections
import torch


def convert_sbert_transformer_encoder_from_uer_to_huggingface(input_model, output_model, layers_num):
    for i in range(layers_num):
        output_model["encoder.layer." + str(i) + ".attention.self.query.weight"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".self_attn.linear_layers.0.weight"]
        output_model["encoder.layer." + str(i) + ".attention.self.query.bias"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".self_attn.linear_layers.0.bias"]
        output_model["encoder.layer." + str(i) + ".attention.self.key.weight"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".self_attn.linear_layers.1.weight"]
        output_model["encoder.layer." + str(i) + ".attention.self.key.bias"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".self_attn.linear_layers.1.bias"]
        output_model["encoder.layer." + str(i) + ".attention.self.value.weight"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".self_attn.linear_layers.2.weight"]
        output_model["encoder.layer." + str(i) + ".attention.self.value.bias"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".self_attn.linear_layers.2.bias"]
        output_model["encoder.layer." + str(i) + ".attention.output.dense.weight"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".self_attn.final_linear.weight"]
        output_model["encoder.layer." + str(i) + ".attention.output.dense.bias"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".self_attn.final_linear.bias"]
        output_model["encoder.layer." + str(i) + ".attention.output.LayerNorm.weight"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".layer_norm_1.gamma"]
        output_model["encoder.layer." + str(i) + ".attention.output.LayerNorm.bias"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".layer_norm_1.beta"]
        output_model["encoder.layer." + str(i) + ".intermediate.dense.weight"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".feed_forward.linear_1.weight"]
        output_model["encoder.layer." + str(i) + ".intermediate.dense.bias"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".feed_forward.linear_1.bias"]
        output_model["encoder.layer." + str(i) + ".output.dense.weight"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".feed_forward.linear_2.weight"]
        output_model["encoder.layer." + str(i) + ".output.dense.bias"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".feed_forward.linear_2.bias"]
        output_model["encoder.layer." + str(i) + ".output.LayerNorm.weight"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".layer_norm_2.gamma"]
        output_model["encoder.layer." + str(i) + ".output.LayerNorm.bias"] = \
            input_model["encoder.encoder_0.transformer." + str(i) + ".layer_norm_2.beta"]


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_model_path", type=str, default="models/input_model.bin",
                        help=".")
    parser.add_argument("--output_model_path", type=str, default="models/output_model.bin",
                        help=".")
    parser.add_argument("--layers_num", type=int, default=12, help=".")

    args = parser.parse_args()

    input_model = torch.load(args.input_model_path, map_location='cpu')

    output_model = collections.OrderedDict()

    output_model["embeddings.word_embeddings.weight"] = \
        input_model["embedding.embedding_0.word.embedding.weight"]
    output_model["embeddings.position_embeddings.weight"] = \
        input_model["embedding.embedding_0.pos.embedding.weight"]
    output_model["embeddings.token_type_embeddings.weight"] = \
        input_model["embedding.embedding_0.seg.embedding.weight"][1:, :]
    output_model["embeddings.LayerNorm.weight"] = \
        input_model["embedding.embedding_0.layer_norm.gamma"]
    output_model["embeddings.LayerNorm.bias"] = \
        input_model["embedding.embedding_0.layer_norm.beta"]

    convert_sbert_transformer_encoder_from_uer_to_huggingface(input_model, output_model, args.layers_num)
    torch.save(output_model, args.output_model_path)

if __name__ == "__main__":
    main()
