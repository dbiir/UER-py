import argparse
import collections
import torch


def convert_bert_transformer_encoder_from_uer_to_huggingface(input_model, output_model, layers_num):
    for i in range(layers_num):
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


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_model_path", type=str, default="models/input_model.bin",
                        help=".")
    parser.add_argument("--output_model_path", type=str, default="models/output_model.bin",
                        help=".")
    parser.add_argument("--layers_num", type=int, default=12, help=".")
    parser.add_argument("--target", choices=["bert", "mlm"], default="bert",
                        help="The training target of the pretraining model.")

    args = parser.parse_args()

    input_model = torch.load(args.input_model_path)

    output_model = collections.OrderedDict()

    output_model["bert.embeddings.word_embeddings.weight"] = input_model["embedding.word_embedding.weight"]
    output_model["bert.embeddings.position_embeddings.weight"] = input_model["embedding.position_embedding.weight"]
    output_model["bert.embeddings.token_type_embeddings.weight"] = input_model["embedding.segment_embedding.weight"][1:, :]
    output_model["bert.embeddings.LayerNorm.weight"] = input_model["embedding.layer_norm.gamma"]
    output_model["bert.embeddings.LayerNorm.bias"] = input_model["embedding.layer_norm.beta"]

    convert_bert_transformer_encoder_from_uer_to_huggingface(input_model, output_model, args.layers_num)

    if args.target == "bert":
        output_model["bert.pooler.dense.weight"] = input_model["target.nsp_linear_1.weight"]
        output_model["bert.pooler.dense.bias"] = input_model["target.nsp_linear_1.bias"]
        output_model["cls.seq_relationship.weight"] = input_model["target.nsp_linear_2.weight"]
        output_model["cls.seq_relationship.bias"] = input_model["target.nsp_linear_2.bias"]
    output_model["cls.predictions.transform.dense.weight"] = input_model["target.mlm_linear_1.weight"]
    output_model["cls.predictions.transform.dense.bias"] = input_model["target.mlm_linear_1.bias"]
    output_model["cls.predictions.transform.LayerNorm.weight"] = input_model["target.layer_norm.gamma"]
    output_model["cls.predictions.transform.LayerNorm.bias"] = input_model["target.layer_norm.beta"]
    output_model["cls.predictions.decoder.weight"] = input_model["target.mlm_linear_2.weight"]
    output_model["cls.predictions.bias"] = input_model["target.mlm_linear_2.bias"]

    torch.save(output_model, args.output_model_path)


if __name__ == "__main__":
    main()
