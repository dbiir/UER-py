import argparse
import collections
import torch


def convert_bert_transformer_encoder_from_huggingface_to_uer(input_model, output_model, layers_num):
    for i in range(layers_num):
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

    input_model = torch.load(args.input_model_path, map_location="cpu")

    output_model = collections.OrderedDict()

    output_model["embedding.word_embedding.weight"] = input_model["bert.embeddings.word_embeddings.weight"]
    output_model["embedding.position_embedding.weight"] = input_model["bert.embeddings.position_embeddings.weight"]
    output_model["embedding.segment_embedding.weight"] = torch.cat((torch.Tensor([[0]*input_model["bert.embeddings.token_type_embeddings.weight"].size()[1]]), input_model["bert.embeddings.token_type_embeddings.weight"]), dim=0)
    output_model["embedding.layer_norm.gamma"] = input_model["bert.embeddings.LayerNorm.weight"]
    output_model["embedding.layer_norm.beta"] = input_model["bert.embeddings.LayerNorm.bias"]

    convert_bert_transformer_encoder_from_huggingface_to_uer(input_model, output_model, args.layers_num)

    if args.target == "bert":
        output_model["target.nsp_linear_1.weight"] = input_model["bert.pooler.dense.weight"]
        output_model["target.nsp_linear_1.bias"] = input_model["bert.pooler.dense.bias"]
        output_model["target.nsp_linear_2.weight"] = input_model["cls.seq_relationship.weight"]
        output_model["target.nsp_linear_2.bias"] = input_model["cls.seq_relationship.bias"]
    output_model["target.mlm_linear_1.weight"] = input_model["cls.predictions.transform.dense.weight"]
    output_model["target.mlm_linear_1.bias"] = input_model["cls.predictions.transform.dense.bias"]
    output_model["target.layer_norm.gamma"] = input_model["cls.predictions.transform.LayerNorm.weight"]
    output_model["target.layer_norm.beta"] = input_model["cls.predictions.transform.LayerNorm.bias"]
    output_model["target.mlm_linear_2.weight"] = input_model["cls.predictions.decoder.weight"]
    output_model["target.mlm_linear_2.bias"] = input_model["cls.predictions.bias"]

    torch.save(output_model, args.output_model_path)


if __name__ == "__main__":
    main()
