import argparse
import collections
import tensorflow as tf
import torch
from tensorflow.python import pywrap_tensorflow


tensors_to_transopse = (
        "dense/kernel",
        "attention/self/query",
        "attention/self/key",
        "attention/self/value"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--layers_num", type=int, default=12,
                        help=".")
    parser.add_argument("--input_model_path", default="models/input_model.ckpt", type=str,
                        help=".")
    parser.add_argument("--output_model_path", default="models/output_model.bin", type=str,
                        help="Path to the output PyTorch model.")
    parser.add_argument("--target", choices=["bert", "mlm"], default="bert",
                        help="The training target of the pretraining model.")

    args = parser.parse_args()

    reader = pywrap_tensorflow.NewCheckpointReader(args.input_model_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    input_model = collections.OrderedDict()

    for key in var_to_shape_map:
        torch_tensor = reader.get_tensor(key)

        if any([x in key for x in tensors_to_transopse]):
            torch_tensor = torch_tensor.T
        if key == "bert/embeddings/token_type_embeddings":
            col_dim = torch_tensor.shape[1]
            sess = tf.Session()
            zeros_var = tf.Variable(tf.zeros([1, col_dim], dtype=tf.float32), name="zeros_var")
            sess.run(zeros_var.initializer)
            torch_tensor = sess.run(tf.concat([sess.run(zeros_var), torch_tensor], 0))
        input_model[key] = torch.Tensor(torch_tensor)

    output_model = collections.OrderedDict()
    output_model["embedding.word_embedding.weight"] = input_model["bert/embeddings/word_embeddings"]
    output_model["embedding.position_embedding.weight"] = input_model["bert/embeddings/position_embeddings"][:512]
    output_model["embedding.segment_embedding.weight"] = input_model["bert/embeddings/token_type_embeddings"]
    output_model["embedding.layer_norm.gamma"] = input_model["bert/embeddings/LayerNorm/gamma"]
    output_model["embedding.layer_norm.beta"] = input_model["bert/embeddings/LayerNorm/beta"]

    for i in range(args.layers_num):
        output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.weight"] = input_model["bert/encoder/layer_" + str(i) + "/attention/self/query/kernel"]
        output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.bias"] = input_model["bert/encoder/layer_" + str(i) + "/attention/self/query/bias"]
        output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.weight"] = input_model["bert/encoder/layer_" + str(i) + "/attention/self/key/kernel"]
        output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.bias"] = input_model["bert/encoder/layer_" + str(i) + "/attention/self/key/bias"]
        output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.weight"] = input_model["bert/encoder/layer_" + str(i) + "/attention/self/value/kernel"]
        output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.bias"] = input_model["bert/encoder/layer_" + str(i) + "/attention/self/value/bias"]
        output_model["encoder.transformer." + str(i) + ".self_attn.final_linear.weight"] = input_model["bert/encoder/layer_" + str(i) + "/attention/output/dense/kernel"]
        output_model["encoder.transformer." + str(i) + ".self_attn.final_linear.bias"] = input_model["bert/encoder/layer_" + str(i) + "/attention/output/dense/bias"]
        output_model["encoder.transformer." + str(i) + ".layer_norm_1.gamma"] = input_model["bert/encoder/layer_" + str(i) + "/attention/output/LayerNorm/gamma"]
        output_model["encoder.transformer." + str(i) + ".layer_norm_1.beta"] = input_model["bert/encoder/layer_" + str(i) + "/attention/output/LayerNorm/beta"]
        output_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"] = input_model["bert/encoder/layer_" + str(i) + "/intermediate/dense/kernel"]
        output_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.bias"] = input_model["bert/encoder/layer_" + str(i) + "/intermediate/dense/bias"]
        output_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"] = input_model["bert/encoder/layer_" + str(i) + "/output/dense/kernel"]
        output_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.bias"] = input_model["bert/encoder/layer_" + str(i) + "/output/dense/bias"]
        output_model["encoder.transformer." + str(i) + ".layer_norm_2.gamma"] = input_model["bert/encoder/layer_" + str(i) + "/output/LayerNorm/gamma"]
        output_model["encoder.transformer." + str(i) + ".layer_norm_2.beta"] = input_model["bert/encoder/layer_" + str(i) + "/output/LayerNorm/beta"]

    if args.target == "bert":
        output_model["target.nsp_linear_1.weight"] = input_model["bert/pooler/dense/kernel"]
        output_model["target.nsp_linear_1.bias"] = input_model["bert/pooler/dense/bias"]
        output_model["target.nsp_linear_2.weight"] = input_model["cls/seq_relationship/output_weights"]
        output_model["target.nsp_linear_2.bias"] = input_model["cls/seq_relationship/output_bias"]
    output_model["target.mlm_linear_1.weight"] = input_model["cls/predictions/transform/dense/kernel"]
    output_model["target.mlm_linear_1.bias"] = input_model["cls/predictions/transform/dense/bias"]
    output_model["target.layer_norm.gamma"] = input_model["cls/predictions/transform/LayerNorm/gamma"]
    output_model["target.layer_norm.beta"] = input_model["cls/predictions/transform/LayerNorm/beta"]
    output_model["target.mlm_linear_2.weight"] = input_model["bert/embeddings/word_embeddings"]
    output_model["target.mlm_linear_2.bias"] = input_model["cls/predictions/output_bias"]

    torch.save(output_model, args.output_model_path)


if __name__ == "__main__":
    main()
