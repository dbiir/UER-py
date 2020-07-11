import os
import numpy as np
import tensorflow as tf
import torch
import argparse
import collections

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--layers_num", type=int, default="12",
                        help=".")
    parser.add_argument("--input_model_path", type=str, default="models/google_model.bin",
                        help=".")
    parser.add_argument("--output_model_path",
                        type=str,
                        default="models/bert_base_chinese.ckpt",
                        help=".")

    args = parser.parse_args()

    input_model = torch.load(args.input_model_path)

    session = tf.Session()
    output_model = collections.OrderedDict()

    output_model["bert/embeddings/word_embeddings"] = input_model["embedding.word_embedding.weight"]
    output_model["bert/embeddings/position_embeddings"] = input_model["embedding.position_embedding.weight"]
    output_model["bert/embeddings/token_type_embeddings"] = input_model["embedding.segment_embedding.weight"][1:, :]
    output_model["bert/embeddings/LayerNorm/gamma"] = input_model["embedding.layer_norm.gamma"]
    output_model["bert/embeddings/LayerNorm/beta"] = input_model["embedding.layer_norm.beta"]

    for i in range(args.layers_num):
        output_model["bert/encoder/layer_" + str(i) + "/attention/self/query/kernel"] = input_model[
            "encoder.transformer." + str(i) + ".self_attn.linear_layers.0.weight"]
        output_model["bert/encoder/layer_" + str(i) + "/attention/self/query/bias"] = input_model[
            "encoder.transformer." + str(i) + ".self_attn.linear_layers.0.bias"]
        output_model["bert/encoder/layer_" + str(i) + "/attention/self/key/kernel"] = input_model[
            "encoder.transformer." + str(i) + ".self_attn.linear_layers.1.weight"]
        output_model["bert/encoder/layer_" + str(i) + "/attention/self/key/bias"] = input_model[
            "encoder.transformer." + str(i) + ".self_attn.linear_layers.1.bias"]
        output_model["bert/encoder/layer_" + str(i) + "/attention/self/value/kernel"] = input_model[
            "encoder.transformer." + str(i) + ".self_attn.linear_layers.2.weight"]
        output_model["bert/encoder/layer_" + str(i) + "/attention/self/value/bias"] = input_model[
            "encoder.transformer." + str(i) + ".self_attn.linear_layers.2.bias"]
        output_model["bert/encoder/layer_" + str(i) + "/attention/output/dense/kernel"] = input_model[
            "encoder.transformer." + str(i) + ".self_attn.final_linear.weight"]
        output_model["bert/encoder/layer_" + str(i) + "/attention/output/dense/bias"] = input_model[
            "encoder.transformer." + str(i) + ".self_attn.final_linear.bias"]
        output_model["bert/encoder/layer_" + str(i) + "/attention/output/LayerNorm/gamma"] = input_model[
            "encoder.transformer." + str(i) + ".layer_norm_1.gamma"]
        output_model["bert/encoder/layer_" + str(i) + "/attention/output/LayerNorm/beta"] = input_model[
            "encoder.transformer." + str(i) + ".layer_norm_1.beta"]
        output_model["bert/encoder/layer_" + str(i) + "/intermediate/dense/kernel"] = input_model[
            "encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"]
        output_model["bert/encoder/layer_" + str(i) + "/intermediate/dense/bias"] = input_model[
            "encoder.transformer." + str(i) + ".feed_forward.linear_1.bias"]
        output_model["bert/encoder/layer_" + str(i) + "/output/dense/kernel"] = input_model[
            "encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"]
        output_model["bert/encoder/layer_" + str(i) + "/output/dense/bias"] = input_model[
            "encoder.transformer." + str(i) + ".feed_forward.linear_2.bias"]
        output_model["bert/encoder/layer_" + str(i) + "/output/LayerNorm/gamma"] = input_model[
            "encoder.transformer." + str(i) + ".layer_norm_2.gamma"]
        output_model["bert/encoder/layer_" + str(i) + "/output/LayerNorm/beta"] = input_model[
            "encoder.transformer." + str(i) + ".layer_norm_2.beta"]

    output_model["bert/pooler/dense/kernel"] = input_model["target.nsp_linear_1.weight"]
    output_model["bert/pooler/dense/bias"] = input_model["target.nsp_linear_1.bias"]
    output_model["cls/seq_relationship/output_weights"] = input_model["target.nsp_linear_2.weight"]
    output_model["cls/seq_relationship/output_bias"] = input_model["target.nsp_linear_2.bias"]
    output_model["cls/predictions/transform/dense/kernel"] = input_model["target.mlm_linear_1.weight"]
    output_model["cls/predictions/transform/dense/bias"] = input_model["target.mlm_linear_1.bias"]
    output_model["cls/predictions/transform/LayerNorm/gamma"] = input_model["target.layer_norm.gamma"]
    output_model["cls/predictions/transform/LayerNorm/beta"] = input_model["target.layer_norm.beta"]
    output_model["cls/predictions/output_bias"] = input_model["target.mlm_linear_2.bias"]

    def assign_tf_var(tensor: np.ndarray, name: str):
        tmp_var = tf.Variable(initial_value=tensor)
        tf_var = tf.get_variable(dtype=tmp_var.dtype, shape=tmp_var.shape, name=name)
        op = tf.assign(ref=tf_var, value=tmp_var)
        session.run(tf.variables_initializer([tmp_var, tf_var]))
        session.run(fetches=[op, tf_var])
        return tf_var
    tf_vars = []
    tensors_to_transopse = (
        "dense/kernel",
        "attention/self/query",
        "attention/self/key",
        "attention/self/value"
    )
    for k, v in output_model.items():
        tf_name = k
        torch_tensor = v.cpu().numpy()
        if any([x in k for x in tensors_to_transopse]):
            torch_tensor = torch_tensor.T
        tf_tensor = assign_tf_var(tensor=torch_tensor, name=tf_name)
        tf_vars.append(tf_tensor)
        print("{0}{1}initialized".format(tf_name, " " * (60 - len(tf_name))))

    saver = tf.train.Saver(tf_vars)
    saver.save(session, args.output_model_path)

if __name__ == "__main__":
    main()

