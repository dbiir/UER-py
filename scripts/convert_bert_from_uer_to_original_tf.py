import sys
import os
import argparse
import collections
import numpy as np
import tensorflow as tf
import torch

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, uer_dir)

from scripts.convert_bert_from_original_tf_to_uer import tensors_to_transopse


def assign_tf_var(tensor: np.ndarray, name: str):
    tf_var = tf.get_variable(dtype=tensor.dtype, shape=tensor.shape, name=name)
    tf.keras.backend.set_value(tf_var, tensor)
    return tf_var


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--layers_num", type=int, default=12,
                        help=".")
    parser.add_argument("--input_model_path", type=str, default="models/input_model.bin",
                        help=".")
    parser.add_argument("--output_model_path",type=str,default="models/output_model.ckpt",
                        help=".")
    parser.add_argument("--target", choices=["bert", "mlm"], default="bert",
                        help="The training target of the pretraining model.")

    args = parser.parse_args()

    input_model = torch.load(args.input_model_path, map_location="cpu")

    session = tf.Session()
    tf.keras.backend.set_session(session)
    
    output_model = collections.OrderedDict()

    output_model["bert/embeddings/word_embeddings"] = input_model["embedding.word_embedding.weight"]
    output_model["bert/embeddings/position_embeddings"] = input_model["embedding.position_embedding.weight"]
    output_model["bert/embeddings/token_type_embeddings"] = input_model["embedding.segment_embedding.weight"][1:, :]
    output_model["bert/embeddings/LayerNorm/gamma"] = input_model["embedding.layer_norm.gamma"]
    output_model["bert/embeddings/LayerNorm/beta"] = input_model["embedding.layer_norm.beta"]

    for i in range(args.layers_num):
        output_model["bert/encoder/layer_" + str(i) + "/attention/self/query/kernel"] = input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.weight"]
        output_model["bert/encoder/layer_" + str(i) + "/attention/self/query/bias"] = input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.bias"]
        output_model["bert/encoder/layer_" + str(i) + "/attention/self/key/kernel"] = input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.weight"]
        output_model["bert/encoder/layer_" + str(i) + "/attention/self/key/bias"] = input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.bias"]
        output_model["bert/encoder/layer_" + str(i) + "/attention/self/value/kernel"] = input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.weight"]
        output_model["bert/encoder/layer_" + str(i) + "/attention/self/value/bias"] = input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.bias"]
        output_model["bert/encoder/layer_" + str(i) + "/attention/output/dense/kernel"] = input_model["encoder.transformer." + str(i) + ".self_attn.final_linear.weight"]
        output_model["bert/encoder/layer_" + str(i) + "/attention/output/dense/bias"] = input_model["encoder.transformer." + str(i) + ".self_attn.final_linear.bias"]
        output_model["bert/encoder/layer_" + str(i) + "/attention/output/LayerNorm/gamma"] = input_model["encoder.transformer." + str(i) + ".layer_norm_1.gamma"]
        output_model["bert/encoder/layer_" + str(i) + "/attention/output/LayerNorm/beta"] = input_model["encoder.transformer." + str(i) + ".layer_norm_1.beta"]
        output_model["bert/encoder/layer_" + str(i) + "/intermediate/dense/kernel"] = input_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"]
        output_model["bert/encoder/layer_" + str(i) + "/intermediate/dense/bias"] = input_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.bias"]
        output_model["bert/encoder/layer_" + str(i) + "/output/dense/kernel"] = input_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"]
        output_model["bert/encoder/layer_" + str(i) + "/output/dense/bias"] = input_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.bias"]
        output_model["bert/encoder/layer_" + str(i) + "/output/LayerNorm/gamma"] = input_model["encoder.transformer." + str(i) + ".layer_norm_2.gamma"]
        output_model["bert/encoder/layer_" + str(i) + "/output/LayerNorm/beta"] = input_model["encoder.transformer." + str(i) + ".layer_norm_2.beta"]

    if args.target == "bert":
        output_model["bert/pooler/dense/kernel"] = input_model["target.nsp_linear_1.weight"]
        output_model["bert/pooler/dense/bias"] = input_model["target.nsp_linear_1.bias"]
        output_model["cls/seq_relationship/output_weights"] = input_model["target.nsp_linear_2.weight"]
        output_model["cls/seq_relationship/output_bias"] = input_model["target.nsp_linear_2.bias"]
    output_model["cls/predictions/transform/dense/kernel"] = input_model["target.mlm_linear_1.weight"]
    output_model["cls/predictions/transform/dense/bias"] = input_model["target.mlm_linear_1.bias"]
    output_model["cls/predictions/transform/LayerNorm/gamma"] = input_model["target.layer_norm.gamma"]
    output_model["cls/predictions/transform/LayerNorm/beta"] = input_model["target.layer_norm.beta"]
    output_model["cls/predictions/output_bias"] = input_model["target.mlm_linear_2.bias"]

    tf_vars = []

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
