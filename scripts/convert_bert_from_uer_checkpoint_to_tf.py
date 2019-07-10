# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert Huggingface Pytorch checkpoint to Tensorflow checkpoint."""

import os
import numpy as np
import tensorflow as tf
import torch
import argparse
import collections

def uer_to_huggingface(args):

    # parser.add_argument("--output_model_path", type=str, default="pytorch_model.bin",
    #                     help=".")

    #args = parser.parse_args()
    path = args.pytorch_model_path

    input_model = torch.load(path)

    output_model = collections.OrderedDict()

    output_model["embeddings.word_embeddings.weight"] = input_model["embedding.word_embedding.weight"]
    output_model["embeddings.position_embeddings.weight"] = input_model["embedding.position_embedding.weight"]
    output_model["embeddings.token_type_embeddings"] = input_model["embedding.segment_embedding.weight"][1:, :]
    output_model["embeddings.LayerNorm.beta"] = input_model["embedding.layer_norm.beta"]
    output_model["embeddings.LayerNorm.gamma"] = input_model["embedding.layer_norm.gamma"]

    for i in range(12):
        output_model["encoder.layer." + str(i) + ".attention.self.query.weight"] = input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.weight"]
        output_model["encoder.layer." + str(i) + ".attention.self.query.bias"] = input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.bias"]
        output_model["encoder.layer." + str(i) + ".attention.self.key.weight"] = input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.weight"]
        output_model["encoder.layer." + str(i) + ".attention.self.key.bias"] = input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.bias"]
        output_model["encoder.layer." + str(i) + ".attention.self.value.weight"] = input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.weight"]
        output_model["encoder.layer." + str(i) + ".attention.self.value.bias"] = input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.bias"]
        output_model["encoder.layer." + str(i) + ".attention.output.dense.weight"] = input_model["encoder.transformer." + str(i) + ".self_attn.final_linear.weight"]
        output_model["encoder.layer." + str(i) + ".attention.output.dense.bias"] = input_model["encoder.transformer."+ str(i) + ".self_attn.final_linear.bias"]
        output_model["encoder.layer." + str(i) + ".attention.output.LayerNorm.gamma"] = input_model["encoder.transformer." + str(i) + ".layer_norm_1.gamma"]
        output_model["encoder.layer." + str(i) + ".attention.output.LayerNorm.beta"] = input_model["encoder.transformer." + str(i) + ".layer_norm_1.beta"]
        output_model["encoder.layer." + str(i) + ".intermediate.dense.weight"] = input_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"]
        output_model["encoder.layer." + str(i) + ".intermediate.dense.bias"] = input_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.bias"]
        output_model["encoder.layer." + str(i) + ".output.dense.weight"] = input_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"]
        output_model["encoder.layer." + str(i) + ".output.dense.bias"] = input_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.bias"]
        output_model["encoder.layer." + str(i) + ".output.LayerNorm.gamma"] = input_model["encoder.transformer." + str(i) + ".layer_norm_2.gamma"]
        output_model["encoder.layer." + str(i) + ".output.LayerNorm.beta"] = input_model["encoder.transformer." + str(i) + ".layer_norm_2.beta"]

    output_model["pooler.dense.weight"] = input_model["target.nsp_linear_1.weight"]
    output_model["pooler.dense.bias"] = input_model["target.nsp_linear_1.bias"]

    return output_model



def convert_pytorch_checkpoint_to_tf(input_model, ckpt_dir: str):
    """
    :param model:BertModel Pytorch model instance to be converted
    :param ckpt_dir: Tensorflow model directory
    :param model_name: model name
    :return:
    Currently supported HF models:
        Y BertModel
        N BertForMaskedLM
        N BertForPreTraining
        N BertForMultipleChoice
        N BertForNextSentencePrediction
        N BertForSequenceClassification
        N BertForQuestionAnswering
    """

    tensors_to_transopse = (
        "dense.weight",
        "attention.self.query",
        "attention.self.key",
        "attention.self.value"
    )

    var_map = (
        ('layer.', 'layer_'),
        ('word_embeddings.weight', 'word_embeddings'),
        ('position_embeddings.weight', 'position_embeddings'),
        ('segment_embeddings.weight', 'token_type_embeddings'),
        ('.', '/'),
        ('layer_norm.gamma', 'LayerNorm/gamma'),
        ('layer_norm.beta', 'LayerNorm/beta'),
        ('weight', 'kernel')
    )

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    session = tf.Session()
    state_dict = input_model
    tf_vars = []

    def to_tf_var_name(name: str):
        for patt, repl in iter(var_map):
            name = name.replace(patt, repl)
        return 'bert/{}'.format(name)

    def assign_tf_var(tensor: np.ndarray, name: str):
        tmp_var = tf.Variable(initial_value=tensor)
        tf_var = tf.get_variable(dtype=tmp_var.dtype, shape=tmp_var.shape, name=name)
        op = tf.assign(ref=tf_var, value=tmp_var)
        session.run(tf.variables_initializer([tmp_var, tf_var]))
        session.run(fetches=[op, tf_var])
        return tf_var

    for var_name in state_dict:
        tf_name = to_tf_var_name(var_name)
        torch_tensor = state_dict[var_name].cpu().numpy()
        if any([x in var_name for x in tensors_to_transopse]):
            torch_tensor = torch_tensor.T
        tf_tensor = assign_tf_var(tensor=torch_tensor, name=tf_name)
        tf_vars.append(tf_tensor)
        print("{0}{1}initialized".format(tf_name, " " * (60 - len(tf_name))))

    saver = tf.train.Saver(tf_vars)
    saver.save(session, os.path.join(ckpt_dir, "bert_chinese.ckpt"))


def main(raw_args=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pytorch_model_path",
                        type=str,
                        required=True,
                        help="/path/to/<pytorch-model-name>.bin")
    parser.add_argument("--tf_cache_dir",
                        type=str,
                        required=True,
                        help="Directory in which to save tensorflow model")
    args = parser.parse_args(raw_args)

    tmp_model = uer_to_huggingface(args)

    convert_pytorch_checkpoint_to_tf(
        tmp_model,
        ckpt_dir=args.tf_cache_dir,
    )


if __name__ == "__main__":
    main()