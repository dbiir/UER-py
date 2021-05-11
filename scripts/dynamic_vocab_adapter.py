"""
Modify model's embedding and softmax layers according to the vocabulary.
"""
import sys
import os
import numpy as np
import torch
import argparse
import collections

bert_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(bert_dir)

from uer.utils.vocab import Vocab


def adapter(old_model, old_vocab, new_vocab):
    new_model = collections.OrderedDict()

    embedding_key = "embedding.word_embedding.weight"
    softmax_key = "target.mlm_linear_2.weight"
    softmax_bias_key = "target.mlm_linear_2.bias"

    # Fit in parameters that would not be modified.
    tensor_name = []
    for k, v in old_model.items():
        tensor_name.append(k)
        if k not in [embedding_key, softmax_key, softmax_bias_key]:
            new_model[k] = v
    bool = softmax_key in tensor_name
    # Get word embedding, mlm, and mlm bias variables.
    old_embedding = old_model.get(embedding_key).data.numpy()
    if bool:
        old_softmax = old_model.get(softmax_key).data.numpy()
        old_softmax_bias = old_model.get(softmax_bias_key).data.numpy()

    # Initialize.
    new_embedding = np.random.normal(0, 0.02, [len(new_vocab), old_embedding.shape[1]])
    if bool:
        new_softmax = np.random.normal(0, 0.02, [len(new_vocab), old_softmax.shape[1]])
        new_softmax_bias = np.random.normal(0, 0.02, [len(new_vocab)])

    # Put corresponding parameters into the new model.
    for i, w in enumerate(new_vocab.i2w):
        if w in old_vocab.w2i:
            old_w_index = old_vocab.w2i[w]
            new_embedding[i] = old_embedding[old_w_index]
            if bool:
                new_softmax[i] = old_softmax[old_w_index]
                new_softmax_bias[i] = old_softmax_bias[old_w_index]

    new_model[embedding_key] = torch.tensor(new_embedding, dtype=torch.float32)
    if bool:
        new_model[softmax_key] = torch.tensor(new_softmax, dtype=torch.float32)
        new_model[softmax_bias_key] = torch.tensor(new_softmax_bias, dtype=torch.float32)
    return new_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input options.
    parser.add_argument("--old_model_path", type=str)
    parser.add_argument("--old_vocab_path", type=str)
    parser.add_argument("--new_vocab_path", type=str)

    # Output options.
    parser.add_argument("--new_model_path", type=str)

    args = parser.parse_args()
    old_vocab = Vocab()
    old_vocab.load(args.old_vocab_path)
    new_vocab = Vocab()
    new_vocab.load(args.new_vocab_path)

    old_model = torch.load(args.old_model_path, map_location="cpu")

    new_model = adapter(old_model, old_vocab, new_vocab)
    print("Output adapted new model.")
    torch.save(new_model, args.new_model_path)