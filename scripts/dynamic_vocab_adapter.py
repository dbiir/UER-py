# -*- encoding:utf-8 -*-
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

from bert.utils.vocab import Vocab


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
    new_model = collections.OrderedDict()
 
    for k in old_model.keys():
        if 'module.' in k:
            print("Loaded model is of multi-gpu version.")
            word_embedding_key = 'module.embedding.word_embedding.weight'    
            mlm_key = 'module.target.output_mlm.weight'
            mlm_bias_key = 'module.target.output_mlm.bias'
        else:
            print("Loaded model is of single-gpu version.")
            word_embedding_key = 'embedding.word_embedding.weight'    
            mlm_key = 'target.output_mlm.weight'
            mlm_bias_key = 'target.output_mlm.bias'
        break
    
    # Fit in parameters that would not be modified.
    for k, v in old_model.items():
        if k not in [word_embedding_key,mlm_key,mlm_bias_key]:
            new_model[k] = v

    # Get word embedding, mlm, and mlm bias variables.
    old_word_embedding = old_model.get(word_embedding_key).data.numpy()
    old_mlm = old_model.get(mlm_key).data.numpy()
    old_bias = old_model.get(mlm_bias_key).data.numpy()

    # Initialize.
    new_word_embedding = np.random.normal(0, 0.02, [len(new_vocab), old_word_embedding.shape[1]])
    new_mlm = np.random.normal(0, 0.02, [len(new_vocab), old_mlm.shape[1]])
    new_bias = np.random.normal(0, 0.02, [len(new_vocab)])

    # Put corresponding parameters into the new model.
    for i, w in enumerate(new_vocab.i2w):
        if w in old_vocab.w2i:
            old_w_index = old_vocab.w2i[w]
            new_word_embedding[i] = old_word_embedding[old_w_index]
            new_mlm[i] = old_mlm[old_w_index]
            new_bias[i] = old_bias[old_w_index]

    new_model[word_embedding_key] = torch.tensor(new_word_embedding, dtype=torch.float32)
    new_model[mlm_key] = torch.tensor(new_mlm, dtype=torch.float32)
    new_model[mlm_bias_key] = torch.tensor(new_bias, dtype=torch.float32)    

    print("Output adapted new model.")
    torch.save(new_model, args.new_model_path)
