"""
  This script provides an example to wrap UER-py for feature extraction.
"""
import sys
import os
import torch
import torch.nn as nn
import argparse
import numpy as np

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.layers import *
from uer.encoders import *
from uer.targets import *
from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.model_loader import load_model
from uer.opts import infer_opts, tokenizer_opts


def batch_loader(batch_size, src, seg):                                                                                             
    instances_num = src.size(0)                                                                                                     
    for i in range(instances_num // batch_size):                                                                                    
        src_batch = src[i * batch_size : (i + 1) * batch_size]                                                                            
        seg_batch = seg[i * batch_size : (i + 1) * batch_size]                                                                            
        yield src_batch, seg_batch                                                                                                  
    if instances_num > instances_num // batch_size * batch_size:                                                                    
        src_batch = src[instances_num // batch_size * batch_size:]                                                                      
        seg_batch = seg[instances_num // batch_size * batch_size:]                                                                      
        yield src_batch, seg_batch


def read_dataset(args, path):
    dataset = []
    PAD_ID = args.tokenizer.vocab.get(PAD_TOKEN)
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            src = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(line))
            if len(src) == 0:
                continue
            src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN]) + src
            seg = [1] * len(src)

            if len(src) > args.seq_length:
                src = src[:args.seq_length]
                seg = seg[:args.seq_length]
            while len(src) < args.seq_length:
                src.append(PAD_ID)
                seg.append(PAD_ID)
            dataset.append((src, seg))
    return dataset


class FeatureExtractor(torch.nn.Module):    
    def __init__(self, args):
        super(FeatureExtractor, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.pooling = args.pooling

    def forward(self, src, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        seg = torch.unsqueeze(seg, dim=-1)
        output = output * seg

        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]

        return output        


class WhiteningHandle(torch.nn.Module):
    """
    Whitening operation.
    @ref: https://github.com/bojone/BERT-whitening/blob/main/demo.py
    """
    def __init__(self, args, vecs):
        super(WhiteningHandle, self).__init__()
        self.kernel, self.bias = self._compute_kernel_bias(vecs)

    def forward(self, vecs, n_components=None, normal=True, pt=True):
        vecs = self._format_vecs_to_np(vecs)
        vecs = self._transform(vecs, n_components)
        vecs = self._normalize(vecs) if normal else vecs
        vecs = torch.tensor(vecs) if pt else vecs
        return vecs

    def _compute_kernel_bias(self, vecs):
        vecs = self._format_vecs_to_np(vecs)
        mu = vecs.mean(axis=0, keepdims=True)
        cov = np.cov(vecs.T)
        u, s, vh = np.linalg.svd(cov)
        W = np.dot(u, np.diag(1 / np.sqrt(s)))
        return W, -mu

    def _transform(self, vecs, n_components):
        w = self.kernel[:, :n_components] \
                if isinstance(n_components, int) else self.kernel
        return (vecs + self.bias).dot(w)

    def _normalize(self, vecs):
        return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

    def _format_vecs_to_np(self, vecs):
        vecs_np = []
        for vec in vecs:
            if isinstance(vec, list):
                vec = np.array(vec)
            elif torch.is_tensor(vec):
                vec = vec.detach().numpy()
            elif isinstance(vec, np.ndarray):
                vec = vec
            else:
                raise Exception('Unknown vec type.')
            vecs_np.append(vec)
        vecs_np = np.array(vecs_np)
        return vecs_np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    parser.add_argument("--pooling", choices=["first", "last", "max", "mean"], \
                                              default="first", help="Pooling Type.")
    parser.add_argument("--whitening_size", type=int, default=None, help="Output vector size after whitening.")

    tokenizer_opts(parser)

    args = parser.parse_args()
    args = load_hyperparam(args)

    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build feature extractor model.
    model = FeatureExtractor(args)
    model = load_model(model, args.load_model_path)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.eval()

    dataset = read_dataset(args, args.test_path)

    src = torch.LongTensor([sample[0] for sample in dataset])
    seg = torch.LongTensor([sample[1] for sample in dataset])

    feature_vectors = []
    for i, (src_batch, seg_batch) in enumerate(batch_loader(args.batch_size, src, seg)):
        src_batch = src_batch.to(device)
        seg_batch = seg_batch.to(device)
        output = model(src_batch, seg_batch)
        feature_vectors.append(output.cpu().detach())
    feature_vectors = torch.cat(feature_vectors, 0)

    # Vector whitening.
    if args.whitening_size is not None:
        whitening = WhiteningHandle(args, feature_vectors)
        feature_vectors = whitening(feature_vectors, args.whitening_size, pt=True)

    print("The size of feature vectors (sentences_num * vector size): {}".format(feature_vectors.shape))
    torch.save(feature_vectors, args.prediction_path)
