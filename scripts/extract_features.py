import sys
import os
import torch
import argparse
import torch.nn as nn
import numpy as np

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils.config import load_hyperparam
from uer.utils.tokenizers import *
from uer.model_builder import build_model
from uer.layers import *
from uer.encoders import *
from uer.targets import *
from uer.utils import *

class SequenceEncoder(torch.nn.Module):
    
    def __init__(self, args):
        super(SequenceEncoder, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.pooling = args.pooling
        # Close dropout.
        self.eval()

    def forward(self, src, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)

        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        elif self.pooling == "first":
            output = output[:, 0, :]
        else:
            output = output

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

    # Path options.
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path of the input file.")
    parser.add_argument("--pretrained_model_path", type=str, required=True,
                        help="Path of the pretrained model.")
    parser.add_argument("--vocab_path", type=str, required=True,
                        help="Path of the vocabulary file.")
    parser.add_argument("--spm_model_path", default=None, type=str,
                        help="Path of the sentence piece model.")
    parser.add_argument("--output_path", required=True,
                        help="Path of the output file.")
    parser.add_argument("--config_path", default="models/bert_base_config.json",
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--mask", choices=["fully_visible", "causal"], default="fully_visible",
                        help="Mask type.")
    parser.add_argument("--embedding", choices=["word", "word_pos", "word_pos_seg"], default="word_pos_seg",
                        help="Emebdding type.")
    parser.add_argument("--encoder", choices=["transformer", "rnn", "lstm", "gru", \
                                              "birnn", "bilstm", "bigru", \
                                              "gatedcnn"], \
                                              default="transformer", help="Encoder type.")
    parser.add_argument("--pooling", choices=["first", "last", "max", "mean"], \
                                              default=None, help="Pooling Type.")
    parser.add_argument("--whitening_size", type=int, default=None, help="Output vector size after whitening.")
    parser.add_argument("--layernorm_positioning", choices=["pre", "post"], default="post",
                        help="Layernorm positioning.") 
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--parameter_sharing", action="store_true", help="Parameter sharing.")
    parser.add_argument("--factorized_embedding_parameterization", action="store_true",
                        help="Factorized embedding parameterization.")
    parser.add_argument("--tie_weights", action="store_true",
                        help="Tie the word embedding and softmax weights.")
    parser.add_argument("--remove_embedding_layernorm", action="store_true",
                        help="Remove layernorm on embedding.")
    parser.add_argument("--remove_embedding_layernorm_bias", action="store_true",
                        help="Remove layernorm bias on embedding.")
    parser.add_argument("--remove_transformer_bias", action="store_true",
                        help="Remove bias on transformer layers.")
    parser.add_argument("--feed_forward", choices=["dense", "gated"], default="dense",
                        help="Feed forward type, specific to transformer model.")
    parser.add_argument("--relative_position_embedding", action="store_true",
                        help="Use relative position embedding.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer." 
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )

    args = parser.parse_args()
    args = load_hyperparam(args)

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build and load modeli.
    model = SequenceEncoder(args)
    pretrained_model = torch.load(args.pretrained_model_path)
    model.load_state_dict(pretrained_model, strict=False)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)
    
    # Build tokenizer
    tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)

    dataset = []
    with open(args.input_path, mode="r", encoding="utf-8") as f:
        for line in f:
            tokens = [vocab.get(t) for t in tokenizer.tokenize(line)]
            if len(tokens) == 0:
                continue
            tokens = [args.vocab.get(CLS_TOKEN)] + tokens
            seg = [1] * len(tokens)

            if len(tokens) > args.seq_length:
                tokens = tokens[:args.seq_length]
                seg = seg[:args.seq_length]
            while len(tokens) < args.seq_length:
                tokens.append(PAD_ID)
                seg.append(PAD_ID)
            dataset.append((tokens, seg))
           
    src = torch.LongTensor([e[0] for e in dataset])
    seg = torch.LongTensor([e[1] for e in dataset])

    def batch_loader(batch_size, src, seg):
        instances_num = src.size(0)
        for i in range(instances_num // batch_size):
            src_batch = src[i*batch_size : (i+1)*batch_size]
            seg_batch = seg[i*batch_size : (i+1)*batch_size]
            yield src_batch, seg_batch
        if instances_num > instances_num // batch_size * batch_size:
            src_batch = src[instances_num//batch_size*batch_size:]
            seg_batch = seg[instances_num//batch_size*batch_size:]
            yield src_batch, seg_batch

    feature_vectors = []
    for i, (src_batch, seg_batch) in enumerate(batch_loader(args.batch_size, src, seg)):
        src_batch = src_batch.to(device)
        seg_batch = seg_batch.to(device)
        output = model(src_batch, seg_batch)
        feature_vectors.append(output)
    feature_vectors = torch.cat(feature_vectors, 0)

    # whitening
    if args.pooling is not None and args.whitening_size is not None:
        print("Conduct whitening({}).".format(args.whitening_size))
        whitening = WhiteningHandle(args, feature_vectors)
        feature_vectors = whitening(feature_vectors, args.whitening_size, pt=True)

    torch.save(feature_vectors,args.output_path)
    print("The size of features vectors: {}".format(feature_vectors.size()))
    print("The number of sentences: {}".format(len(feature_vectors)))
    
