import sys
import os
import torch
import argparse
import numpy as np

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.layers import *
from uer.encoders import *
from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.utils.vocab import Vocab
from uer.opts import model_opts, tokenizer_opts


class SequenceEncoder(torch.nn.Module):
    def __init__(self, args):
        super(SequenceEncoder, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)

    def forward(self, src, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)

        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    model_opts(parser)

    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path of the input model.")
    parser.add_argument("--vocab_path", default=None, type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--cand_vocab_path", default=None, type=str,
                        help="Path of the candidate vocabulary file.")
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path of the target word an its context.")
    parser.add_argument("--config_path", default="models/bert/base_config.json", type=str,
                        help="Path of the config file.")

    tokenizer_opts(parser)

    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")

    parser.add_argument("--topn", type=int, default=15)

    args = parser.parse_args()
    args = load_hyperparam(args)

    args.spm_model_path = None

    vocab = Vocab()
    vocab.load(args.vocab_path)

    cand_vocab = Vocab()
    cand_vocab.load(args.cand_vocab_path)

    args.tokenizer = str2tokenizer[args.tokenizer](args)

    model = SequenceEncoder(args)    
 
    pretrained_model = torch.load(args.load_model_path)
    model.load_state_dict(pretrained_model, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.eval()

    PAD_ID = args.tokenizer.vocab.get(PAD_TOKEN)
    with open(args.test_path, mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split("\t")
            if len(line) != 2:
                continue
            target_word, context = line[0], line[1]
            print("Original sentence: " + context)
            print("Target word: " + target_word)
            src = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(context))
            seg = [1] * len(src)
            if len(src) > args.seq_length:
                src = src[:args.seq_length]
                seg = seg[:args.seq_length]
            while len(src) < args.seq_length:
                src.append(PAD_ID)
                seg.append(PAD_ID)

            target_word_id = vocab.get(target_word)
            if target_word_id in src:
                position = src.index(target_word_id)
            else:
                print("The target word is not in the sentence.")
                continue

            output = model(torch.LongTensor([src]).to(device), torch.LongTensor([seg]).to(device))
            output = output.cpu().data.numpy()
            output = output.reshape([args.seq_length, -1])
            target_embedding = output[position, :]
            target_embedding = target_embedding.reshape(1, -1).astype("float")

            cand_words_batch, cand_embeddings = [], []
            for i, word in enumerate(cand_vocab.i2w):
                cand_words_batch.append(vocab.w2i.get(word))
                if len(cand_words_batch) == args.batch_size or i == (len(cand_vocab.i2w)-1):
                    src_batch = torch.LongTensor([src] * len(cand_words_batch))
                    seg_batch = [seg] * len(cand_words_batch)
                    src_batch[:, position] = torch.LongTensor(cand_words_batch)
                    output = model(torch.LongTensor(src_batch).to(device), torch.LongTensor(seg_batch).to(device))
                    output = output.cpu().data.numpy()
                    output = np.reshape(output, (len(output), args.seq_length, -1))
                    cand_embeddings.extend(output[:, position, :].tolist())
                    cand_words_batch = []

            sims = torch.nn.functional.cosine_similarity(torch.FloatTensor(target_embedding), \
                                                         torch.FloatTensor(cand_embeddings))
           
            sorted_ids = torch.argsort(sims, descending=True)
            for j in sorted_ids[1: args.topn + 1]:
                print(cand_vocab.i2w[j].strip() + "\t" + str(sims[j].item()))
