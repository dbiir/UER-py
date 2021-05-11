import sys
import os
import torch
import argparse

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.utils.vocab import Vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path of the input model.")
    parser.add_argument("--vocab_path", default=None, type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--spm_model_path", default=None, type=str,
                        help="Path of the sentence piece model.")
    parser.add_argument("--test_path", default=None, type=str,
                        help="Path of the target words file.")
    parser.add_argument("--topn", type=int, default=15)

    args = parser.parse_args()

    if args.spm_model_path:
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("You need to install SentencePiece to use XLNetTokenizer: https://github.com/google/sentencepiece"
                                                    "pip install sentencepiece")
        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(args.spm_model_path)
        vocab = Vocab()
        vocab.i2w = {i: sp_model.IdToPiece(i) for i in range(sp_model.GetPieceSize())}
        vocab.w2i = {sp_model.IdToPiece(i): i for i in range(sp_model.GetPieceSize())}
    else:
        vocab = Vocab()
        vocab.load(args.vocab_path)

    pretrained_model = torch.load(args.load_model_path)
    embedding = pretrained_model["embedding.word_embedding.weight"]

    with open(args.test_path, mode="r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().split()[0]
            if len(word) <= 0:
                continue
            print("Target word: " + word)
            target_embedding = embedding[vocab.w2i[word], :]

            sims = torch.nn.functional.cosine_similarity(target_embedding.view(1, -1), embedding)
            sorted_id = torch.argsort(sims, descending=True)
            for i in sorted_id[1: args.topn+1]:
                print(vocab.i2w[i].strip() + "\t" + str(sims[i].item()))
            print()
