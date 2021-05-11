"""
  This script provides an example to wrap UER-py for embedding extraction.
"""
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
    parser.add_argument("--word_embedding_path", default=None, type=str,
                        help="Path of the output word embedding.")

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
    else:
        vocab = Vocab()
        vocab.load(args.vocab_path)

    pretrained_model = torch.load(args.load_model_path)
    embedding = pretrained_model["embedding.word_embedding.weight"]

    with open(args.word_embedding_path, mode="w", encoding="utf-8") as f:
        head = str(list(embedding.size())[0]) + " " + str(list(embedding.size())[1]) + "\n"
        f.write(head)

        for i in range(len(vocab.i2w)):
            word = vocab.i2w[i]
            word_embedding = embedding[vocab.get(word), :]
            word_embedding = word_embedding.cpu().numpy().tolist()
            line = str(word)
            for j in range(len(word_embedding)):
                line = line + " " + str(word_embedding[j])
            line += "\n"
            f.write(line)
