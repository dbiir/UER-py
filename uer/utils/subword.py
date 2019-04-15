# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
from uer.utils.constants import *


def word2sub(word_ids, vocab, sub_vocab, subword_type):
    '''
    word_ids: batch_size, seq_length
    '''
    batch_size, seq_length = word_ids.size()
    device = word_ids.device
    word_ids = word_ids.contiguous().view(-1).tolist()
    words = [vocab.i2w[i] for i in word_ids]
    max_length = max([len(w) for w in words])
    sub_ids = torch.zeros((len(words), max_length), dtype=torch.long).to(device)
    for i in range(len(words)):
        for j, c in enumerate(words[i]):
            sub_ids[i, j] = sub_vocab.w2i.get(c, UNK_ID)
    return sub_ids
