# -*- encoding:utf-8 -*-
import os
import torch
from multiprocessing import Pool
from uer.utils.constants import *
from uer.utils.misc import count_lines


class Vocab(object):
    """
    """
    def __init__(self):
        self.w2i = {} 
        self.i2w = [] 
        self.w2c = {} 
        self.reserved_vocab_path = \
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/reserved_vocab.txt"))
        
    def load(self, vocab_path, is_quiet=False):
        with open(vocab_path, mode="r", encoding="utf-8") as reader:
            for index, line in enumerate(reader):
                w = line.strip("\n").split()[0] if line.strip() else line.strip("\n")
                self.w2i[w] = index
                self.i2w.append(w)
        if not is_quiet:
            print("Vocabulary size: ", len(self))

    def save(self, save_path):
        print("Vocabulary size: ", len(self))
        with open(save_path, mode="w", encoding="utf-8") as f:
            for w in self.i2w:
                f.write(w + "\n")
        print("Vocabulary saving done.")

    def get(self, w):
        return self.w2i[w]
        
    def __len__(self):
        return len(self.i2w)
        
    def worker(self, corpus_path, tokenizer, start, end):
        """ 
        Worker that creates vocabulary from corpus[start:end].
        """
        w2i, i2w, w2c = {}, [], {}
        pos = 0
        with open(corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                tokens = tokenizer.tokenize(line, use_vocab=False)
                for t in tokens:
                    if t not in w2i:
                        w2i[t], w2c[t] = len(i2w), 1
                        i2w.append(t)
                    else:
                        w2c[t] += 1
                if pos >= end - 1:
                    return (w2i, i2w, w2c)
                            
    def union(self, vocab_list):
        """ Union vocab in all workers. """
        w2i, i2w, w2c = {}, [], {}
        index = 0
        for v_p in vocab_list:
            w2i_p, i2w_p, w2c_p = v_p.get()
            for w in i2w_p:
                if w not in w2i:
                    w2i[w], w2c[w] = len(i2w), w2c_p[w]
                    i2w.append(w)
                else:
                    w2c[w] += w2c_p[w]
        return (w2i, i2w, w2c)
                    
    def build(self, corpus_path, tokenizer, workers_num=1, min_count=1):
        """ Build vocabulary from the given corpus. """
        print("Start %d workers for building vocabulary..." % workers_num)
        lines_num = count_lines(corpus_path)
        pool = Pool(workers_num)
        vocab_list = []
        for i in range(workers_num):
            start = i * lines_num // workers_num
            end = (i+1) * lines_num // workers_num
            vocab_list.append((pool.apply_async(func=self.worker, args=[corpus_path, tokenizer, start, end])))
        pool.close()
        pool.join()
        
        # Union vocab in all workers.
        w2i, i2w, w2c = self.union(vocab_list)
        # Sort w2c according to word count.
        sorted_w2c = sorted(w2c.items(), key=lambda item:item[1], reverse=True)

        # Add special symbols and remove low frequency words.
        with open(self.reserved_vocab_path, mode="r", encoding="utf-8") as reader:
            self.i2w = [line.strip().split()[0] for line in reader]

        for i, w in enumerate(self.i2w):
            self.w2i[w] = i
            self.w2c[w] = -1

        for w, c in sorted_w2c:
            if c < min_count:
                break
            if w not in self.w2i:
                self.w2i[w], self.w2c[w] = len(self.i2w), c
                self.i2w.append(w)           
