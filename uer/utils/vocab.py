# -*- encoding:utf-8 -*-
import os
import torch
from multiprocessing import Pool
from uer.utils.constants import *

class Vocab(object):
    """
    Abstract class, defining member dictionary variables to store tokens.
    Sub-classes must override the `tokenize` mehtod.

    """
    def __init__(self):
        # add pre-defined special tokens
        self.w2i = {} 
        self.i2w = [] 
        self.w2c = {} 
        self.reserved_vocab_path = \
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/reserved_vocab.txt"))
        
    def load(self, vocab_path):
        with open(vocab_path, mode="r", encoding="utf-8") as reader:
            for index, line in enumerate(reader):
                try:
                    w = line.strip().split()[0]
                    self.w2i[w] = index
                    self.i2w.append(w)
                except:
                    self.w2i["???"+str(index)] = index
                    self.i2w.append("???"+str(index))
                    print("Vocabulary file line " + str(index+1) + " has bad format token")
            assert len(self.w2i) == len(self.i2w)
        print("Vocabulary Size: ", len(self))

    def save(self, save_path):
        print("Vocabulary Size: ", len(self))
        with open(save_path, mode="w", encoding="utf-8") as writer:
            for w in self.i2w:
                writer.write(w + "\n")
        print("Vocabulary saving done.")

    def get(self, w):
        return self.w2i.get(w, UNK_ID)
        
    def __len__(self):
        return len(self.i2w)
        
    def worker(self, corpus_path, tokenizer, start, end):
        """ 
        worker that creates vocabulary from corpus[seek_start:seek_end]
        """
        w2i, i2w, w2c = {}, [], {}
        with open(corpus_path, mode="r", encoding="utf-8") as f:
            f.seek(start)
            while True:
                try:
                    line = f.readline()
                except UnicodeDecodeError:
                    # If decode error, skip.
                    continue
                tokens = tokenizer.tokenize(line)
                for t in tokens:
                    if t not in w2i:
                        w2i[t], w2c[t] = len(i2w), 1
                        i2w.append(t)
                    else:
                        w2c[t] += 1
                pos = f.tell()
                if pos >= end:
                    return (w2i, i2w, w2c)
                            
    def union(self, results):
        """ Union vocab in all workers. """
        w2i, i2w, w2c = {}, [], {}
        index = 0
        for res in results:
            w2i_p, i2w_p, w2c_p = res.get()
            for k in i2w_p:
                if k not in w2i:
                    w2i[k] = index
                    i2w.append(k)
                    # udpate count with w2c_p[k]
                    w2c[k] = w2c_p[k]
                    index += 1
                else:
                    w2c[k] += w2c_p[k]
        return (w2i, i2w, w2c)
                    
    def build(self, corpus_path, tokenizer, workers_num=1, min_count=1):
        """ Build vocabulary from the given corpus. """
        print("Start %d workers for building vocabulary..." % workers_num)
        file_size = os.path.getsize(corpus_path)
        pool = Pool(workers_num)
        results = []
        for i in range(workers_num):
            start = i * file_size // workers_num
            end = (i+1) * file_size // workers_num
            res = pool.apply_async(func=self.worker, args=[corpus_path, tokenizer, start, end])
            results.append(res)
        pool.close()
        pool.join()
        
        # Union vocab in all workers.
        w2i,i2w,w2c = self.union(results)

        # Add special symbols and remove low frequency words.
        with open(self.reserved_vocab_path, mode="r", encoding="utf-8") as reader:
            reserved_vocab = [line.strip().split()[0] for line in reader]

        for i, w in enumerate(reserved_vocab):
            self.w2i[w] = i
            self.w2c[w] = -1
        self.i2w = reserved_vocab

        # Sort w2c according to word count.
        sorted_w2c = sorted(w2c.items(), key=lambda item:item[1], reverse=True)

        for w, c in sorted_w2c:
            if c < min_count:
                break
            self.w2c[w] = c
            self.w2i[w] = len(self.i2w)
            self.i2w.append(w)
