# -*- encoding:utf-8 -*-
import os
import torch
import codecs
import random
import pickle
from multiprocessing import Pool
from uer.utils.constants import *
from uer.utils.misc import count_lines
from uer.utils.seed import set_seed


def mask_seq(src, vocab_size):
    """
    mask input sequence for MLM task
    args:
        src: a list of tokens
        vocab_size: the vocabulary size
    """
    tgt_mlm = []
    for (i, token) in enumerate(src):
        if token == CLS_ID or token == SEP_ID:
            continue
        prob = random.random()
        if prob < 0.15:
            prob /= 0.15
            if prob < 0.8:
                src[i] = MASK_ID
            elif prob < 0.9:
                while True:
                    rdi = random.randint(1, vocab_size-1)
                    if rdi not in [CLS_ID, SEP_ID, MASK_ID]:
                        break
                src[i] = rdi
            tgt_mlm.append((i, token))
    return src, tgt_mlm


def merge_dataset(dataset_path, workers_num):
        # Merge datasets.
        f_writer = open(dataset_path, "wb")
        for i in range(workers_num):
            tmp_dataset_reader = open("dataset-tmp-"+str(i)+".pt", "rb")
            while True:
                tmp_data = tmp_dataset_reader.read(2^20)
                if tmp_data:
                    f_writer.write(tmp_data)
                else:
                    break
            tmp_dataset_reader.close()
            os.remove("dataset-tmp-"+str(i)+".pt")
        f_writer.close()


class Dataset(object):
    def __init__(self, args, vocab, tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.corpus_path = args.corpus_path
        self.dataset_path = args.dataset_path
        self.seq_length = args.seq_length
        self.seed = args.seed

    def build_and_save(self, workers_num):
        """
        Build dataset from the given corpus.
        Start workers_num processes and each process deals with a part of data.
        """
        lines_num = count_lines(self.corpus_path)
        print("Starting %d workers for building datasets ... " % workers_num)
        assert(workers_num >= 1)
        if workers_num == 1:
            self.worker(0, 0, lines_num)
        else:
            pool = Pool(workers_num)
            for i in range(workers_num):
                start = i * lines_num // workers_num
                end = (i+1) * lines_num // workers_num
                pool.apply_async(func=self.worker, args=[i, start, end])
            pool.close()
            pool.join()

        # Merge datasets.
        merge_dataset(self.dataset_path, workers_num)

    def worker(self, proc_id, start, end):
        raise NotImplementedError()


class DataLoader(object):
    def __init__(self, args, dataset_path, batch_size, proc_id, proc_num, shuffle=False):
        self.batch_size = batch_size
        self.instances_buffer_size = args.instances_buffer_size
        self.proc_id = proc_id
        self.proc_num = proc_num
        self.shuffle = shuffle
        self.f_read = open(dataset_path, "rb")
        self.read_count = 0
        self.start = 0
        self.end = 0
        self.buffer = []

    def _fill_buf(self):
        try:
            self.buffer = []
            while True:
                instance = pickle.load(self.f_read)
                self.read_count += 1
                if (self.read_count - 1) % self.proc_num == self.proc_id:
                    self.buffer.append(instance)
                    if len(self.buffer) >= self.instances_buffer_size:
                        break
        except EOFError:
            # Reach file end.
            self.f_read.seek(0)

        if self.shuffle:
            random.shuffle(self.buffer)
        self.start = 0
        self.end = len(self.buffer)

    def _empty(self):
        return self.start >= self.end

    def __del__(self):
        self.f_read.close()


class BertDataset(Dataset):
    """
    Construct dataset for MLM and NSP tasks from the given corpus.
    Each document consists of multiple sentences, 
    and each sentence occupies a single line. 
    Documents in corpus must be separated by empty lines.
    """
    def __init__(self, args, vocab, tokenizer):
        super(BertDataset, self).__init__(args, vocab, tokenizer)
        self.docs_buffer_size = args.docs_buffer_size
        self.dup_factor = args.dup_factor
        self.short_seq_prob = args.short_seq_prob

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        docs_buffer = []
        document = []
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                except:
                    continue
                finally:
                    pos += 1
                if not line.strip():
                    if len(document) >= 1:
                        docs_buffer.append(document)
                    document = []
                    if len(docs_buffer) == self.docs_buffer_size:
                        # Build instances from documents.                    
                        instances = self.build_instances(docs_buffer)
                        # Save instances.
                        for instance in instances:
                            pickle.dump(instance, f_write)
                        # Clear buffer.
                        docs_buffer = []
                        instances = []
                    continue
                sentence = [self.vocab.get(w) for w in self.tokenizer.tokenize(line)]
                if len(sentence) > 0:
                    document.append(sentence)
        
                if pos >= end - 1:
                    if len(docs_buffer) > 0:
                        instances = self.build_instances(docs_buffer)
                        for instance in instances:
                            pickle.dump(instance, f_write)
                    break
        f_write.close()

    def build_instances(self, all_documents):
        instances = []
        for _ in range(self.dup_factor):
            for doc_index in range(len(all_documents)):
                instances.extend(self.create_ins_from_doc(all_documents, doc_index))
        return instances

    def create_ins_from_doc(self, all_documents, document_index):
        document = all_documents[document_index]
        max_num_tokens = self.seq_length - 3
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)
        instances = []
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []
                    is_random_next = 0

                    # Random next
                    if len(current_chunk) == 1 or random.random() < 0.5:
                        is_random_next = 1
                        target_b_length = target_seq_length - len(tokens_a)

                        for _ in range(10):
                            random_document_index = random.randint(0, len(all_documents) - 1)
                            if random_document_index != document_index:
                                break

                        random_document = all_documents[random_document_index]
                        random_start = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break

                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments

                    # Actual next
                    else:
                        is_random_next = 0
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    self.truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                    # assert len(tokens_a) >= 1
                    # assert len(tokens_b) >= 1

                    src = []

                    src.append(CLS_ID)
                    for token in tokens_a:
                        src.append(token)

                    src.append(SEP_ID)

                    seg_pos = [len(src)]

                    for token in tokens_b:
                        src.append(token)
            
                    src.append(SEP_ID)

                    seg_pos.append(len(src))

                    src, tgt_mlm = mask_seq(src, len(self.vocab))
                    
                    while len(src) != self.seq_length:
                        src.append(PAD_ID)

                    instance = (src, tgt_mlm, is_random_next, seg_pos)
                    instances.append(instance)
                current_chunk = []
                current_length = 0
            i += 1
        return instances

    def truncate_seq_pair(self, tokens_a, tokens_b, max_num_tokens):
        """ truncate sequence pair to specific length """
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_num_tokens:
                break
                
            trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
            # assert len(trunc_tokens) >= 1

            if random.random() < 0.5:
                del trunc_tokens[0]
            else:
                trunc_tokens.pop()


class BertDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size
        
            src = []
            tgt_mlm = []
            is_next = []
            seg = []

            masked_words_num = 0
            for ins in instances:
                masked_words_num += len(ins[1])
            if masked_words_num == 0:
                continue
            
            for ins in instances:
                src.append(ins[0])
                tgt_mlm.append([0]*len(ins[0]))
                for mask in ins[1]:
                    tgt_mlm[-1][mask[0]] = mask[1]
                is_next.append(ins[2])
                seg.append([1]*ins[3][0] + [2]*(ins[3][1]-ins[3][0]) + [PAD_ID]*(len(ins[0])-ins[3][1]))

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_mlm), \
                torch.LongTensor(is_next), \
                torch.LongTensor(seg)


class LmDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                except:
                    continue
                finally:
                    pos += 1

                src = [self.vocab.get(w) for w in self.tokenizer.tokenize(line)]
                tgt = src[1:]
                src = src[:-1]
                seg = [1] * len(src)
                if len(src) >= self.seq_length:
                    src = src[:self.seq_length]
                    tgt = tgt[:self.seq_length]
                    seg = seg[:self.seq_length]
                else:
                    while len(src) != self.seq_length:
                        src.append(PAD_ID)
                        tgt.append(PAD_ID)
                        seg.append(PAD_ID)

                pickle.dump((src, tgt, seg), f_write)

                if pos >= end - 1:
                    break

        f_write.close()


class LmDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size
        
            src = []
            tgt = []
            seg = []

            for ins in instances:
                src.append(ins[0])
                tgt.append(ins[1])
                seg.append(ins[2])

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)


class BilmDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                except:
                    continue
                finally:
                    pos += 1

                src = [self.vocab.get(w) for w in self.tokenizer.tokenize(line)]
                if len(src) < 1:
                    continue
                tgt_forward = src[1:] + [SEP_ID]
                tgt_backward = [CLS_ID] + src[:-1]
                seg = [1] * len(src)
                if len(src) >= self.seq_length:
                    src = src[:self.seq_length]
                    tgt_forward = tgt_forward[:self.seq_length]
                    tgt_backward = tgt_backward[:self.seq_length]
                    seg = seg[:self.seq_length]
                else:
                    while len(src) != self.seq_length:
                        src.append(PAD_ID)
                        tgt_forward.append(PAD_ID)
                        tgt_backward.append(PAD_ID)
                        seg.append(PAD_ID)
                
                pickle.dump((src, tgt_forward, tgt_backward, seg), f_write)

                if pos >= end - 1:
                    break

        f_write.close()


class BilmDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size
        
            src = []
            tgt_forward = []
            tgt_backward = []
            seg = []

            for ins in instances:
                src.append(ins[0])
                tgt_forward.append(ins[1])
                tgt_backward.append(ins[2])
                seg.append(ins[3])

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_forward), \
                torch.LongTensor(tgt_backward), \
                torch.LongTensor(seg)


class ClsDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        pos = 0
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                except:
                    continue
                finally:
                    pos += 1

                line = line.strip().split('\t')
                if len(line) == 2:
                    label = int(line[0])
                    text = " ".join(line[1:])
                    src = [self.vocab.get(t) for t in self.tokenizer.tokenize(text)]
                    src = [CLS_ID] + src
                    tgt = label
                    seg = [1] * len(src)
                    if len(src) >= self.seq_length:
                        src = src[:self.seq_length]
                        seg = seg[:self.seq_length]
                    else:
                        while len(src) != self.seq_length:
                            src.append(PAD_ID)
                            seg.append(PAD_ID)
                    pickle.dump((src, tgt, seg), f_write)
                elif len(line) == 3: # For sentence pair input.
                    label = int(line[0])
                    text_a, text_b = line[1], line[2]

                    src_a = [self.vocab.get(t) for t in self.tokenizer.tokenize(text_a)]
                    src_a = [CLS_ID] + tokens_a + [SEP_ID]
                    src_b = [vocab.get(t) for t in tokenizer.tokenize(text_b)]
                    src_b = tokens_b + [SEP_ID]

                    src = src_a + src_b
                    seg = [1] * len(src_a) + [2] * len(src_b)

                    if len(src) >= self.seq_length:
                        src = src[:self.seq_length]
                        seg = seg[:self.seq_length]
                    else:
                        while len(src) != self.seq_length:
                            src.append(PAD_ID)
                            seg.append(PAD_ID)
                    pickle.dump((src, tgt, seg), f_write)
                else:
                    pass

                if pos >= end - 1:
                    break

        f_write.close()


class ClsDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size
        
            src = []
            tgt = []
            seg = []

            for ins in instances:
                src.append(ins[0])
                tgt.append(ins[1])
                seg.append(ins[2])

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)


class MlmDataset(Dataset):
    def __init__(self, args, vocab, tokenizer):
        super(MlmDataset, self).__init__(args, vocab, tokenizer)
        self.dup_factor = args.dup_factor

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        f_write = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        for _ in range(self.dup_factor):
            pos = 0
            with open(self.corpus_path, mode="r", encoding="utf-8") as f:
                while pos < start:
                    try:
                        f.readline()
                    except:
                        continue
                    finally:
                        pos += 1
                while True:
                    try:
                        line = f.readline()
                    except:
                        continue
                    finally:
                        pos += 1

                    src = [self.vocab.get(w) for w in self.tokenizer.tokenize(line)]

                    if len(src) > self.seq_length:
                        src = src[:self.seq_length]
                    seg = [1] * len(src)

                    src, tgt = mask_seq(src, len(self.vocab))

                    while len(src) != self.seq_length:
                        src.append(PAD_ID)
                        seg.append(PAD_ID)

                    pickle.dump((src, tgt, seg), f_write)

                    if pos >= end - 1:
                        break

        f_write.close()


class MlmDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []

            masked_words_num = 0
            for ins in instances:
                masked_words_num += len(ins[1])
            if masked_words_num == 0:
                continue

            for ins in instances:
                src.append(ins[0])
                seg.append(ins[2])
                tgt.append([0]*len(ins[0]))
                for mask in ins[1]:
                    tgt[-1][mask[0]] = mask[1]

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)
