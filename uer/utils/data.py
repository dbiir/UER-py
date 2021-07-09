import os
import random
import pickle
import torch
from multiprocessing import Pool
from uer.utils.constants import *
from uer.utils.tokenizers import *
from uer.utils.misc import count_lines
from uer.utils.seed import set_seed
from uer.utils.mask import mask_seq


def merge_dataset(dataset_path, workers_num):
    # Merge datasets.
    dataset_writer = open(dataset_path, "wb")
    for i in range(workers_num):
        tmp_dataset_reader = open("dataset-tmp-" + str(i) + ".pt", "rb")
        while True:
            tmp_data = tmp_dataset_reader.read(2**20)
            if tmp_data:
                dataset_writer.write(tmp_data)
            else:
                break
        tmp_dataset_reader.close()
        os.remove("dataset-tmp-" + str(i) + ".pt")
    dataset_writer.close()


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """ truncate sequence pair to specific length """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b

        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


class Dataset(object):
    def __init__(self, args, vocab, tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.corpus_path = args.corpus_path
        self.dataset_path = args.dataset_path
        self.seq_length = args.seq_length
        self.seed = args.seed
        self.dynamic_masking = args.dynamic_masking
        self.whole_word_masking = args.whole_word_masking
        self.span_masking = args.span_masking
        self.span_geo_prob = args.span_geo_prob
        self.span_max_length = args.span_max_length
        self.docs_buffer_size = args.docs_buffer_size
        self.dup_factor = args.dup_factor

    def build_and_save(self, workers_num):
        """
        Build dataset from the given corpus.
        Start workers_num processes and each process deals with a part of data.
        """
        lines_num = count_lines(self.corpus_path)
        print("Starting %d workers for building datasets ... " % workers_num)
        assert (workers_num >= 1)
        if workers_num == 1:
            self.worker(0, 0, lines_num)
        else:
            pool = Pool(workers_num)
            for i in range(workers_num):
                start = i * lines_num // workers_num
                end = (i + 1) * lines_num // workers_num
                pool.apply_async(func=self.worker, args=[i, start, end])
            pool.close()
            pool.join()

        # Merge datasets.
        merge_dataset(self.dataset_path, workers_num)

    def worker(self, proc_id, start, end):
        raise NotImplementedError()


class DataLoader(object):
    def __init__(self, args, dataset_path, batch_size, proc_id, proc_num, shuffle=False):
        self.tokenizer = args.tokenizer
        self.batch_size = batch_size
        self.instances_buffer_size = args.instances_buffer_size
        self.proc_id = proc_id
        self.proc_num = proc_num
        self.shuffle = shuffle
        self.dataset_reader = open(dataset_path, "rb")
        self.read_count = 0
        self.start = 0
        self.end = 0
        self.buffer = []
        self.vocab = args.vocab
        self.whole_word_masking = args.whole_word_masking
        self.span_masking = args.span_masking
        self.span_geo_prob = args.span_geo_prob
        self.span_max_length = args.span_max_length

    def _fill_buf(self):
        try:
            self.buffer = []
            while True:
                instance = pickle.load(self.dataset_reader)
                self.read_count += 1
                if (self.read_count - 1) % self.proc_num == self.proc_id:
                    self.buffer.append(instance)
                    if len(self.buffer) >= self.instances_buffer_size:
                        break
        except EOFError:
            # Reach file end.
            self.dataset_reader.seek(0)

        if self.shuffle:
            random.shuffle(self.buffer)
        self.start = 0
        self.end = len(self.buffer)

    def _empty(self):
        return self.start >= self.end

    def __del__(self):
        self.dataset_reader.close()


class BertDataset(Dataset):
    """
    Construct dataset for MLM and NSP tasks from the given corpus.
    Each document consists of multiple sentences,
    and each sentence occupies a single line.
    Documents in corpus must be separated by empty lines.
    """

    def __init__(self, args, vocab, tokenizer):
        super(BertDataset, self).__init__(args, vocab, tokenizer)
        self.short_seq_prob = args.short_seq_prob

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        docs_buffer = []
        document = []
        pos = 0
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                if pos >= end:
                    if len(docs_buffer) > 0:
                        instances = self.build_instances(docs_buffer)
                        for instance in instances:
                            pickle.dump(instance, dataset_writer)
                    break

                if not line.strip():
                    if len(document) >= 1:
                        docs_buffer.append(document)
                    document = []
                    if len(docs_buffer) == self.docs_buffer_size:
                        # Build instances from documents.
                        instances = self.build_instances(docs_buffer)
                        # Save instances.
                        for instance in instances:
                            pickle.dump(instance, dataset_writer)
                        # Clear buffer.
                        docs_buffer = []
                    continue
                sentence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line))
                if len(sentence) > 0:
                    document.append(sentence)

        dataset_writer.close()

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

                    else:
                        is_random_next = 0
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))

                    while len(src) != self.seq_length:
                        src.append(self.vocab.get(PAD_TOKEN))

                    if not self.dynamic_masking:
                        src, tgt_mlm = mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                        instance = (src, tgt_mlm, is_random_next, seg_pos)
                    else:
                        instance = (src, is_random_next, seg_pos)

                    instances.append(instance)
                current_chunk = []
                current_length = 0
            i += 1
        return instances


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
                if len(ins) == 4:
                    src.append(ins[0])
                    masked_words_num += len(ins[1])
                    tgt_mlm.append([0] * len(ins[0]))
                    for mask in ins[1]:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    is_next.append(ins[2])
                    seg.append([1] * ins[3][0] + [2] * (ins[3][1] - ins[3][0]) + [0] * (len(ins[0]) - ins[3][1]))
                else:
                    src_single, tgt_mlm_single = mask_seq(ins[0], self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                    masked_words_num += len(tgt_mlm_single)
                    src.append(src_single)
                    tgt_mlm.append([0] * len(ins[0]))
                    for mask in tgt_mlm_single:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    is_next.append(ins[1])
                    seg.append([1] * ins[2][0] + [2] * (ins[2][1] - ins[2][0]) + [0] * (len(ins[0]) - ins[2][1]))

            if masked_words_num == 0:
                continue

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_mlm), \
                torch.LongTensor(is_next), \
                torch.LongTensor(seg)


class MlmDataset(Dataset):
    def __init__(self, args, vocab, tokenizer):
        super(MlmDataset, self).__init__(args, vocab, tokenizer)
        self.full_sentences = args.full_sentences

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        docs_buffer = []
        for _ in range(self.dup_factor):
            pos = 0
            with open(self.corpus_path, mode="r", encoding="utf-8") as f:
                while pos < start:
                    f.readline()
                    pos += 1
                while True:
                    line = f.readline()
                    pos += 1

                    document = [self.vocab.get(CLS_TOKEN)] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line)) + [self.vocab.get(SEP_TOKEN)]

                    if self.full_sentences:
                        if len(document) > 0:
                            docs_buffer.append(document)
                        if len(docs_buffer) == self.docs_buffer_size:
                            # Build instances from documents.
                            all_documents = self.concatenate_docs(docs_buffer)
                            instances = self.build_instances(all_documents)
                            # Save instances.
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)
                            # Clear buffer.
                            docs_buffer = []
                        if pos >= end:
                            if len(docs_buffer) > 0:
                                all_documents = self.concatenate_docs(docs_buffer)
                                instances = self.build_instances(all_documents)
                                # Save instances.
                                for instance in instances:
                                    pickle.dump(instance, dataset_writer)
                            break
                    else:
                        if len(document) > 0:
                            instances = self.build_instances(document)
                            # Save instances.
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)

                    if pos >= end:
                        break

        dataset_writer.close()

    def concatenate_docs(self, docs_buffer):
        all_documents = []
        for i in range(len(docs_buffer)):
            all_documents += docs_buffer[i]
        return all_documents

    def build_instances(self, all_documents):
        instances = []
        instances_num = len(all_documents) // self.seq_length
        for i in range(instances_num):
            src = all_documents[i * self.seq_length: (i + 1) * self.seq_length]
            seg_pos = [len(src)]

            if not self.dynamic_masking:
                src, tgt = mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                instance = (src, tgt, seg_pos)
            else:
                instance = (src, seg_pos)

            instances.append(instance)

        src = all_documents[instances_num * self.seq_length:]
        seg_pos = [len(src)]

        while len(src) != self.seq_length:
            src.append(self.vocab.get(PAD_TOKEN))

        if not self.dynamic_masking:
            src, tgt = mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
            instance = (src, tgt, seg_pos)
        else:
            instance = (src, seg_pos)

        instances.append(instance)
        return instances


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
                if len(ins) == 3:
                    src.append(ins[0])
                    masked_words_num += len(ins[1])
                    tgt.append([0] * len(ins[0]))
                    for mask in ins[1]:
                        tgt[-1][mask[0]] = mask[1]
                    seg.append([1] * ins[2][0] + [0] * (len(ins[0]) - ins[2][0]))
                else:
                    src_single, tgt_single = mask_seq(ins[0], self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                    masked_words_num += len(tgt_single)
                    src.append(src_single)
                    tgt.append([0] * len(ins[0]))
                    for mask in tgt_single:
                        tgt[-1][mask[0]] = mask[1]
                    seg.append([1] * ins[1][0] + [0] * (len(ins[0]) - ins[1][0]))

            if masked_words_num == 0:
                continue

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)


class AlbertDataset(Dataset):
    """
    Construct dataset for MLM and SOP tasks from the given corpus.
    Each document consists of multiple sentences,
    and each sentence occupies a single line.
    Documents in corpus must be separated by empty lines.
    """

    def __init__(self, args, vocab, tokenizer):
        super(AlbertDataset, self).__init__(args, vocab, tokenizer)
        self.short_seq_prob = args.short_seq_prob

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        document = []
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        for _ in range(self.dup_factor):
            pos = 0
            with open(self.corpus_path, mode="r", encoding="utf-8") as f:
                while pos < start:
                    f.readline()
                    pos += 1
                while True:
                    line = f.readline()
                    pos += 1
                    if not line.strip():
                        if len(document) >= 1:
                            instances = self.build_instances(document)
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)
                        document = []
                    sentence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line))
                    if len(sentence) > 0:
                        document.append(sentence)
                    if pos >= end:
                        if len(document) >= 1:
                            instances = self.build_instances(document)
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)
                        break
        dataset_writer.close()

    def build_instances(self, document):
        instances = []
        instances.extend(self.create_ins_from_doc(document))
        return instances

    def create_ins_from_doc(self, document):
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
                    is_wrong_order = 0
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                    if random.random() < 0.5:
                        is_wrong_order = 1
                        tmp = tokens_a
                        tokens_a = tokens_b
                        tokens_b = tmp

                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))

                    while len(src) != self.seq_length:
                        src.append(self.vocab.get(PAD_TOKEN))

                    if not self.dynamic_masking:
                        src, tgt_mlm = mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                        instance = (src, tgt_mlm, is_wrong_order, seg_pos)
                    else:
                        instance = (src, is_wrong_order, seg_pos)

                    instances.append(instance)
                current_chunk = []
                current_length = 0
            i += 1
        return instances


class AlbertDataLoader(BertDataLoader):
    '''
    AlbertDataLoader can reuse the code of BertDataLoader.
    '''
    pass


class LmDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        pos = 0
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                document = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line))
                document = [self.vocab.get(CLS_TOKEN)] + document + [self.vocab.get(SEP_TOKEN)]

                instances_num = len(document) // (self.seq_length + 1)
                for i in range(instances_num):
                    src = document[i * (self.seq_length + 1): (i + 1) * (self.seq_length + 1)]
                    seg_pos = self.seq_length
                    pickle.dump((src, seg_pos), dataset_writer)

                src = document[instances_num * (self.seq_length + 1):]
                if len(src) > 0:
                    seg_pos = len(src)
                    while len(src) != self.seq_length + 1:
                        src.append(self.vocab.get(PAD_TOKEN))
                    pickle.dump((src, seg_pos), dataset_writer)

                if pos >= end:
                    break

        dataset_writer.close()


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
                src.append(ins[0][:-1])
                tgt.append(ins[0][1:])
                if ins[1] == len(ins[0]):
                    seg.append([1] * (ins[1] - 1))
                else:
                    seg.append([1] * ins[1] + [0] * (len(ins[0]) - 1 - ins[1]))

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)


class BilmDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        pos = 0
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                document = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line))

                instances_num = len(document) // self.seq_length
                for i in range(instances_num):
                    src = document[i * self.seq_length: (i + 1) * self.seq_length]
                    tgt_forward = src[1:] + [self.vocab.get(SEP_TOKEN)]
                    tgt_backward = [self.vocab.get(CLS_TOKEN)] + src[:-1]
                    seg = [1] * len(src)
                    pickle.dump((src, tgt_forward, tgt_backward, seg), dataset_writer)

                src = document[instances_num * self.seq_length:]
                if len(src) < 1:
                    continue
                tgt_forward = src[1:] + [self.vocab.get(SEP_TOKEN)]
                tgt_backward = [self.vocab.get(CLS_TOKEN)] + src[:-1]
                seg = [1] * len(src)
                while len(src) != self.seq_length:
                    src.append(self.vocab.get(PAD_TOKEN))
                    tgt_forward.append(self.vocab.get(PAD_TOKEN))
                    tgt_backward.append(self.vocab.get(PAD_TOKEN))
                    seg.append(0)
                pickle.dump((src, tgt_forward, tgt_backward, seg), dataset_writer)

                if pos >= end:
                    break

        dataset_writer.close()


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


class Seq2seqDataset(Dataset):
    def __init__(self, args, vocab, tokenizer):
        super(Seq2seqDataset, self).__init__(args, vocab, tokenizer)
        self.tgt_seq_length = args.tgt_seq_length
        self.src_vocab, self.src_tokenizer = vocab, tokenizer
        self.tgt_tokenizer = args.tgt_tokenizer
        self.tgt_vocab = self.tgt_tokenizer.vocab

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        pos = 0
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                if len(line.strip().split("\t")) != 2:
                    if pos >= end:
                        break
                    continue
                document_src, document_tgt = line.strip().split("\t")
                src = self.src_tokenizer.convert_tokens_to_ids(self.src_tokenizer.tokenize(document_src))
                tgt = self.tgt_tokenizer.convert_tokens_to_ids(self.tgt_tokenizer.tokenize(document_tgt))

                src = [self.src_vocab.get(CLS_TOKEN)] + src + [self.src_vocab.get(SEP_TOKEN)]
                tgt = [self.tgt_vocab.get(CLS_TOKEN)] + tgt + [self.tgt_vocab.get(SEP_TOKEN)]

                src, tgt = src[:self.seq_length], tgt[:self.tgt_seq_length + 1]
                seg_pos = [len(src)]
                while len(src) != self.seq_length:
                    src.append(self.vocab.get(PAD_TOKEN))
                while len(tgt) != self.tgt_seq_length + 1:
                    tgt.append(self.vocab.get(PAD_TOKEN))
                pickle.dump((src, tgt, seg_pos), dataset_writer)

                if pos >= end:
                    break

            dataset_writer.close()


class Seq2seqDataLoader(DataLoader):
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
            tgt_in = []
            tgt_out = []
            seg = []

            for ins in instances:
                src.append(ins[0])
                tgt_in.append(ins[1][:-1])
                tgt_out.append(ins[1][1:])
                seg.append([1] * ins[2][0] + [0] * (len(ins[0]) - ins[2][0]))

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_in), \
                torch.LongTensor(tgt_out), \
                torch.LongTensor(seg)


class T5Dataset(MlmDataset):
    '''
    T5 can reuse the code of MlmDataset.
    '''
    pass


class T5DataLoader(DataLoader):
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
            tgt_in = []
            tgt_out = []
            seg = []

            tgt_seq_length = 0

            for _, ins in enumerate(instances):
                if len(ins) == 3:
                    src_single = ins[0]
                    tgt_single = ins[1]
                    seg.append([1] * ins[2][0] + [0] * (len(ins[0]) - ins[2][0]))
                else:
                    src_single, tgt_single = mask_seq(ins[0], self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                    seg.append([1] * ins[1][0] + [0] * (len(ins[0]) - ins[1][0]))

                MASK_ID = self.vocab.get(MASK_TOKEN)
                SENTINEL_ID = self.vocab.get(SENTINEL_TOKEN)
                PAD_ID = self.vocab.get(PAD_TOKEN)

                for src_index, _ in tgt_single:
                    if src_single[src_index] != MASK_ID:
                        src_single[src_index] = MASK_ID

                tgt_in_single = [self.vocab.get(CLS_TOKEN)]
                mask_index = 0
                src_with_sentinel = []
                for token_id in src_single:
                    if token_id == MASK_ID:
                        if len(src_with_sentinel) > 0 and src_with_sentinel[-1] == (SENTINEL_ID - 1):
                            pass
                        else:
                            src_with_sentinel.append(SENTINEL_ID)
                            tgt_in_single.append(SENTINEL_ID)
                            if SENTINEL_ID < len(self.vocab) - 1:
                                SENTINEL_ID += 1
                        tgt_in_single.append(tgt_single[mask_index][1])
                        mask_index += 1
                    else:
                        src_with_sentinel.append(token_id)
                tgt_in_single.append(SENTINEL_ID)
                tgt_in_single.append(self.vocab.get(SEP_TOKEN))

                while len(src_with_sentinel) < len(src_single):
                    src_with_sentinel.append(PAD_ID)

                if len(tgt_in_single) > tgt_seq_length:
                    tgt_seq_length = len(tgt_in_single)

                src.append(src_with_sentinel)
                tgt_in.append(tgt_in_single)
                tgt_out.append(tgt_in[-1][1:] + [PAD_ID])

            for i in range(len(tgt_in)):
                while len(tgt_in[i]) != tgt_seq_length:
                    tgt_in[i].append(PAD_ID)
                    tgt_out[i].append(PAD_ID)

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_in), \
                torch.LongTensor(tgt_out), \
                torch.LongTensor(seg)


class GsgDataset(BertDataset):
    def __init__(self, args, vocab, tokenizer):
        super(GsgDataset, self).__init__(args, vocab, tokenizer)
        self.sentence_selection_strategy = args.sentence_selection_strategy
        self.tgt_seq_length = args.tgt_seq_length

    def create_single_instance(self, src, tgt):
        src = [self.vocab.get(CLS_TOKEN)] + src + [self.vocab.get(SEP_TOKEN)]
        tgt = [self.vocab.get(CLS_TOKEN)] + tgt + [self.vocab.get(SEP_TOKEN)]
        seg_pos = [len(src)]
        while len(src) != self.seq_length:
            src.append(self.vocab.get(PAD_TOKEN))
        while len(tgt) != self.tgt_seq_length:
            tgt.append(self.vocab.get(PAD_TOKEN))
        instance = (src, tgt, seg_pos)
        return instance

    def create_ins_from_doc(self, all_documents, document_index):
        sentence_selection_strategy = self.sentence_selection_strategy
        instances = []
        mask_seq_list = []
        tmp_document = []
        src = []
        tgt = []
        i = 0
        document = all_documents[document_index]
        target_seq_length, target_tgt_seq_length = self.seq_length - 2, self.tgt_seq_length - 2
        for segment in document:
            if len(segment) < target_seq_length and len(segment) < target_tgt_seq_length:
                tmp_document.append(segment)
        document = tmp_document
        mask_seq_num = int(round(len(document) * 0.3, 0))
        if sentence_selection_strategy == "random":
            mask_seq_list = random.sample(range(0, len(document) - 1), mask_seq_num)
        else:
            mask_seq_list = list(range(0, mask_seq_num))

        while i < len(document):
            segment = document[i]
            if i in mask_seq_list and len(tgt) + len(segment) < target_tgt_seq_length and len(src) + 1 < target_seq_length:
                tgt = tgt + segment
                src = src + [self.vocab.get(MASK_TOKEN)]
            elif i not in mask_seq_list and len(src) + len(segment) < target_seq_length:
                src = src + segment
            else:
                if len(tgt) > 0 and len(src) > 0:
                    instance = self.create_single_instance(src, tgt)
                    instances.append(instance)
                if i in mask_seq_list:
                    tgt = segment
                    src = [self.vocab.get(MASK_TOKEN)]
                else:
                    src = segment
                    tgt = []
            i += 1

        if len(tgt) > 0 and len(src) > 0:
            instance = self.create_single_instance(src, tgt)
            instances.append(instance)
        return instances


class GsgDataLoader(Seq2seqDataLoader):
    pass


class BartDataset(BertDataset):

    def create_single_instance(self, src, tgt):
        src = [self.vocab.get(CLS_TOKEN)] + src + [self.vocab.get(SEP_TOKEN)]
        tgt = [self.vocab.get(CLS_TOKEN)] + tgt + [self.vocab.get(SEP_TOKEN)]
        seg_pos = len(src)
        while len(src) != self.seq_length:
            src.append(self.vocab.get(PAD_TOKEN))
            tgt.append(self.vocab.get(PAD_TOKEN))
        instance = (src, tgt, seg_pos)

        return instance

    def create_ins_from_doc(self, all_documents, document_index):
        document = all_documents[document_index]
        target_seq_length = self.seq_length - 2
        src = []
        tgt = []
        instances = []
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]
            if len(segment) > target_seq_length:
                i += 1
                continue
            if current_length + len(segment) < target_seq_length:
                current_chunk.append(segment)
                current_length += len(segment)
            else:
                shuf_chunk = current_chunk.copy()
                random.shuffle(shuf_chunk)
                for k in range(len(current_chunk)):
                    src = src + shuf_chunk[k]
                    tgt = tgt + current_chunk[k]
                instance = self.create_single_instance(src, tgt)
                instances.append(instance)
                current_length = len(segment)
                current_chunk = [segment]
                src = []
                tgt = []
            i += 1
        if len(current_chunk) > 0:
            shuf_chunk = current_chunk.copy()
            random.shuffle(shuf_chunk)
            for k in range(len(current_chunk)):
                src = src + shuf_chunk[k]
                tgt = tgt + current_chunk[k]
            instance = self.create_single_instance(src, tgt)
            instances.append(instance)

        return instances


class BartDataLoader(DataLoader):
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
            tgt_in = []
            tgt_out = []
            seg = []

            for _, ins in enumerate(instances):
                src_single, tgt_single = mask_seq(ins[0], self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                seg_pos = ins[2]
                tgt_in.append(ins[1][:-1])
                tgt_out.append(ins[1][1:])

                MASK_ID = self.vocab.get(MASK_TOKEN)

                src_with_span_mask = []
                for token_id in src_single:
                    if token_id == MASK_ID:
                        if len(src_with_span_mask) > 0 and src_with_span_mask[-1] == MASK_ID:
                            seg_pos -= 1
                        else:
                            src_with_span_mask.append(MASK_ID)
                    else:
                        src_with_span_mask.append(token_id)

                while len(src_with_span_mask) < len(src_single):
                    src_with_span_mask.append(self.vocab.get(PAD_TOKEN))

                seg.append(([1] * seg_pos + [0] * (len(ins[0]) - seg_pos)))
                src.append(src_with_span_mask)


            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_in), \
                torch.LongTensor(tgt_out), \
                torch.LongTensor(seg)


class ClsDataset(Dataset):
    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        pos = 0
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                line = line.strip().split('\t')
                if len(line) == 2:
                    label = int(line[0])
                    text = line[1]
                    src = [self.vocab.get(t) for t in self.tokenizer.tokenize(text)]
                    src = [self.vocab.get(CLS_TOKEN)] + src
                    tgt = label
                    seg = [1] * len(src)
                    if len(src) >= self.seq_length:
                        src = src[:self.seq_length]
                        seg = seg[:self.seq_length]
                    else:
                        while len(src) != self.seq_length:
                            src.append(self.vocab.get(PAD_TOKEN))
                            seg.append(0)
                    pickle.dump((src, tgt, seg), dataset_writer)
                elif len(line) == 3:  # For sentence pair input.
                    label = int(line[0])
                    text_a, text_b = line[1], line[2]

                    src_a = [self.vocab.get(t) for t in self.tokenizer.tokenize(text_a)]
                    src_a = [self.vocab.get(CLS_TOKEN)] + src_a + [self.vocab.get(SEP_TOKEN)]
                    src_b = [self.vocab.get(t) for t in self.tokenizer.tokenize(text_b)]
                    src_b = src_b + [self.vocab.get(SEP_TOKEN)]

                    src = src_a + src_b
                    tgt = label
                    seg = [1] * len(src_a) + [2] * len(src_b)

                    if len(src) >= self.seq_length:
                        src = src[:self.seq_length]
                        seg = seg[:self.seq_length]
                    else:
                        while len(src) != self.seq_length:
                            src.append(self.vocab.get(PAD_TOKEN))
                            seg.append(0)
                    pickle.dump((src, tgt, seg), dataset_writer)
                else:
                    pass

                if pos >= end:
                    break

        dataset_writer.close()


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


class PrefixlmDataset(Dataset):

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        pos = 0
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                f.readline()
                pos += 1
            while True:
                line = f.readline()
                pos += 1

                if len(line.strip().split("\t")) != 2:
                    if pos >= end:
                        break
                    continue
                document_src, document_tgt = line.strip().split("\t")
                src = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(document_src))
                tgt = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(document_tgt))
                src = [self.vocab.get(CLS_TOKEN)] + src + [self.vocab.get(SEP_TOKEN)]
                tgt = tgt + [self.vocab.get(SEP_TOKEN)]
                seg_pos = [len(src)]

                if seg_pos[0] >= self.seq_length:
                    continue

                src = src + tgt
                tgt = [0] * (seg_pos[0] - 1) + tgt + [self.vocab.get(PAD_TOKEN)]
                seg_pos.append(len(src))
                src, tgt = src[:self.seq_length], tgt[:self.seq_length]
                while len(src) != self.seq_length:
                    src.append(self.vocab.get(PAD_TOKEN))
                    tgt.append(self.vocab.get(PAD_TOKEN))
                if seg_pos[1] > self.seq_length:
                    seg_pos[1] = self.seq_length

                pickle.dump((src, tgt, seg_pos), dataset_writer)

                if pos >= end:
                    break

            dataset_writer.close()


class PrefixlmDataLoader(DataLoader):
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
                seg.append([1] * ins[2][0] + [2] * (ins[2][1] - ins[2][0]) + [0] * (len(ins[0]) - ins[2][1]))

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)
