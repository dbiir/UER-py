import os
import random
import pickle
import torch
from uer.utils.constants import *
from uer.utils.tokenizers import *
from uer.utils.mask import mask_seq

class Dataloader(object):
    def __init__(self, args, dataset_path, batch_size, global_rank, world_size, local_rank, shuffle=False):
        self.tokenizer = args.tokenizer
        self.batch_size = batch_size
        self.instances_buffer_size = args.instances_buffer_size
        self.global_rank = global_rank
        self.world_size = world_size
        self.local_rank = local_rank
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
                if (self.read_count - 1) % self.world_size == self.global_rank:
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


class BertDataloader(Dataloader):
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
                src_single, pad_num = ins[0]
                src_single += [self.vocab.get(PAD_TOKEN)] * pad_num

                if len(ins) == 4:
                    src.append(src_single)
                    masked_words_num += len(ins[1])
                    tgt_mlm.append([0] * len(src_single))
                    for mask in ins[1]:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    is_next.append(ins[2])
                    seg.append([1] * ins[3][0] + [2] * (ins[3][1] - ins[3][0]) + [0] * pad_num)
                else:
                    src_single, tgt_mlm_single = mask_seq(src_single, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                    masked_words_num += len(tgt_mlm_single)
                    src.append(src_single)
                    tgt_mlm.append([0] * len(src_single))
                    for mask in tgt_mlm_single:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    is_next.append(ins[1])
                    seg.append([1] * ins[2][0] + [2] * (ins[2][1] - ins[2][0]) + [0] * pad_num)

            if masked_words_num == 0:
                continue

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_mlm), \
                torch.LongTensor(is_next), \
                torch.LongTensor(seg)


class MlmDataloader(Dataloader):
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
                src_single, pad_num = ins[0]
                src_single += [self.vocab.get(PAD_TOKEN)] * pad_num

                if len(ins) == 3:
                    src.append(src_single)
                    masked_words_num += len(ins[1])
                    tgt.append([0] * len(src_single))
                    for mask in ins[1]:
                        tgt[-1][mask[0]] = mask[1]
                    seg.append([1] * ins[2][0] + [0] * pad_num)
                else:
                    src_single, tgt_single = mask_seq(src_single, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                    masked_words_num += len(tgt_single)
                    src.append(src_single)
                    tgt.append([0] * len(src_single))
                    for mask in tgt_single:
                        tgt[-1][mask[0]] = mask[1]
                    seg.append([1] * ins[1][0] + [0] * pad_num)

            if masked_words_num == 0:
                continue

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)


class AlbertDataloader(BertDataloader):
    '''
    AlbertDataloader can reuse the code of BertDataloader.
    '''
    pass


class LmDataloader(Dataloader):
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
                src_single, pad_num = ins[0]
                src_single += [self.vocab.get(PAD_TOKEN)] * pad_num
                src.append(src_single[:-1])
                tgt.append(src_single[1:])
                seg.append([1] * ins[1][0] + [0] * (len(src_single) - 1 - ins[1][0]))

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)


class BilmDataloader(Dataloader):
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
                src_single, pad_num = ins[0]
                tgt_forward_single, tgt_backward_single = ins[1], ins[2]
                src_single += [self.vocab.get(PAD_TOKEN)] * pad_num
                tgt_forward_single += [self.vocab.get(PAD_TOKEN)] * pad_num
                tgt_backward_single += [self.vocab.get(PAD_TOKEN)] * pad_num
                src.append(src_single)
                tgt_forward.append(tgt_forward_single)
                tgt_backward.append(tgt_backward_single)
                seg.append([1] * ins[3][0] + [0] * (len(src_single) - ins[3][0]))

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_forward), \
                torch.LongTensor(tgt_backward), \
                torch.LongTensor(seg)


class MtDataloader(Dataloader):
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
            tgt_seg = []

            for ins in instances:
                src_single, pad_num = ins[0]
                src_single += [self.vocab.get(PAD_TOKEN)] * pad_num
                tgt_single, pad_num = ins[1]
                tgt_single += [self.vocab.get(PAD_TOKEN)] * pad_num

                src.append(src_single)
                tgt_in.append(tgt_single[:-1])
                tgt_out.append(tgt_single[1:])
                seg.append([1] * ins[2][0] + [0] * (len(src_single) - ins[2][0]))
                pad_num = max(ins[1][1] - 1, 0)  # left shifted, pad_num >= 0
                tgt_seg.append([1] * (len(tgt_in[-1]) - pad_num) + [0] * pad_num)

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_out), \
                torch.LongTensor(seg), \
                torch.LongTensor(tgt_in), \
                torch.LongTensor(tgt_seg)


class T5Dataloader(Dataloader):
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
            tgt_seg = []

            tgt_seq_length = 0

            for _, ins in enumerate(instances):
                src_single, pad_num = ins[0]
                src_single += [self.vocab.get(PAD_TOKEN)] * pad_num

                if len(ins) == 3:
                    tgt_single = ins[1]
                    seg.append([1] * ins[2][0] + [0] * pad_num)
                else:
                    src_single, tgt_single = mask_seq(src_single, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                    seg.append([1] * ins[1][0] + [0] * pad_num)

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

                tgt_seg_single = [1] * len(tgt_in_single)

                while len(src_with_sentinel) < len(src_single):
                    src_with_sentinel.append(PAD_ID)

                if len(tgt_in_single) > tgt_seq_length:
                    tgt_seq_length = len(tgt_in_single)

                src.append(src_with_sentinel)
                tgt_in.append(tgt_in_single)
                tgt_seg.append(tgt_seg_single)
                tgt_out.append(tgt_in[-1][1:] + [PAD_ID])

            for i in range(len(tgt_in)):
                while len(tgt_in[i]) != tgt_seq_length:
                    tgt_in[i].append(PAD_ID)
                    tgt_out[i].append(PAD_ID)
                    tgt_seg[i].append(0)

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_out), \
                torch.LongTensor(seg), \
                torch.LongTensor(tgt_in), \
                torch.LongTensor(tgt_seg)


class GsgDataloader(MtDataloader):
    pass


class BartDataloader(Dataloader):
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
            tgt_seg = []

            for _, ins in enumerate(instances):
                src_single, pad_num = ins[0]
                src_single += [self.vocab.get(PAD_TOKEN)] * pad_num
                tgt_single, pad_num = ins[1]
                tgt_single += [self.vocab.get(PAD_TOKEN)] * pad_num

                src_single, _ = mask_seq(src_single, self.tokenizer, self.whole_word_masking, self.span_masking,
                                         self.span_geo_prob, self.span_max_length)
                seg_pos = ins[2][0]
                tgt_in.append(tgt_single[:-1])
                tgt_out.append(tgt_single[1:])
                pad_num = max(ins[1][1] - 1, 0)  # left shifted, pad_num >= 0
                tgt_seg.append([1] * (len(tgt_in[-1]) - pad_num) + [0] * pad_num)


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

                seg.append([1] * seg_pos + [0] * (len(src_single) - seg_pos))
                src.append(src_with_span_mask)


            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_out), \
                torch.LongTensor(seg), \
                torch.LongTensor(tgt_in), \
                torch.LongTensor(tgt_seg)


class ClsDataloader(Dataloader):
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
                src_single, pad_num = ins[0]
                seg_pos_single = ins[2]

                if len(seg_pos_single) == 1:
                    seg_single = [1] * seg_pos_single[0]
                elif len(seg_pos_single) == 2:
                    seg_single = [1] * seg_pos_single[0] + [2] * seg_pos_single[1]
                
                src_single += [self.vocab.get(PAD_TOKEN)] * pad_num
                seg_single += [0] * pad_num
                
                src.append(src_single)
                tgt.append(ins[1])
                seg.append(seg_single)

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)


class PrefixlmDataloader(Dataloader):
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
                src_single, pad_num = ins[0]
                tgt_single = ins[1]
                src_single += [self.vocab.get(PAD_TOKEN)] * pad_num
                tgt_single += [self.vocab.get(PAD_TOKEN)] * pad_num
                src.append(src_single)
                tgt.append(tgt_single)
                seg.append([1] * ins[2][0] + [2] * (ins[2][1] - ins[2][0]) + [0] * (len(src_single) - ins[2][1]))

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)


class ClsMlmDataloader(Dataloader):
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
            tgt_cls = []
            seg = []

            masked_words_num = 0

            for ins in instances:
                src_single, pad_num = ins[0]
                seg_pos_single = ins[-1]
                tgt_cls.append(ins[-2])

                if len(seg_pos_single) == 1:
                    seg_single = [1] * seg_pos_single[0]
                elif len(seg_pos_single) == 2:
                    seg_single = [1] * seg_pos_single[0] + [2] * seg_pos_single[1]
                
                src_single += [self.vocab.get(PAD_TOKEN)] * pad_num
                seg_single += [0] * pad_num
                seg.append(seg_single)

                if len(ins) == 4:
                    src.append(src_single)
                    masked_words_num += len(ins[1])
                    tgt_mlm.append([0] * len(src_single))
                    for mask in ins[1]:
                        tgt_mlm[-1][mask[0]] = mask[1]
                else:
                    src_single, tgt_single = mask_seq(src_single, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                    src.append(src_single)
                    masked_words_num += len(tgt_single)
                    tgt_mlm.append([0] * len(src_single))
                    for mask in tgt_single:
                        tgt_mlm[-1][mask[0]] = mask[1]

            if masked_words_num == 0:
                continue

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_mlm), \
                torch.LongTensor(tgt_cls), \
                torch.LongTensor(seg)
