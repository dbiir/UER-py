"""
This script provides an example to wrap UER-py for Chinese machine reading comprehension.
"""
import sys
import os
import re
import argparse
import json
import random
import torch
import torch.nn as nn

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.layers import *
from uer.encoders import *
from uer.utils.constants import *
from uer.utils.tokenizers import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.opts import finetune_opts
from finetune.run_classifier import build_optimizer, load_or_initialize_parameters


class MachineReadingComprehension(nn.Module):
    def __init__(self, args):
        super(MachineReadingComprehension, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.output_layer = nn.Linear(args.hidden_size, 2)

    def forward(self, src, seg, start_position, end_position):
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output = self.encoder(emb, seg)
        # Target.
        logits = self.output_layer(output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)
        start_loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(start_logits), start_position)
        end_loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(end_logits), end_position)
        loss = (start_loss + end_loss) / 2

        return loss, start_logits, end_logits


def read_examples(path):
    # Read squad-style examples.
    examples = []
    with open(path, mode="r", encoding="utf-8") as f:
        for article in json.load(f)["data"]:
            for para in article["paragraphs"]:
                context = para["context"]
                for qa in para["qas"]:
                    question = qa["question"]
                    question_id = qa["id"]
                    answer_texts, start_positions, end_positions = [], [], []
                    for answer in qa["answers"]:
                        answer_texts.append(answer["text"])
                        start_positions.append(answer["answer_start"])
                        end_positions.append(answer["answer_start"] + len(answer["text"]) - 1)
                    examples.append((context, question, question_id, start_positions, end_positions, answer_texts))
    return examples


def convert_examples_to_dataset(args, examples):
    # Converts a list of examples into a dataset that can be directly given as input to a model.
    dataset = []
    print("The number of questions in the dataset:", len(examples))
    for i in range(len(examples)):
        context = examples[i][0]
        question = examples[i][1]
        question_id = examples[i][2]
        # Only consider the first answer.
        start_position_absolute = examples[i][3][0]
        end_position_absolute = examples[i][4][0]
        answers = examples[i][5]
        max_context_length = args.seq_length - len(question) - 3
        # Divide the context into multiple spans.
        doc_spans = []
        start_offset = 0
        while start_offset < len(context):
            length = len(context) - start_offset
            if length > max_context_length:
                length = max_context_length
            doc_spans.append((start_offset, length))
            if start_offset + length == len(context):
                break
            start_offset += min(length, args.doc_stride)

        for doc_span_index, doc_span in enumerate(doc_spans):
            start_offset = doc_span[0]
            span_context = context[start_offset : start_offset + doc_span[1]]
            # Convert absolute position to relative position.
            start_position = start_position_absolute - start_offset + len(question) + 2
            end_position = end_position_absolute - start_offset + len(question) + 2

            # If span does not contain the complete answer, we use it for data augmentation.
            if start_position < len(question) + 2:
                start_position = len(question) + 2
            if end_position > doc_span[1] + len(question) + 1:
                end_position = doc_span[1] + len(question) + 1
            if start_position > doc_span[1] + len(question) + 1 or end_position < len(question) + 2:
                start_position, end_position = 0, 0

            src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(question) + [SEP_TOKEN])
            src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(span_context) + [SEP_TOKEN])
            src = src_a + src_b
            seg = [1] * len(src_a) + [2] * len(src_b)
            PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
            while len(src) < args.seq_length:
                src.append(PAD_ID)
                seg.append(0)

            dataset.append((src, seg, start_position, end_position, answers, question_id, len(question), doc_span_index, start_offset))
    return dataset


def read_dataset(args, path):
    examples = read_examples(path)
    dataset = convert_examples_to_dataset(args, examples)
    return dataset, examples


def batch_loader(batch_size, src, seg, start_position, end_position):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        start_position_batch = start_position[i * batch_size : (i + 1) * batch_size]
        end_position_batch = end_position[i * batch_size : (i + 1) * batch_size]
        yield src_batch, seg_batch, start_position_batch, end_position_batch
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        start_position_batch = start_position[instances_num // batch_size * batch_size :]
        end_position_batch = end_position[instances_num // batch_size * batch_size :]
        yield src_batch, seg_batch, start_position_batch, end_position_batch


def train(args, model, optimizer, scheduler, src_batch, seg_batch, start_position_batch, end_position_batch):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    start_position_batch = start_position_batch.to(args.device)
    end_position_batch = end_position_batch.to(args.device)

    loss, _, _ = model(src_batch, seg_batch, start_position_batch, end_position_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    if args.fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()
    scheduler.step()

    return loss


# Evaluation script from CMRC2018.
# We modify the tokenizer.
def mixed_segmentation(in_str, rm_punc=False):
    #in_str = str(in_str).decode('utf-8').lower().strip()
    n_str = str(in_str).lower().strip()
    segs_out = []
    temp_str = ""
    sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
               '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
               '「','」','（','）','－','～','『','』']
    for char in in_str:
        if rm_punc and char in sp_char:
            continue
        #if re.search(ur'[\u4e00-\u9fa5]', char) or char in sp_char:
        if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
            if temp_str != "":
                #ss = nltk.word_tokenize(temp_str)
                ss = list(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    if temp_str != "":
        #ss = nltk.word_tokenize(temp_str)
        ss = list(temp_str)
        segs_out.extend(ss)

    return segs_out


def find_lcs(s1, s2):
    m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i+1][j+1] = m[i][j]+1
                if m[i+1][j+1] > mmax:
                    mmax=m[i+1][j+1]
                    p=i+1
    return s1[p-mmax:p], mmax


def remove_punctuation(in_str):
    #in_str = str(in_str).decode('utf-8').lower().strip()
    in_str = str(in_str).lower().strip()
    sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
               '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
               '「','」','（','）','－','～','『','』']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


def calc_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        ans_segs = mixed_segmentation(ans, rm_punc=True)
        prediction_segs = mixed_segmentation(prediction, rm_punc=True)
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        precision   = 1.0*lcs_len/len(prediction_segs)
        recall      = 1.0*lcs_len/len(ans_segs)
        f1          = (2*precision*recall)/(precision+recall)
        f1_scores.append(f1)
    return max(f1_scores)


def calc_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = remove_punctuation(ans)
        prediction_ = remove_punctuation(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em


def get_answers(dataset, start_prob_all, end_prob_all):
    previous_question_id = -1
    pred_answers = []
    # For each predicted answer, we store its span index, start position, end position, and score.
    current_answer = (-1, -1, -1, -100.0)
    for i in range(len(dataset)):
        question_id = dataset[i][5]
        question_length = dataset[i][6]
        span_index = dataset[i][7]
        start_offset = dataset[i][8]

        start_scores, end_scores = start_prob_all[i], end_prob_all[i]

        start_pred = torch.argmax(start_scores[question_length + 2 :], dim=0) + question_length + 2
        end_pred = start_pred + torch.argmax(end_scores[start_pred:], dim=0)
        score = start_scores[start_pred] + end_scores[end_pred]

        start_pred_absolute = start_pred + start_offset - question_length - 2
        end_pred_absolute = end_pred + start_offset - question_length - 2

        if question_id == previous_question_id:
            if score > current_answer[3]:
                current_answer = (span_index, start_pred_absolute, end_pred_absolute, score)
        else:
            if i > 0:
                pred_answers.append(current_answer)
            previous_question_id = question_id
            current_answer = (span_index, start_pred_absolute, end_pred_absolute, score)
    pred_answers.append(current_answer)
    return pred_answers


# Evaluation function.
def evaluate(args, dataset, examples):
    src = torch.LongTensor([sample[0] for sample in dataset])
    seg = torch.LongTensor([sample[1] for sample in dataset])
    start_position = torch.LongTensor([sample[2] for sample in dataset])
    end_position = torch.LongTensor([sample[3] for sample in dataset])

    batch_size = args.batch_size
    instances_num = src.size()[0]

    args.model.eval()
    start_prob_all, end_prob_all = [], []

    for i, (src_batch, seg_batch, start_position_batch, end_position_batch) in enumerate(batch_loader(batch_size, src, seg, start_position, end_position)):
        src_batch = src_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        start_position_batch = start_position_batch.to(args.device)
        end_position_batch = end_position_batch.to(args.device)

        with torch.no_grad():
            loss, start_logits, end_logits = args.model(src_batch, seg_batch, start_position_batch, end_position_batch)

        start_prob = nn.Softmax(dim=1)(start_logits)
        end_prob = nn.Softmax(dim=1)(end_logits)

        for j in range(start_prob.size()[0]):
            start_prob_all.append(start_prob[j])
            end_prob_all.append(end_prob[j])

    pred_answers = get_answers(dataset, start_prob_all, end_prob_all)

    f1, em = 0, 0
    total_count, skip_count = len(examples), 0
    for i in range(len(examples)):
        answers = examples[i][5]
        start_pred_pos = pred_answers[i][1]
        end_pred_pos = pred_answers[i][2]

        if end_pred_pos <= start_pred_pos:
            skip_count += 1
            continue

        prediction = examples[i][0][start_pred_pos: end_pred_pos + 1]

        f1 += calc_f1_score(answers, prediction)
        em += calc_em_score(answers, prediction)

    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count
    avg = (f1_score + em_score) * 0.5
    print("Avg: {:.4f},F1:{:.4f},EM:{:.4f},Total:{},Skip:{}".format(avg, f1_score, em_score, total_count, skip_count))
    return avg


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Build tokenizer.
    args.tokenizer = CharTokenizer(args)

    # Build machine reading comprehension model.
    model = MachineReadingComprehension(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)

    # Build tokenizer.
    args.tokenizer = CharTokenizer(args)

    # Training phase.
    batch_size = args.batch_size
    print("Batch size: ", batch_size)
    trainset, _ = read_dataset(args, args.train_path)
    random.shuffle(trainset)
    instances_num = len(trainset)

    src = torch.LongTensor([sample[0] for sample in trainset])
    seg = torch.LongTensor([sample[1] for sample in trainset])
    start_position = torch.LongTensor([sample[2] for sample in trainset])
    end_position = torch.LongTensor([sample[3] for sample in trainset])

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("The number of training instances:", instances_num)

    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer,opt_level=args.fp16_opt_level)

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model

    total_loss = 0.0
    result = 0.0
    best_result = 0.0

    print("Start training.")

    for epoch in range(1, args.epochs_num + 1):
        model.train()

        for i, (src_batch, seg_batch, start_position_batch, end_position_batch) in enumerate(batch_loader(batch_size, src, seg, start_position, end_position)):
            loss = train(args, model, optimizer, scheduler, src_batch, seg_batch, start_position_batch, end_position_batch)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i+1, total_loss / args.report_steps))
                total_loss = 0.0

        result = evaluate(args, *read_dataset(args, args.dev_path))
        if result > best_result:
            best_result = result
            save_model(model, args.output_model_path)

    # Evaluation phase.
    if args.test_path is not None:
        print("Test set evaluation.")
        if torch.cuda.device_count() > 1:
            args.model.module.load_state_dict(torch.load(args.output_model_path))
        else:
            args.model.load_state_dict(torch.load(args.output_model_path))
        evaluate(args, *read_dataset(args, args.test_path))


if __name__ == "__main__":
    main()
