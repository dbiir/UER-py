"""
This script provides an example to use prompt for classification.
"""
import re
import sys
import os
import logging
import random
import argparse
import torch
import torch.nn as nn

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from finetune.run_classifier import *
from uer.targets import *


class ClozeTest(nn.Module):
    def __init__(self, args):
        super(ClozeTest, self).__init__()
        self.embedding = Embedding(args)
        for embedding_name in args.embedding:
            tmp_emb = str2embedding[embedding_name](args, len(args.tokenizer.vocab))
            self.embedding.update(tmp_emb, embedding_name)
        self.encoder = str2encoder[args.encoder](args)
        self.target = MlmTarget(args, len(args.tokenizer.vocab))
        if args.tie_weights:
            self.target.linear_2.weight = self.embedding.word.embedding.weight
        self.answer_position = args.answer_position
        self.device = args.device

    def forward(self, src, tgt, seg):
        emb = self.embedding(src, seg)
        memory_bank = self.encoder(emb, seg)
        output_mlm = self.target.act(self.target.linear_1(memory_bank))
        output_mlm = self.target.layer_norm(output_mlm)
        tgt_mlm = tgt.contiguous().view(-1)
        if self.target.factorized_embedding_parameterization:
            output_mlm = output_mlm.contiguous().view(-1, self.target.emb_size)
        else:
            output_mlm = output_mlm.contiguous().view(-1, self.target.hidden_size)
        output_mlm = output_mlm[tgt_mlm > 0, :]
        tgt_mlm = tgt_mlm[tgt_mlm > 0]
        self.answer_position = self.answer_position.to(self.device).view(-1)
        logits = self.target.linear_2(output_mlm)
        logits = logits * self.answer_position
        prob = self.target.softmax(logits)
        loss = self.target.criterion(prob, tgt_mlm)
        pred = prob[:, self.answer_position > 0].argmax(dim=-1)

        return loss, pred, logits


def read_dataset(args, path):
    dataset, columns = [], {}
    count, ignore_count = 0, 0
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.rstrip("\r\n").split("\t")):
                    columns[column_name] = i
                continue
            line = line.rstrip("\r\n").split("\t")
            mask_position = -1
            label = args.answer_word_dict[str(line[columns["label"]])]
            tgt_token_id = args.tokenizer.vocab[label]
            src = [args.tokenizer.vocab.get(CLS_TOKEN)]
            if "text_b" not in columns:  # Sentence classification.
                text_a = line[columns["text_a"]]
                text_a_token_id = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_a))
                max_length = args.seq_length - args.template_length - 2
                text_a_token_id = text_a_token_id[:max_length]
                for prompt_token in args.prompt_template:
                    if prompt_token == "[TEXT_A]":
                        src += text_a_token_id
                    elif prompt_token == "[ANS]":
                        src += [args.tokenizer.vocab.get(MASK_TOKEN)]
                        mask_position = len(src) - 1
                    else:
                        src += prompt_token
            else:  # Sentence-pair classification.
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                text_a_token_id = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_a))
                text_b_token_id = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b))
                max_length = args.seq_length - args.template_length - len(text_a_token_id) - 3
                text_b_token_id = text_b_token_id[:max_length]
                for prompt_token in args.prompt_template:
                    if prompt_token == "[TEXT_A]":
                        src += text_a_token_id
                        src += [args.tokenizer.vocab.get(SEP_TOKEN)]
                    elif prompt_token == "[ANS]":
                        src += [args.tokenizer.vocab.get(MASK_TOKEN)]
                        mask_position = len(src) - 1
                    elif prompt_token == "[TEXT_B]":
                        src += text_b_token_id
                    else:
                        src += prompt_token
            src += [args.tokenizer.vocab.get(SEP_TOKEN)]
            seg = [1] * len(src)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]

            if len(src) < args.seq_length:
                PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
                src += [PAD_ID] * (args.seq_length - len(src))
                seg += [0] * (args.seq_length - len(seg))
            tgt = [0] * len(src)
            # Ignore the sentence which the answer is not in a sequence
            if mask_position >= args.seq_length:
                ignore_count += 1
                continue
            tgt[mask_position] = tgt_token_id
            count += 1
            dataset.append((src, tgt, seg))
        args.logger.info(f"read dataset, count:{count}, ignore_count:{ignore_count}")
    return dataset


def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)

    loss, _, _ = model(src_batch, tgt_batch, seg_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    loss.backward()

    optimizer.step()
    scheduler.step()

    return loss


def process_prompt_template(args):
    with open(args.prompt_path, "r", encoding="utf-8") as f_json:
        temp_dict = json.load(f_json)
        template_str = temp_dict[args.prompt_id]["template"]
        template_list = re.split(r"(\[TEXT_B\]|\[TEXT_A\]|\[ANS\])", template_str)
        args.prompt_template = []
        template_length = 0
        for term in template_list:
            if len(term) > 0:
                if term not in ["[TEXT_B]", "[TEXT_A]", "[ANS]"]:
                    term_tokens = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(term))
                    args.prompt_template.append(term_tokens)
                    template_length += len(term_tokens)
                elif term in ["[TEXT_B]", "[TEXT_A]"]:
                    args.prompt_template.append(term)
                else:
                    args.prompt_template.append(term)
                    template_length += 1
        print(args.prompt_template)
        args.answer_word_dict = temp_dict[args.prompt_id]["answer_words"]
        args.answer_word_dict_inv = {v: k for k, v in args.answer_word_dict.items()}
        args.template_length = template_length


def evaluate(args, dataset):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    batch_size = args.batch_size

    correct = 0
    labels = {}
    for k in sorted([args.tokenizer.vocab[k] for k in args.answer_word_dict_inv]):
        labels[k] = len(labels)
    labels_inv = {v: k for k, v in labels.items()}
    confusion = torch.zeros(len(labels), len(labels), dtype=torch.long)
    args.model.eval()

    for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)

        with torch.no_grad():
            _, pred, _ = args.model(src_batch, tgt_batch, seg_batch)
        gold = tgt_batch[tgt_batch > 0]
        for j in range(pred.size()[0]):
            pred[j] = labels_inv[int(pred[j])]
            confusion[labels[int(pred[j])], labels[int(gold[j])]] += 1
        correct += torch.sum(pred == gold).item()

    args.logger.debug("Confusion matrix:")
    args.logger.debug(confusion)
    args.logger.debug("Report precision, recall, and f1:")

    eps = 1e-9
    for i in range(confusion.size()[0]):
        p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
        r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
        f1 = 2 * p * r / (p + r + eps)
        args.logger.debug("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))

    args.logger.info("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))
    return correct / len(dataset), confusion


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    tokenizer_opts(parser)
    finetune_opts(parser)

    parser.add_argument("--prompt_id", type=str, default="chnsenticorp_char")
    parser.add_argument("--prompt_path", type=str, default="models/prompts.json")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    args.tokenizer = str2tokenizer[args.tokenizer](args)
    set_seed(args.seed)

    process_prompt_template(args)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    answer_position = [0] * len(args.tokenizer.vocab)
    for answer in args.answer_word_dict_inv:
        answer_position[int(args.tokenizer.vocab[answer])] = 1
    args.answer_position = torch.LongTensor(answer_position)
    # Build classification model.
    model = ClozeTest(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    # Get logger.
    args.logger = init_logger(args)

    model = model.to(args.device)

    # Training phase.
    trainset = read_dataset(args, args.train_path)

    instances_num = len(trainset)
    batch_size = args.batch_size

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    args.logger.info("Batch size: {}".format(batch_size))
    args.logger.info("The number of training instances: {}".format(instances_num))
    optimizer, scheduler = build_optimizer(args, model)

    if torch.cuda.device_count() > 1:
        args.logger.info("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model

    total_loss, result, best_result = 0.0, 0.0, 0.0

    args.logger.info("Start training.")
    for epoch in range(1, args.epochs_num + 1):
        random.shuffle(trainset)
        src = torch.LongTensor([example[0] for example in trainset])
        tgt = torch.LongTensor([example[1] for example in trainset])
        seg = torch.LongTensor([example[2] for example in trainset])

        model.train()
        for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg, None)):
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                args.logger.info("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0

        result = evaluate(args, read_dataset(args, args.dev_path))
        if result[0] > best_result:
            best_result = result[0]
            save_model(model, args.output_model_path)

    # Evaluation phase.
    if args.epochs_num == 0:
        args.output_model_path = args.pretrained_model_path
    if args.test_path is not None:
        args.logger.info("Test set evaluation.")
        if torch.cuda.device_count() > 1:
            args.model.module.load_state_dict(torch.load(args.output_model_path), strict=False)
        else:
            args.model.load_state_dict(torch.load(args.output_model_path), strict=False)
        evaluate(args, read_dataset(args, args.test_path))


if __name__ == "__main__":
    main()
