"""
This script provides an example to wrap UER for text-to-text fine-tuning.
"""
import sys
import os
import random
import argparse
import torch

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.model_saver import save_model
from uer.targets import Seq2seqTarget
from finetune.run_classifier import *


class Text2text(torch.nn.Module):
    def __init__(self, args):
        super(Text2text, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.target = Seq2seqTarget(args, len(args.tokenizer.vocab))

    def forward(self, src, tgt, seg):
        tgt_in, tgt_out, _ = tgt
        emb = self.embedding(src, seg)
        memory_bank = self.encoder(emb, seg)
        decoder_emb = self.target.embedding(tgt_in, None)
        hidden = self.target.decoder(memory_bank, decoder_emb, (src,))
        output = self.target.output_layer(hidden)
        loss = self.target(memory_bank, tgt)[0]
        return loss, output


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line[:-1].split('\t')

            if len(columns) == 3:
                text = line[columns['text_a']] + SEP_TOKEN + line[columns['text_b']]
                label = line[columns["label"]]
            else:
                text, label = line[columns["text_a"]], line[columns["label"]]

            src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text) + [SEP_TOKEN])
            tgt_in = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(label) + [SEP_TOKEN])
            tgt_out = tgt_in[1:] + [PAD_ID]
            seg = [1] * len(src)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
            while len(src) < args.seq_length:
                src.append(PAD_ID)
                seg.append(0)
            while len(tgt_in) < args.tgt_seq_length:
                tgt_in.append(PAD_ID)
                tgt_out.append(PAD_ID)

            dataset.append((src, tgt_in, tgt_out, seg))

    return dataset


def batch_loader(batch_size, src, tgt_in, tgt_out, seg):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_in_batch = tgt_in[i * batch_size : (i + 1) * batch_size, :]
        tgt_out_batch = tgt_out[i * batch_size : (i + 1) * batch_size, :]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        yield src_batch, tgt_in_batch, tgt_out_batch, seg_batch, None

    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_in_batch = tgt_in[instances_num // batch_size * batch_size :, :]
        tgt_out_batch = tgt_out[instances_num // batch_size * batch_size :, :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        yield src_batch, tgt_in_batch, tgt_out_batch, seg_batch, None


def train_model(args, model, optimizer, scheduler, src_batch, tgt_in_batch, tgt_out_batch, seg_batch):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_in_batch = tgt_in_batch.to(args.device)
    tgt_out_batch = tgt_out_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)

    loss, _ = model(src_batch, (tgt_in_batch, tgt_out_batch, src_batch), seg_batch)

    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    if args.fp16:
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()
    scheduler.step()

    return loss


def evaluate(args, dataset):

    src = torch.LongTensor([example[0] for example in dataset])
    tgt_in = torch.LongTensor([example[1] for example in dataset])
    tgt_out = torch.LongTensor([example[2] for example in dataset])
    seg = torch.LongTensor([example[3] for example in dataset])

    generated_sentences = []
    args.model.eval()

    for i, (src_batch, tgt_in_batch, tgt_out_batch, seg_batch, _) in enumerate(batch_loader(args.batch_size, src, tgt_in, tgt_out, seg)):

        src_batch = src_batch.to(args.device)
        tgt_in_batch = torch.zeros(tgt_in_batch.size()[0],1, dtype = torch.long, device = args.device)
        for j in range(tgt_in_batch.size()[0]):
            tgt_in_batch[j][-1] = args.tokenizer.vocab.get(CLS_TOKEN)

        seg_batch = seg_batch.to(args.device)

        for _ in range(args.tgt_seq_length):

            tgt_out_batch = tgt_in_batch
            with torch.no_grad():
                _, outputs = args.model(src_batch, (tgt_in_batch, tgt_out_batch, src_batch), seg_batch)

            tgt_in_batch = torch.cat([tgt_in_batch, torch.zeros(tgt_in_batch.size()[0],1, dtype = torch.long, device = args.device)], dim=1)
            for j in range(len(outputs)):
                next_token_logits = outputs[j][-1]
                next_token = torch.argmax(next_token_logits)
                tgt_in_batch[j][-1] = next_token

        for j in range(len(outputs)):
            sentence = " ".join([args.tokenizer.inv_vocab[token_id.item()] for token_id in tgt_in_batch[j][1:]])
            generated_sentences.append(sentence)

    labels = {}
    labels_num = 0
    for example in dataset:
        label = "".join([args.tokenizer.inv_vocab[token_id] for token_id in example[2][:-2]]).split(SEP_TOKEN)[0]
        if not labels.get(label, None):
            labels[label] = labels_num
            labels_num += 1
    confusion_matrix = torch.zeros(labels_num, labels_num, dtype=torch.long)
    correct = 0

    for i, example in enumerate(dataset):
        tgt = example[2]
        tgt_token = " ".join([args.tokenizer.inv_vocab[token_id] for token_id in tgt[:-2]])
        generated_sentences[i] = generated_sentences[i].split(SEP_TOKEN)[0]

        pred = "".join(generated_sentences[i].split(' '))
        gold = "".join(tgt_token.split(SEP_TOKEN)[0].split(' '))

        if pred in labels.keys():
            confusion_matrix[labels[pred], labels[gold]] += 1

        if pred == gold:
            correct += 1

    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))
    return correct / len(dataset)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    tokenizer_opts(parser)

    parser.add_argument("--tgt_embedding", choices=["word", "word_pos", "word_pos_seg", "word_sinusoidalpos"], default="word_pos_seg",
                        help="Target embedding type.")
    parser.add_argument("--decoder", choices=["transformer"], default="transformer", help="Decoder type.")
    parser.add_argument("--tie_weights", action="store_true",
                        help="Tie the word embedding and softmax weights.")
    parser.add_argument("--has_lmtarget_bias", action="store_true",
                        help="Add bias on output_layer for lm target.")
    parser.add_argument("--tgt_seq_length", type=int, default=32,
                        help="Output sequence length.")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model.
    model = Text2text(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)

    # Training phase.
    trainset = read_dataset(args, args.train_path)
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    src = torch.LongTensor([example[0] for example in trainset])
    tgt_in = torch.LongTensor([example[1] for example in trainset])
    tgt_out = torch.LongTensor([example[2] for example in trainset])
    seg = torch.LongTensor([example[3] for example in trainset])

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model

    total_loss, result, best_result = 0.0, 0.0, 0.0

    print("Start training.")

    for epoch in range(1, args.epochs_num + 1):
        model.train()
        for i, (src_batch, tgt_in_batch, tgt_out_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt_in, tgt_out, seg)):
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_in_batch, tgt_out_batch, seg_batch)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0

        result = evaluate(args, read_dataset(args, args.dev_path))
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
        evaluate(args, read_dataset(args, args.test_path), True)


if __name__ == "__main__":
    main()
