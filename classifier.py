# -*- encoding:utf-8 -*-
"""
  This script provides an exmaple to wrap bert-pytorch for classification.
"""
import torch
import json
import random
import argparse
import collections
import torch.nn as nn
from bert.utils.vocab import Vocab
from bert.utils.constants import *
from bert.utils.tokenizer import * 
from bert.model_builder import build_model
from bert.utils.optimizers import  BertAdam
from bert.utils.config import load_hyperparam
from bert.utils.seed import set_seed
from bert.model_saver import save_model


class BertClassifier(nn.Module):
    def __init__(self, args, bert_model):
        super(BertClassifier, self).__init__()
        self.embedding = bert_model.embedding
        self.encoder = bert_model.encoder
        self.target = bert_model.target
        self.labels_num = args.labels_num
        self.classifier = nn.Linear(args.hidden_size, self.labels_num)
        self.dropout = nn.Dropout(args.dropout)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()

    def forward(self, src, label, mask):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size]
            mask: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, mask)
        # Encoder.
        # seq_length = emb.size(1)
        # mask_attn = (mask>0).unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1)
        # mask_attn = mask_attn.float()
        # mask_attn = (1.0 - mask_attn) * -10000.0
        output = self.encoder(emb, mask)
        # Target.
        output = torch.tanh(self.target.pooler(output[:,0,:]))
        output = self.dropout(output)
        logits = self.classifier(output)
        loss = self.criterion(self.softmax(logits.view(-1, self.labels_num)), label.view(-1))
        return loss, logits


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default="./models/google_model.bin", type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="./models/classifier_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default="./models/google_vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str, required=True,
                        help="Path of the devset.") 
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path of the testset.")
    parser.add_argument("--config_path", default="./models/google_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=100,
                        help="Sequence length.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "word", "space"], default="bert",
                        help="Specify the tokenizer." 
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Word tokenizer supports online word segmentation based on jieba segmentor."
                             "Space tokenizer segments sentences into words according to space."
                             )

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=3,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    labels_set = set()
    with open(args.train_path, mode="r", encoding="utf-8") as f:
        for line in f:
            try:
                line = line.strip().split()
                label = int(line[0])
                labels_set.add(label)
            except:
                pass
    args.labels_num = len(labels_set) 

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)

    # Build bert model.
    bert_model = build_model(args, len(vocab))

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        bert_model.load_state_dict(torch.load(args.pretrained_model_path), strict=True)  
    else:
        # Initialize with normal distribution.
        for n, p in list(bert_model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)
    
    # Build classification model.
    model = BertClassifier(args, bert_model)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)
    
    # Datset loader.
    def batch_loader(batch_size, input_ids, label_ids, mask_ids):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i*batch_size: (i+1)*batch_size, :]
            label_ids_batch = label_ids[i*batch_size: (i+1)*batch_size]
            mask_ids_batch = mask_ids[i*batch_size: (i+1)*batch_size, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num//batch_size*batch_size:, :]
            label_ids_batch = label_ids[instances_num//batch_size*batch_size:]
            mask_ids_batch = mask_ids[instances_num//batch_size*batch_size:, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch

    # Build tokenizer.
    tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)

    # # Read dataset.
    # def read_dataset(path):
    #     dataset = []
    #     with open(path, mode="r", encoding="utf-8") as f:
    #         for line in f:
    #             try:
    #                 line = line.strip().split()
    #                 label = int(line[0])
    #                 text = " ".join(line[1:])
    #                 tokens = [vocab.get(t) for t in tokenizer.tokenize(text)]
    #                 tokens = [CLS_ID] + tokens
    #                 mask = [1] * len(tokens)
    #                 if len(tokens) > args.seq_length:
    #                     tokens = tokens[:args.seq_length]
    #                     mask = mask[:args.seq_length]
    #                 while len(tokens) < args.seq_length:
    #                     tokens.append(0)
    #                     mask.append(0)
    #                 dataset.append((tokens, label, mask))
    #             except:
    #                 pass
    #     return dataset

    # Read dataset.
    def read_dataset(path):
        dataset = []
        with open(path, mode="r", encoding="utf-8") as f:
            for line in f:
                try:
                    line = line.strip().split('\t')
                    if len(line) == 2:
                        label = int(line[0])
                        text = " ".join(line[1:])
                        tokens = [vocab.get(t) for t in tokenizer.tokenize(text)]
                        tokens = [CLS_ID] + tokens
                        mask = [1] * len(tokens)
                        if len(tokens) > args.seq_length:
                            tokens = tokens[:args.seq_length]
                            mask = mask[:args.seq_length]
                        while len(tokens) < args.seq_length:
                            tokens.append(0)
                            mask.append(0)
                        dataset.append((tokens, label, mask))
                    elif len(line) == 3: # For sentence pair input.
                        label = int(line[0])
                        text_a, text_b = line[1], line[2]

                        tokens_a = [vocab.get(t) for t in tokenizer.tokenize(text_a)]
                        tokens_a = [CLS_ID] + tokens_a + [SEP_ID]
                        tokens_b = [vocab.get(t) for t in tokenizer.tokenize(text_b)]
                        tokens_b = tokens_b + [SEP_ID]

                        tokens = tokens_a + tokens_b
                        mask = [1] * len(tokens_a) + [2] * len(tokens_b)
                        
                        if len(tokens) > args.seq_length:
                            tokens = tokens[:args.seq_length]
                            mask = mask[:args.seq_length]
                        while len(tokens) < args.seq_length:
                            tokens.append(0)
                            mask.append(0)
                        dataset.append((tokens, label, mask))
                    else:
                        pass
                        
                except:
                    pass
        return dataset

    # Evaluation function.
    def evaluate(args, is_test):
        if is_test:
            dataset = read_dataset(args.test_path)
        else:
            dataset = read_dataset(args.dev_path)
        random.shuffle(dataset)
        input_ids = torch.LongTensor([sample[0] for sample in dataset])
        label_ids = torch.LongTensor([sample[1] for sample in dataset])
        mask_ids = torch.LongTensor([sample[2] for sample in dataset])

        batch_size = args.batch_size
        instances_num = input_ids.size()[0]
        if is_test:
            print("The number of evaluation instances: ", instances_num)

        correct = 0
        # Confusion matrix.
        confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

        model.eval()

        for i, (input_ids_batch, label_ids_batch,  mask_ids_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids)):
            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            with torch.no_grad():
                loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch)
            logits = nn.Softmax(dim=1)(logits)
            pred = torch.argmax(logits, dim=1)
            gold = label_ids_batch
            for j in range(pred.size()[0]):
                confusion[pred[j], gold[j]] += 1
            correct += torch.sum(pred == gold).item()
        
        if is_test:
            print("Confusion matrix:")
            print(confusion)
            print("Report precision, recall, and f1:")
        for i in range(confusion.size()[0]):
            p = confusion[i,i].item()/confusion[i,:].sum().item()
            r = confusion[i,i].item()/confusion[:,i].sum().item()
            f1 = 2*p*r / (p+r)
            if is_test:
                print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i,p,r,f1))
        print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct/len(dataset), correct, len(dataset)))
        return correct/len(dataset)

    # Training phase.
    print("Start training.")
    trainset = read_dataset(args.train_path)
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    input_ids = torch.LongTensor([example[0] for example in trainset])
    label_ids = torch.LongTensor([example[1] for example in trainset])
    mask_ids = torch.LongTensor([example[2] for example in trainset])

    train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=train_steps)

    total_loss = 0.
    acc = 0.0
    best_acc = 0.0
    
    for epoch in range(1, args.epochs_num+1):
        model.train()
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids)):
            model.zero_grad()

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)

            loss, _ = model(input_ids_batch, label_ids_batch, mask_ids_batch)
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i+1, total_loss / args.report_steps))
                total_loss = 0.
            loss.backward()
            optimizer.step()
        acc = evaluate(args, False)
        if acc > best_acc:
            best_acc = acc
            save_model(model, args.output_model_path)
        else:
            break


    # Evaluation phase.
    print("Start evaluation.")

    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load(args.output_model_path))
    else:
        model.load_state_dict(torch.load(args.output_model_path))

    evaluate(args, True)


if __name__ == "__main__":
    main()
