# -*- coding: utf-8 -*-
from __future__ import print_function
from collections import Counter, OrderedDict
import string
import re
import argparse
import json
import sys
import importlib
importlib.reload(sys)
import nltk
import pdb

import torch
import random
import collections
import torch.nn as nn

from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils.tokenizer import * 
from uer.model_builder import build_model
from uer.utils.optimizers import  BertAdam
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model


class BertQuestionAnswering(nn.Module):
    def __init__(self, args, bert_model):
        super(BertQuestionAnswering, self).__init__()
        self.embedding = bert_model.embedding
        self.encoder = bert_model.encoder
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, 2)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()
    def forward(self, src, mask, start_positions, end_positions):
        """
        Args:
            src: [batch_size x seq_length]
            start_positions,end_positions: [batch_size]
            mask: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, mask)
        # Encoder.
        output = self.encoder(emb, mask)
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)#batch_size*seq*2
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
 
        start_loss = self.criterion(self.softmax(start_logits), start_positions)
        end_loss = self.criterion(self.softmax(end_logits), end_positions)
        loss = (start_loss + end_loss) / 2
        
        return loss, start_logits, end_logits


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="./models/QA_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default="./models/google_vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str, required=True,
                        help="Path of the devset.") 
    parser.add_argument("--test_path", type=str,
                        help="Path of the testset.")
    parser.add_argument("--config_path", default="./models/google_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=100,
                        help="Sequence length.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--embedding", choices=["bert", "word"], default="bert",
                        help="Emebdding type.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                                   "cnn", "gatedcnn", "attn", \
                                                   "rcnn", "crnn", "gpt"], \
                                                   default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")


    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="char",
                        help="Specify the tokenizer." 
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=3e-5,
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
      
    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    args.target = "bert"
    bert_model = build_model(args)
    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model. 
        bert_model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)  
    else:
        # Initialize with normal distribution.
        for n, p in list(bert_model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)
    
    # Build QA model.
    model = BertQuestionAnswering(args,bert_model)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)

    # Dataset loader.
    def batch_loader(batch_size, input_ids, mask_ids, start_positions, end_positions):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i*batch_size: (i+1)*batch_size, :]
            mask_ids_batch = mask_ids[i*batch_size: (i+1)*batch_size, :]
            start_positions_batch = start_positions[i*batch_size: (i+1)*batch_size]
            end_positions_batch = end_positions[i*batch_size: (i+1)*batch_size]
            yield input_ids_batch, mask_ids_batch, start_positions_batch, end_positions_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num//batch_size*batch_size:, :]
            mask_ids_batch = mask_ids[instances_num//batch_size*batch_size:, :]
            start_positions_batch = start_positions[instances_num//batch_size*batch_size:]
            end_positions_batch = end_positions[instances_num//batch_size*batch_size:]
            yield input_ids_batch, mask_ids_batch, start_positions_batch, end_positions_batch

    # Build tokenizer.
    tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)

    # Read examples.
    def read_examples(path):
        examples = []
        with open(path,'r',encoding='utf-8') as fp:
            all_dict = json.loads(fp.read())
            v1 = all_dict["data"]
            for i in range(len(v1)):
                data_dict = v1[i]
                v2 = data_dict["paragraphs"]

                for j in range(len(v2)):
                    para_dict = v2[j]
                    context = para_dict["context"]
                    v3 = para_dict["qas"]

                    for m in range(len(v3)):
                        qas_dict = v3[m]
                        question = qas_dict["question"]                                          
                        question_id = qas_dict["id"]
                        v4 = qas_dict["answers"]
                        
                        answers=[]
                        start_positions=[]
                        end_positions=[]

                        for n in range(len(v4)):
                            ans_dict = v4[n]
                            answer = ans_dict["text"]
                            start_position = ans_dict["answer_start"]
                            end_position = start_position + len(answer)
                            
                            answers.append(answer)
                            start_positions.append(start_position)
                            end_positions.append(end_position)

                        examples.append((context,question,question_id,start_positions,end_positions,answers))
        
        return examples


    def convert_examples_to_dataset(examples, args):
        dataset = []
        print("The number of questions in the dataset",len(examples))
        for i in range(len(examples)):
            context = examples[i][0]
            question = examples[i][1]
            q_len = len(question)
            question_id = examples[i][2]

            start_positions_true = examples[i][3][0]#待修改
            end_positions_true = examples[i][4][0]
            
            answers = examples[i][5]
            max_context_length = args.seq_length - q_len - 3
            # divide the context to some spans
            _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
                "DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(context):
                length = len(context) - start_offset
                if length > max_context_length:
                    length = max_context_length
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(context):
                    break
                start_offset += min(length, args.doc_stride)

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                doc_span_start=doc_span.start
                span_context = context[doc_span_start:doc_span_start+doc_span.length]         
                # convert the start or end position to real position in tokens
                start_positions = start_positions_true - doc_span_start + q_len + 2
                end_positions = end_positions_true - doc_span_start + q_len + 2
                # the answers of some question are not in the doc_span, we ignore them.
                if start_positions < q_len+2 or start_positions > doc_span.length+q_len+2 or end_positions < q_len+2 or end_positions > doc_span.length+q_len+2:
                    continue 

                tokens_a = [vocab.get(t) for t in tokenizer.tokenize(question)]
                tokens_a = [CLS_ID] + tokens_a + [SEP_ID]
                tokens_b = [vocab.get(t) for t in tokenizer.tokenize(span_context)]
                tokens_b = tokens_b + [SEP_ID] 
                tokens = tokens_a + tokens_b
                mask = [1] * len(tokens_a) + [2] * len(tokens_b)

                while len(tokens) < args.seq_length:
                    tokens.append(0)
                    mask.append(0)

                dataset.append((tokens,mask,start_positions,end_positions,answers,question_id,q_len,doc_span_index,doc_span_start))       
        return dataset


    # Evaluation function.
    def evaluate(args, is_test):
        # some calculation functions
        def mixed_segmentation(in_str, rm_punc=False):
            in_str = str(in_str).lower().strip()
            segs_out = []
            temp_str = ""
            sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
                   '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
                   '「','」','（','）','－','～','『','』']
            for char in in_str:
                if rm_punc and char in sp_char:
                    continue
                if  re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
                    if temp_str != "":
                        ss = nltk.word_tokenize(temp_str)
                        segs_out.extend(ss)
                        temp_str = ""
                    segs_out.append(char)
                else:
                    temp_str += char

            #handling last part
            if temp_str != "":
                ss = nltk.word_tokenize(temp_str)
                segs_out.extend(ss)

            return segs_out


        # remove punctuation
        def remove_punctuation(in_str):
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


        # find longest common string
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

        def calc_f1_score(answers, prediction):
            f1_scores = []     
            for i in range(len(answers)):
                ans = answers[i]
                ans_segs = mixed_segmentation(ans, rm_punc=True)
                prediction_segs = mixed_segmentation(prediction, rm_punc=True)
                lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
                if lcs_len == 0:
                    f1_scores.append(0)
                else:
                    precision   = 1.0*lcs_len/len(prediction_segs)
                    recall      = 1.0*lcs_len/len(ans_segs)
                    f1          = (2*precision*recall)/(precision+recall)
                    f1_scores.append(f1)
            return max(f1_scores)


        def calc_em_score(answers, prediction):
            em = 0
            for i in range(len(answers)):
                ans = answers[i]
                ans_ = remove_punctuation(ans)
                prediction_ = remove_punctuation(prediction)
                if ans_ == prediction_:
                    em = 1
                    break
            return em

        def is_max_score(score_list):
            score_max = -100
            index_max = 0
            best_start_prediction = 0
            best_end_prediction = 0
            for i in range(len(score_list)): 
                if score_max <= score_list[i][3]:
                    score_max = score_list[i][3]
                    index_max = score_list[i][0]
                    best_start_prediction = score_list[i][1]
                    best_end_prediction = score_list[i][2]
            return index_max, best_start_prediction,best_end_prediction

        if is_test:
            examples = read_examples(args.test_path)
            dataset = convert_examples_to_dataset(examples,args)

        else:
            examples = read_examples(args.dev_path)
            dataset = convert_examples_to_dataset(examples,args)
        
        input_ids = torch.LongTensor([sample[0] for sample in dataset])
        mask_ids = torch.LongTensor([sample[1] for sample in dataset])
        start_positions = torch.LongTensor([sample[2] for sample in dataset])
        end_positions = torch.LongTensor([sample[3] for sample in dataset])
        
        batch_size = args.batch_size
        instances_num = input_ids.size()[0]
        
        if is_test:
            print("The number of evaluation instances: ", instances_num)
        model.eval()
        start_logits_all = []
        end_logits_all = []
        start_pred_all = []
        end_pred_all = []
        for i, (input_ids_batch, mask_ids_batch, start_positions_batch, end_positions_batch) in enumerate(batch_loader(batch_size, input_ids, mask_ids, start_positions, end_positions)):
            model.zero_grad()
            input_ids_batch = input_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            start_positions_batch = start_positions_batch.to(device)
            end_positions_batch = end_positions_batch.to(device)
                
            with torch.no_grad():
                loss, start_logits, end_logits = model(input_ids_batch, mask_ids_batch, start_positions_batch, end_positions_batch)
                
            start_logits = nn.Softmax(dim=1)(start_logits)
            end_logits = nn.Softmax(dim=1)(end_logits)

            start_pred = torch.argmax(start_logits, dim=1)
            end_pred = torch.argmax(end_logits, dim=1)

            start_pred=start_pred.cpu().numpy().tolist()
            end_pred=end_pred.cpu().numpy().tolist()
            
            start_logits=start_logits.cpu().numpy().tolist()
            end_logits=end_logits.cpu().numpy().tolist()

            start_logits_max=[]
            end_logits_max=[]
            for j in range(len(start_pred)):
                start_logits_max.append(start_logits[j][start_pred[j]])
                end_logits_max.append(end_logits[j][end_pred[j]])
            
            start_logits_all += start_logits_max
            end_logits_all += end_logits_max
            start_pred_all += start_pred
            end_pred_all  += end_pred
        
        assert len(start_pred_all)==len(dataset)
        assert len(start_logits_all)==len(dataset)        

        # couster by question id and chose the best answer in doc_spans
        order = -1
        pred_list = []
        templist=[]
        for i in range(len(dataset)):
            qid = dataset[i][5]
            q_len = dataset[i][6]
            span_index =dataset[i][7]
            doc_span_start = dataset[i][8]

            score1 = float(start_logits_all[i])
            score2 = float(end_logits_all[i])
            score = (score1+score2)/2
            
            pre_start_pred = start_pred_all[i] + doc_span_start - q_len - 2
            pre_end_pred = end_pred_all[i] + doc_span_start - q_len - 2
            
            if qid == order:
                templist.append((span_index,pre_start_pred,pre_end_pred,score))
            else:
                order = qid
                if i > 0:
                    span_index_max, best_start_prediction,best_end_prediction = is_max_score(templist)   
                    pred_list.append((span_index_max, best_start_prediction,best_end_prediction))
                templist = []
                templist.append((span_index,pre_start_pred,pre_end_pred,score))
        span_index_max, best_start_prediction, best_end_prediction = is_max_score(templist)   
        pred_list.append((span_index_max, best_start_prediction,best_end_prediction))
   
        assert len(pred_list) == len(examples)

        #strat pred
        f1 = 0
        em = 0
        total_count = len(examples)
        skip_count = 0
        for i in range(len(examples)):
            question_id = examples[i][2]
            answers = examples[i][5]
            span_index = pred_list[i][0]
            start_prediction = pred_list[i][1]
            end_prediction = pred_list[i][2]
            
            #error prediction
            if end_prediction <= start_prediction:
                skip_count += 1
                continue
                
            prediction = examples[i][0][start_prediction:end_prediction]
            
            f1 += calc_f1_score(answers, prediction)
            em += calc_em_score(answers, prediction)
        
        f1_score = 100.0 * f1 / total_count
        em_score = 100.0 * em / total_count
        avg = (f1_score+em_score)*0.5
        print("Avg: {:.4f},F1:{:.4f},EM:{:.4f},Total:{},Skip:{}".format(avg,f1_score,em_score,total_count,skip_count))
        return avg 

    # Training phase
    print("Start training.")
    batch_size = args.batch_size
    print("Batch size: ", batch_size)
    examples = read_examples(args.train_path)
    trainset = convert_examples_to_dataset(examples,args)
    random.shuffle(trainset)
    instances_num = len(trainset)

    input_ids = torch.LongTensor([sample[0] for sample in trainset])
    mask_ids = torch.LongTensor([sample[1] for sample in trainset])
    start_positions = torch.LongTensor([sample[2] for sample in trainset])
    end_positions = torch.LongTensor([sample[3] for sample in trainset])

    train_steps = int(instances_num * args.epochs_num / batch_size) + 1
   
    print("The number of training instances:", instances_num)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=train_steps)

    total_loss = 0.
    result = 0.0
    best_result = 0.0
    
    for epoch in range(1, args.epochs_num+1):
        model.train()
        
        for i, (input_ids_batch, mask_ids_batch, start_positions_batch, end_positions_batch) in enumerate(batch_loader(batch_size, input_ids, mask_ids, start_positions, end_positions)):
            model.zero_grad()
            input_ids_batch = input_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            start_positions_batch = start_positions_batch.to(device)
            end_positions_batch = end_positions_batch.to(device)

            loss, _, _ = model(input_ids_batch, mask_ids_batch, start_positions_batch, end_positions_batch)
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i+1, total_loss / args.report_steps))
                total_loss = 0.
            loss.backward()
            optimizer.step()
        result = evaluate(args, False)
        if result > best_result:
            best_result = result
            save_model(model, args.output_model_path)
        else:
            continue

    # Evaluation phase.
    if args.test_path is not None:
        print("Test set evaluation.")
        model = load_model(model, args.output_model_path)
        evaluate(args, True)


if __name__ == "__main__":
    main()
