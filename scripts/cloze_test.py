"""
  This script provides an exmaple to wrap UER-py for cloze test.
  One character in a line is masked.
  We should use the target that contains MLM.
"""
import sys
import os
import torch
import argparse
import random

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.layers import *
from uer.encoders import *
from uer.targets import *
from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.model_loader import load_model
from uer.opts import infer_opts, tokenizer_opts


def mask_token(tokens, seq_length, tokenizer):
    """
    Mask a random token for prediction.
    """
    start = 1
    end = len(tokens) if len(tokens) < seq_length else seq_length
    mask_pos = random.randint(start, end-1)
    token = tokens[mask_pos]
    tokens[mask_pos] = tokenizer.convert_tokens_to_ids([MASK_TOKEN])[0]
    return (tokens, mask_pos, token)


def batch_loader(batch_size, src, seg, mask_pos, label):
    instances_num = src.size(0)                                                                                               
    for i in range(instances_num // batch_size):                                                                                    
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]                                                                
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]                                                                    
        mask_pos_batch = mask_pos[i * batch_size : (i + 1) * batch_size]                                                      
        label_batch = label[i * batch_size : (i + 1) * batch_size]                                                                
        yield src_batch, seg_batch, mask_pos_batch, label_batch                                                 
                                                                                                                                        
    if instances_num > instances_num // batch_size * batch_size:                                                                    
        src_batch = src[instances_num // batch_size * batch_size :, :]                                                          
        seg_batch = seg[instances_num // batch_size * batch_size :, :]                                                              
        mask_pos_batch = mask_pos[instances_num // batch_size * batch_size :]                                                
        label_batch = label[instances_num // batch_size * batch_size :]                                                          
        yield src_batch, seg_batch, mask_pos_batch, label_batch


def read_dataset(args, path):
    dataset = []
    PAD_ID = args.tokenizer.vocab.get(PAD_TOKEN)
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            src = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(line.strip()))
            if len(src) == 0:
                continue

            src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN]) + src + args.tokenizer.convert_tokens_to_ids([SEP_TOKEN])
            src, mask_pos, label = mask_token(src, args.seq_length, args.tokenizer)

            seg = [1] * len(src)
            if len(src) > args.seq_length:
                src = src[:args.seq_length]
                seg = seg[:args.seq_length]
            while len(src) < args.seq_length:
                src.append(PAD_ID)
                seg.append(PAD_ID)
            
            dataset.append((src, seg, mask_pos, label))
    return dataset


class ClozeTest(torch.nn.Module):
    def __init__(self, args):
        super(ClozeTest, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.target = str2target[args.target](args, len(args.tokenizer.vocab))
        self.act = str2act[args.hidden_act]

    def forward(self, src, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        output = self.act(self.target.mlm_linear_1(output))
        output = self.target.layer_norm(output)
        output = self.target.mlm_linear_2(output)
        prob = torch.nn.Softmax(dim=-1)(output)
        return prob


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    parser.add_argument("--target", choices=["bert", "mlm", "albert"], default="bert",
                        help="The training target of the pretraining model.")

    tokenizer_opts(parser)

    parser.add_argument("--topn", type=int, default=10,
                        help="Print top n nearest neighbours.")
    
    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build cloze test model.
    model = ClozeTest(args)
    model = load_model(model, args.load_model_path)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.                                                                 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                                                               
    model = model.to(device)                                                                                                            
    if torch.cuda.device_count() > 1:                                                                                                   
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))                                               
        model = torch.nn.DataParallel(model)
    model.eval()

    dataset = read_dataset(args, args.test_path)

    src = torch.LongTensor([sample[0] for sample in dataset])
    seg = torch.LongTensor([sample[1] for sample in dataset])

    mask_pos = [sample[2] for sample in dataset]
    label = [sample[3] for sample in dataset]
    
    f_pred = open(args.prediction_path, mode="w", encoding="utf-8")
               
    for i, (src_batch, seg_batch, mask_pos_batch, label_batch) in \
        enumerate(batch_loader(args.batch_size, src, seg, mask_pos, label)):
        src_batch = src_batch.to(device)
        seg_batch = seg_batch.to(device)
        prob = model(src_batch, seg_batch)

        for j, p in enumerate(mask_pos_batch):
            topn_ids = (-prob[j][p]).argsort()[:args.topn]

            sentence = "".join([args.tokenizer.convert_ids_to_tokens([token_id.item()])[0] for token_id in src_batch[j] if token_id != 0])
            pred_tokens = " ".join(args.tokenizer.convert_ids_to_tokens([token_id.item()])[0] for token_id in topn_ids)

            label_token = args.tokenizer.convert_ids_to_tokens([label_batch[j]])[0]

            f_pred.write(sentence + '\n')
            f_pred.write("Predicted answer: " + pred_tokens + '\n')
            f_pred.write("Correct answer: " + label_token + '\n')
            f_pred.write("\n")
    
    f_pred.close()
