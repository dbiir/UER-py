"""
  This script provides an exmaple to wrap UER-py for generate.
  We randomly give the beginning of story and use GPT to generate the full of story.
"""
import sys
import os
import torch
import torch.nn.functional as F
import argparse
import random

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 
sys.path.append(uer_dir) 
from uer.utils.act_fun import gelu
from uer.utils.constants import *
from uer.utils.tokenizers import *
from uer.layers.layer_norm import LayerNorm
from uer.utils.config import load_hyperparam
from uer.utils.vocab import Vocab
from uer.model_builder import build_model
from uer.layers import *
from uer.encoders import *
from uer.targets import *
from uer.utils import *

class S2sGenerationModel(torch.nn.Module):
    def __init__(self, args):
        super(S2sGenerationModel, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.target = str2target[args.target](args,len(args.tgt_vocab))
        # Open eval mode.
        self.eval()

    def forward(self, src, seg, tgt):
        emb = self.embedding(src, seg)
        #print("emb",emb.shape)
        memory_bank = self.encoder(emb, seg)
        #print("mem",memory_bank.shape)
        emb = self.target.embedding(tgt, None)
        hidden = self.target.decoder(memory_bank, emb, (src,))
        #print("hidden",hidden.shape)
        output = self.target.output_layer(hidden) 
        #print("output",output)
        return output

def read_dataset(args, path):
    dataset = []
    with open(path, mode="r", encoding="utf-8") as f:
        for idx,line in enumerate(f):
            src = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(line.strip()))        
            seg = [1] * len(src)
            tgt = [args.tgt_vocab.get("[CLS]")]
            if len(src) > args.seq_length:
                src = src[:args.seq_length]
                seg = seg[:args.seq_length]
            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)
            dataset.append((src, tgt, seg, line))

    return dataset


def top_k_top_p_filtering(logits, top_k, top_p):
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float("Inf")

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("Inf")
    return logits

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", type=str, required=True,
                        help="Path of the pretrained model.")
    #change default models/google_vocab.txt->models/google_zh_vocab.txt
    parser.add_argument("--vocab_path", default="models/google_zh_vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--tgt_vocab_path", default="models/google_zh_vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path of the input file, containing the beginning of a story.")
    parser.add_argument("--translation_path", type=str, required=True,
                        help="Path of the output file, containing the entire story.")
    parser.add_argument("--config_path", default="models/bert_base_config.json", type=str,
                        help="Path of the config file.")
    #add
    parser.add_argument("--has_lmtarget_bias", action="store_true",
                        help="Add bias on output_layer for lm target.")
    #add
    parser.add_argument("--tie_weights", action="store_true",
                        help="Tie the word embedding and softmax weights.")
    # Model options.
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--tgt_seq_length", type=int, default=128,
                        help="Sequence length.")

    #change default 0->70
    parser.add_argument("--top_k", type=int, default=70)
    #change default 0.6->0
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    #change add choice gpt
    parser.add_argument("--embedding", choices=["word", "word_pos", "word_pos_seg"], default="word_pos_seg",
                        help="Emebdding type.")
    parser.add_argument("--tgt_embedding", choices=["word", "word_pos", "word_pos_seg"], default="word_pos_seg",
                        help="Target embedding type.")
    #change add choice gpt2
    parser.add_argument("--encoder", choices=["transformer", "rnn", "lstm", "gru", \
                                              "birnn", "bilstm", "bigru", \
                                              "gatedcnn"], \
                                              default="transformer", help="Encoder type.")
    parser.add_argument("--decoder", choices=["transformer"], \
                                              default="transformer", help="Decoder type.")
    parser.add_argument("--layernorm_positioning", choices=["pre", "post"], default="post",
                        help="Layernorm positioning.")
    parser.add_argument("--target", choices=["lm", "seq2seq"], default="seq2seq",
                        help="The training target of the pretraining model.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder_type", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    #add
    parser.add_argument("--spm_model_path", default=None, type=str,
                        help="Path of the sentence piece model.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )
    parser.add_argument("--mask", choices=["fully_visible", "causal"], default="fully_visible",
                        help="Mask type.")
    parser.add_argument("--factorized_embedding_parameterization", action="store_true", help="Factorized embedding parameterization.")
    parser.add_argument("--parameter_sharing", action="store_true", help="Parameter sharing.")
 
    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)
    
    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Load Vocabulary
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    if args.target == "seq2seq":
        tgt_vocab = Vocab()
        tgt_vocab.load(args.tgt_vocab_path)
        args.tgt_vocab = tgt_vocab

    model = S2sGenerationModel(args)
    # Load pretrained model.
    pretrained_model_dict = torch.load(args.pretrained_model_path)
    model.load_state_dict(pretrained_model_dict, strict=True)
    
    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    dataset = read_dataset(args, args.input_path)

    print("The number of translation instances: ", instances_num)

    model.eval()

    with open(args.translation_path, mode="w", encoding="utf-8") as f:
        for i in range(len(dataset)):
            src_tensor = torch.LongTensor([dataset[i][0]]).to(device)
            tgt_tensor = torch.LongTensor([dataset[i][1]]).to(device)
            seg_tensor = torch.LongTensor([dataset[i][2]]).to(device)

            for j in range(args.tgt_seq_length-1):
                outputs = model(src_tensor, seg_tensor, tgt_tensor)
                next_token_logits = outputs[0][-1] / args.temperature
                filtered_logits = top_k_top_p_filtering(next_token_logits, args.top_k, args.top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                
                tgt_tensor = torch.cat([tgt_tensor, next_token.view(1,1)], dim=1)
                #seg_tensor = torch.cat([seg_tensor, torch.tensor([[1]])], dim=1)

            f.write(dataset[i][3])
            generated_sentence = "".join([args.tgt_vocab.i2w[token_id] for token_id in tgt_tensor[0]])
            f.write(generated_sentence + "\n")
            f.write("\n")



