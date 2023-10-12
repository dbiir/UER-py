"""
  This script provides an example to use prompt for classification inference.
"""
import sys
import os

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.model_loader import load_model
from uer.opts import infer_opts, tokenizer_opts
from finetune.run_classifier_prompt import *


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.rstrip("\r\n").split("\t")):
                    columns[column_name] = i
                continue
            line = line.rstrip("\r\n").split("\t")
            mask_position = -1
            tgt_token_id = [1]
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
            PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
            while len(src) < args.seq_length:
                src.append(PAD_ID)
                seg.append(0)
            tgt = [0] * len(src)
            tgt[mask_position] = tgt_token_id[0]
            dataset.append((src, tgt, seg))

    return dataset


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    tokenizer_opts(parser)

    parser.add_argument("--output_logits", action="store_true", help="Write logits to output file.")
    parser.add_argument("--output_prob", action="store_true", help="Write probabilities to output file.")

    parser.add_argument("--prompt_id", type=str, default="chnsenticorp_char")
    parser.add_argument("--prompt_path", type=str, default="models/prompts.json")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    process_prompt_template(args)

    answer_position = [0] * len(args.tokenizer.vocab)
    for answer in args.answer_word_dict_inv:
        answer_position[int(args.tokenizer.vocab[answer])] = 1
    args.answer_position = torch.LongTensor(answer_position)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Build classification model and load parameters.
    model = ClozeTest(args)
    model = load_model(model, args.load_model_path)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    model = model.to(args.device)
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    dataset = read_dataset(args, args.test_path)

    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    batch_size = args.batch_size
    instances_num = src.size()[0]

    print("The number of prediction instances: ", instances_num)

    model.eval()

    with open(args.prediction_path, mode="w", encoding="utf-8") as f:
        f.write("label")
        if args.output_logits:
            f.write("\t" + "logits")
        if args.output_prob:
            f.write("\t" + "prob")
        f.write("\n")
        for _, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):
            src_batch = src_batch.to(args.device)
            tgt_batch = tgt_batch.to(args.device)
            seg_batch = seg_batch.to(args.device)
            with torch.no_grad():
                _, pred, logits = model(src_batch, tgt_batch, seg_batch)

            logits = logits[:, args.answer_position > 0]
            prob = nn.Softmax(dim=1)(logits)
            logits = logits.cpu().numpy().tolist()
            prob = prob.cpu().numpy().tolist()

            for j in range(len(pred)):
                f.write(str(pred[j]))
                if args.output_logits:
                    f.write("\t" + " ".join([str(v) for v in logits[j]]))
                if args.output_prob:
                    f.write("\t" + " ".join([str(v) for v in prob[j]]))
                f.write("\n")


if __name__ == "__main__":
    main()
