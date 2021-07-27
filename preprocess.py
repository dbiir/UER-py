import argparse
import six
from packaging import version
from uer.utils.data import *
from uer.utils import *
from uer.opts import *


assert version.parse(six.__version__) >= version.parse("1.12.0")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--corpus_path", type=str, required=True,
                        help="Path of the corpus for pretraining.")
    parser.add_argument("--dataset_path", type=str, default="dataset.pt",
                        help="Path of the preprocessed dataset.")

    # Preprocess options.
    tokenizer_opts(parser)
    tgt_tokenizer_opts(parser)
    parser.add_argument("--processes_num", type=int, default=1,
                        help="Split the whole dataset into `processes_num` parts, "
                             "and process them with `processes_num` processes.")
    parser.add_argument("--target", choices=["bert", "lm", "mlm", "bilm", "albert", "seq2seq", "t5", "cls", "prefixlm", "gsg", "bart"], default="bert",
                        help="The training target of the pretraining model.")
    parser.add_argument("--docs_buffer_size", type=int, default=100000,
                        help="The buffer size of documents in memory, specific to targets that require negative sampling.")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length of instances.")
    parser.add_argument("--tgt_seq_length", type=int, default=128, help="Target sequence length of instances.")
    parser.add_argument("--dup_factor", type=int, default=5,
                        help="Duplicate instances multiple times.")
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of truncating sequence."
                             "The larger value, the higher probability of using short (truncated) sequence.")
    parser.add_argument("--full_sentences", action="store_true", help="Full sentences.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")

    # Masking options.
    parser.add_argument("--dynamic_masking", action="store_true", help="Dynamic masking.")
    parser.add_argument("--whole_word_masking", action="store_true", help="Whole word masking.")
    parser.add_argument("--span_masking", action="store_true", help="Span masking.")
    parser.add_argument("--span_geo_prob", type=float, default=0.2,
                        help="Hyperparameter of geometric distribution for span masking.")
    parser.add_argument("--span_max_length", type=int, default=10,
                        help="Max length for span masking.")

    # Sentence selection strategy options.
    parser.add_argument("--sentence_selection_strategy", choices=["lead", "random"], default="lead",
                        help="Sentence selection strategy for gap-sentences generation task.")

    args = parser.parse_args()

    # Dynamic masking.
    if args.dynamic_masking:
        args.dup_factor = 1

    # Build tokenizer.
    tokenizer = str2tokenizer[args.tokenizer](args)
    if args.target == "seq2seq":
        args.tgt_tokenizer = str2tokenizer[args.tgt_tokenizer](args, False)

    # Build and save dataset.
    dataset = str2dataset[args.target](args, tokenizer.vocab, tokenizer)
    dataset.build_and_save(args.processes_num)


if __name__ == "__main__":
    main()
