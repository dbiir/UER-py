def model_opts(parser):
    parser.add_argument("--embedding", choices=["word", "pos", "seg", "sinusoidalpos", "dual"], default="word", nargs='+',
                        help="Embedding type.")
    parser.add_argument("--tgt_embedding", choices=["word", "pos", "seg", "sinusoidalpos", "dual"], default="word", nargs='+',
                        help="Target embedding type.")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Max sequence length for word embedding.")
    parser.add_argument("--relative_position_embedding", action="store_true",
                        help="Use relative position embedding.")
    parser.add_argument("--share_embedding", action="store_true",
                        help="Shared embedding and target embedding parameters.")
    parser.add_argument("--remove_embedding_layernorm", action="store_true",
                        help="Remove layernorm on embedding.")
    parser.add_argument("--factorized_embedding_parameterization", action="store_true", help="Factorized embedding parameterization.")
    parser.add_argument("--encoder", choices=["transformer", "rnn", "lstm", "gru", "birnn",
                                              "bilstm", "bigru", "gatedcnn", "dual"],
                        default="transformer", help="Encoder type.")
    parser.add_argument("--decoder", choices=[None, "transformer"], default=None, help="Decoder type.")
    parser.add_argument("--mask", choices=["fully_visible", "causal", "causal_with_prefix"], default="fully_visible",
                        help="Mask type.")
    parser.add_argument("--layernorm_positioning", choices=["pre", "post"], default="post",
                        help="Layernorm positioning.")
    parser.add_argument("--feed_forward", choices=["dense", "gated"], default="dense",
                        help="Feed forward type, specific to transformer model.")
    parser.add_argument("--relative_attention_buckets_num", type=int, default=32,
                        help="Buckets num of relative position embedding.")
    parser.add_argument("--remove_attention_scale", action="store_true",
                        help="Remove attention scale.")
    parser.add_argument("--remove_transformer_bias", action="store_true",
                        help="Remove bias on transformer layers.")
    parser.add_argument("--layernorm", choices=["normal", "t5"], default="normal",
                        help="Layernorm type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--parameter_sharing", action="store_true", help="Parameter sharing.")
    parser.add_argument("--has_residual_attention", action="store_true", help="Add residual attention.")
    parser.add_argument("--has_lmtarget_bias", action="store_true",
                        help="Add bias on output_layer for lm target.")
    parser.add_argument("--target", choices=["sp", "lm", "mlm", "bilm", "cls"], default="mlm", nargs='+',
                        help="The training target of the pretraining model.")
    parser.add_argument("--tie_weights", action="store_true",
                        help="Tie the word embedding and softmax weights.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")
    parser.add_argument("--prefix_lm_loss", action="store_true",
                        help="Only compute output loss when SFT.")


def log_opts(parser):
    parser.add_argument("--log_path", type=str, default=None,
                        help="Log file path, default no output file.")
    parser.add_argument("--log_level", choices=["ERROR", "INFO", "DEBUG", "NOTSET"], default="INFO",
                        help="Console log level. Verbosity: ERROR < INFO < DEBUG < NOTSET")
    parser.add_argument("--log_file_level", choices=["ERROR", "INFO", "DEBUG", "NOTSET"], default="INFO",
                        help="Log file level. Verbosity: ERROR < INFO < DEBUG < NOTSET")


def optimization_opts(parser):
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")
    parser.add_argument("--lr_decay", type=float, default=0.5,
                        help="Learning rate decay value.")
    parser.add_argument("--optimizer", choices=["adamw", "adafactor"],
                        default="adamw",
                        help="Optimizer type.")
    parser.add_argument("--scheduler", choices=["linear", "cosine", "cosine_with_restarts", "polynomial",
                                                "constant", "constant_with_warmup", "inverse_sqrt", "tri_stage"],
                        default="linear", help="Scheduler type.")


def training_opts(parser):
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=3,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")
    log_opts(parser)


def finetune_opts(parser):
    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="models/finetuned_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str, required=True,
                        help="Path of the devset.")
    parser.add_argument("--test_path", default=None, type=str,
                        help="Path of the testset.")
    parser.add_argument("--config_path", default="models/bert/base_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    model_opts(parser)

    # Optimization options.
    optimization_opts(parser)

    # Training options.
    training_opts(parser)


def infer_opts(parser):
    # Path options.
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path of the input model.")
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path of the testset.")
    parser.add_argument("--prediction_path", type=str, required=True,
                        help="Path of the prediction file.")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path of the config file.")

    # Model options.
    model_opts(parser)

    # Inference options.
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")


def tokenizer_opts(parser):
    parser.add_argument("--tokenizer", choices=["bert", "bpe", "char", "space", "xlmroberta"], default="bert",
                        help="Specify the tokenizer." 
                             "Original Google BERT uses bert tokenizer."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             "Original XLM-RoBERTa uses xlmroberta tokenizer."
                             )
    parser.add_argument("--vocab_path", default=None, type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--merges_path", default=None, type=str,
                        help="Path of the merges file.")
    parser.add_argument("--spm_model_path", default=None, type=str,
                        help="Path of the sentence piece model.")
    parser.add_argument("--do_lower_case", choices=["true", "false"], default="true",
                        help="Whether to lower case the input")


def tgt_tokenizer_opts(parser):
    parser.add_argument("--tgt_tokenizer", choices=["bert", "bpe", "char", "space", "xlmroberta"], default="bert",
                        help="Specify the tokenizer for target side.")
    parser.add_argument("--tgt_vocab_path", default=None, type=str,
                        help="Path of the target vocabulary file.")
    parser.add_argument("--tgt_merges_path", default=None, type=str,
                        help="Path of the target merges file.")
    parser.add_argument("--tgt_spm_model_path", default=None, type=str,
                        help="Path of the target sentence piece model.")
    parser.add_argument("--tgt_do_lower_case", choices=["true", "false"], default="true",
                        help="Whether to lower case the target input")


def adv_opts(parser):
    parser.add_argument("--use_adv", action="store_true",
                        help=".")
    parser.add_argument("--adv_type", choices=["fgm", "pgd"], default="fgm",
                        help="Specify the adversal training type.")
    parser.add_argument("--fgm_epsilon", type=float, default=1e-6,
                        help="Epsilon for FGM.")
    parser.add_argument("--pgd_k", type=int, default=3,
                        help="Steps for PGD.")
    parser.add_argument("--pgd_epsilon", type=float, default=1.,
                        help="Epsilon for PGD.")
    parser.add_argument("--pgd_alpha", type=float, default=0.3,
                        help="Alpha for PGD.")
