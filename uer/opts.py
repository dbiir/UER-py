def model_opts(parser):
    parser.add_argument("--embedding", choices=["word", "word_pos", "word_pos_seg", "word_sinusoidalpos"], default="word_pos_seg",
                        help="Emebdding type.")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Max sequence length for word embedding.")
    parser.add_argument("--relative_position_embedding", action="store_true",
                        help="Use relative position embedding.")
    parser.add_argument("--relative_attention_buckets_num", type=int, default=32,
                        help="Buckets num of relative position embedding.")
    parser.add_argument("--remove_embedding_layernorm", action="store_true",
                        help="Remove layernorm on embedding.")
    parser.add_argument("--remove_attention_scale", action="store_true",
                        help="Remove attention scale.")
    parser.add_argument("--encoder", choices=["transformer", "rnn", "lstm", "gru",
                                              "birnn", "bilstm", "bigru",
                                              "gatedcnn"],
                        default="transformer", help="Encoder type.")
    parser.add_argument("--mask", choices=["fully_visible", "causal", "causal_with_prefix"], default="fully_visible",
                        help="Mask type.")
    parser.add_argument("--layernorm_positioning", choices=["pre", "post"], default="post",
                        help="Layernorm positioning.")
    parser.add_argument("--feed_forward", choices=["dense", "gated"], default="dense",
                        help="Feed forward type, specific to transformer model.")
    parser.add_argument("--remove_transformer_bias", action="store_true",
                        help="Remove bias on transformer layers.")
    parser.add_argument("--layernorm", choices=["normal", "t5"], default="normal",
                        help="Layernorm type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--factorized_embedding_parameterization", action="store_true", help="Factorized embedding parameterization.")
    parser.add_argument("--parameter_sharing", action="store_true", help="Parameter sharing.")
    parser.add_argument("--has_residual_attention", action="store_true", help="Add residual attention.")


def optimization_opts(parser):
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")
    parser.add_argument("--fp16", action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit.")
    parser.add_argument("--fp16_opt_level", choices=["O0", "O1", "O2", "O3" ], default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--optimizer", choices=["adamw", "adafactor"],
                        default="adamw",
                        help="Optimizer type.")
    parser.add_argument("--scheduler", choices=["linear", "cosine", "cosine_with_restarts", "polynomial",
                                                "constant", "constant_with_warmup"],
                        default="linear", help="Scheduler type.")


def training_opts(parser):
    parser.add_argument("--batch_size", type=int, default=32,                                                             
                        help="Batch size.")                                                                               
    parser.add_argument("--seq_length", type=int, default=128,                                                            
                        help="Sequence length.")                                                                          
    parser.add_argument("--dropout", type=float, default=0.5,                                                             
                        help="Dropout.")                                                                                  
    parser.add_argument("--epochs_num", type=int, default=3,                                                              
                        help="Number of epochs.")                                                                         
    parser.add_argument("--report_steps", type=int, default=100,                                                          
                        help="Specific steps to print prompt.")                                                           
    parser.add_argument("--seed", type=int, default=7,                                                                    
                        help="Random seed.")


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
    parser.add_argument("--config_path", default="models/bert/base_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    model_opts(parser)

    # Inference options.
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")


def tokenizer_opts(parser):
    parser.add_argument("--tokenizer", choices=["bert", "char", "space", "xlmroberta"], default="bert",
                        help="Specify the tokenizer." 
                             "Original Google BERT uses bert tokenizer."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             "Original XLM-RoBERTa uses xlmroberta tokenizer."
                             )
    parser.add_argument("--vocab_path", default=None, type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--spm_model_path", default=None, type=str,
                        help="Path of the sentence piece model.")


def tgt_tokenizer_opts(parser):
    parser.add_argument("--tgt_tokenizer", choices=["bert", "char", "space", "xlmroberta"], default="bert",
                        help="Specify the tokenizer for target side.")
    parser.add_argument("--tgt_vocab_path", default=None, type=str,
                        help="Path of the target vocabulary file.")
    parser.add_argument("--tgt_spm_model_path", default=None, type=str,
                        help="Path of the target sentence piece model.")
