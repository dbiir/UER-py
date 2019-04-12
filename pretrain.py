# -*- encoding:utf-8 -*-
import os
import json
import torch
import argparse
import torch.multiprocessing as mp
import uer.trainer as trainer
from uer.utils.config import save_hyperparam, load_hyperparam


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Path options.
    parser.add_argument("--dataset_path", type=str, default="dataset",
                        help="Base path of the preprocessed dataset.")
    parser.add_argument("--vocab_path", type=str, required=True,
                        help="Path of the vocabulary file.")
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", type=str, required=True,
                        help="Path of the output model.")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Config file of model hyper-parameters.")

    # Training and saving options. 
    parser.add_argument("--total_steps", type=int, default=100000,
                        help="Total training steps.")
    parser.add_argument("--save_checkpoint_steps", type=int, default=10000,
                        help="Specific steps to save model checkpoint.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Specific steps to accumulate gradient.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size. The actual batch_size is [batch_size x world_size x accumulation_steps].")
    parser.add_argument("--instances_buffer_size", type=int, default=1000000,
                        help="The buffer size of instances in memory.")

    # Model options.
    parser.add_argument("--emb_size", type=int, default=768, help="Embedding dimension.")
    parser.add_argument("--hidden_size", type=int, default=768,  help="Hidden state dimension.")
    parser.add_argument("--feedforward_size", type=int, default=3072, help="Feed forward layer dimension.")
    parser.add_argument("--kernel_size", type=int, default=3,  help="Kernel size for CNN.")
    parser.add_argument("--heads_num", type=int, default=12, help="The number of heads in multiple-head attention.")
    parser.add_argument("--layers_num", type=int, default=12, help="The number of encoder layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout value.")
    parser.add_argument("--seed", type=int, default=7,  help="Random seed.")
    parser.add_argument("--encoder_type", choices=["bert", "lstm", "gru", \
                                                   "cnn", "gatedcnn", "attn", \
                                                   "rcnn", "crnn", "gpt"], \
                                                   default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--target", choices=["bert", "lm", "cls", "mlm", "nsp", "s2s"], default="bert",
                        help="The training target of the pretraining model.")
    parser.add_argument("--labels_num", type=int, default=2, help="Specific to classification target.")

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Initial learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1, help="Warm up value.")

    # GPU options.
    parser.add_argument("--world_size", type=int, default=1, help="Total number of processes (GPUs) for training.")
    parser.add_argument("--gpu_ranks", default=[], nargs='+', type=int, help="List of ranks of each process."
                        " Each process has a unique integer rank whose value in the interval [0, world_size], and runs in a single GPU.")
    parser.add_argument("--master_ip", default="tcp://localhost:12345", type=str, help="IP-Port of master for training.")
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl", type=str, help="Distributed backend.")
    
    args = parser.parse_args()

    # Load hyper-parameters from config file. 
    if args.config_path:
        load_hyperparam(args)

    ranks_num = len(args.gpu_ranks)

    if args.world_size > 1:
        assert torch.cuda.is_available(), "No available GPUs." 
        assert ranks_num <= args.world_size, "Started processes exceed `world_size` upper limit." 
        assert ranks_num <= torch.cuda.device_count(), "Started processes exceeds the available GPUs." 
        # Multiprocessing distributed mode.
        args.dist_train = True
        args.ranks_num = ranks_num
        print("Using distributed mode for training.")
    elif args.world_size == 1 and ranks_num == 1:
        assert torch.cuda.is_available(), "No available GPUs." 
        # Single GPU mode.
        gpu_id = args.gpu_ranks[0]
        assert gpu_id <= torch.cuda.device_count(), "Invalid specified GPU device." 
        args.dist_train = False
        args.single_gpu = True
        args.gpu_id = gpu_id
        print("Using single GPU:%d for training." % gpu_id)
    else:
        # CPU mode.
        args.dist_train = False
        args.single_gpu = False
        print("Using CPU mode for training.")

    trainer.train_and_validate(args)


if __name__ == "__main__":
    main()
