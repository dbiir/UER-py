# -*- encoding:utf-8 -*-
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from uer.model_saver import save_model
from uer.model_builder import build_model
from uer.utils.optimizers import BertAdam
from uer.utils.data import *
from uer.utils.vocab import Vocab
from uer.utils.seed import set_seed


def train_and_validate(args):
    set_seed(args.seed)

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build model.
    model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)  
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)

    if args.dist_train:
        # Multiprocessing distributed mode.
        mp.spawn(worker, nprocs=args.ranks_num, args=(args.gpu_ranks, args, model), daemon=False)
    elif args.single_gpu:
        # Single GPU mode.
        worker(args.gpu_id, None, args, model)
    else:
        # CPU mode.
        worker(None, None, args, model)


def worker(gpu_id, gpu_ranks, args, model):
    """
    Args:
        gpu_id: The id of GPU for single GPU mode;
            The id of process (and GPU) for multiprocessing distributed mode.
        gpu_ranks: List of ranks of each process.
    """
    set_seed(args.seed)

    if gpu_ranks is None:
        train_loader = globals()[args.target.capitalize() + "DataLoader"](args, args.dataset_path, args.batch_size, gpu_id, 1, True)
    else:
        train_loader = globals()[args.target.capitalize() + "DataLoader"](args, args.dataset_path, args.batch_size, gpu_id, len(gpu_ranks), True)

    if gpu_id is not None: 
        torch.cuda.set_device(gpu_id)
        model.cuda(gpu_id)

    # Build optimizer.
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=args.total_steps)

    rank = -1 # Each process has a unique rank in multiprocessing distributed mode.
    if args.dist_train:
        rank = gpu_ranks[gpu_id]
        # Initialize multiprocessing distributed training environment.
        dist.init_process_group(backend=args.backend,
                                init_method=args.master_ip,
                                world_size=args.world_size,
                                rank=rank)
        model = DistributedDataParallel(model, device_ids=[gpu_id])
        print("Worker %d is training ... " % rank)
    else:
        print("Worker is training ...")
    
    globals().get("train_"+args.target)(args, gpu_id, rank, train_loader, model, optimizer)
    

def train_bert(args, gpu_id, rank, loader, model, optimizer):
    model.train()
    start_time = time.time()
    total_loss, total_loss_mlm, total_loss_nsp = 0., 0., 0.
    # Calculate MLM accuracy.
    total_correct_mlm, total_denominator = 0., 0. 
    # Calculate NSP accuracy.
    total_correct_nsp, total_instances = 0., 0.
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        src, tgt_mlm, tgt_nsp, seg = next(loader_iter)

        if gpu_id is not None:
            src = src.cuda(gpu_id)
            tgt_mlm = tgt_mlm.cuda(gpu_id)
            tgt_nsp = tgt_nsp.cuda(gpu_id)
            seg = seg.cuda(gpu_id)
        
        # Forward.
        loss_info = model(src, (tgt_mlm, tgt_nsp), seg)
        loss_mlm, loss_nsp, correct_mlm, correct_nsp, denominator = loss_info
        
        # Backward.
        loss = loss_mlm + loss_nsp
        total_loss += loss.item()
        total_loss_mlm += loss_mlm.item()
        total_loss_nsp += loss_nsp.item()
        total_correct_mlm += correct_mlm.item()
        total_correct_nsp += correct_nsp.item()
        total_denominator += denominator.item()
        total_instances += src.size(0)

        loss = loss / args.accumulation_steps
        loss.backward()

        if steps % args.accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()
        
        if steps % args.report_steps == 0  and \
            (not args.dist_train or (args.dist_train and rank == 0)):

            loss = total_loss / args.report_steps
            loss_mlm = total_loss_mlm / args.report_steps
            loss_nsp = total_loss_nsp / args.report_steps

            elapsed = time.time() - start_time

            done_tokens = \
                args.batch_size * src.size(1) * args.report_steps * args.world_size \
                if args.dist_train \
                else args.batch_size * src.size(1) * args.report_steps

            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| loss_mlm: {:3.3f}"
                  "| loss_nsp: {:3.3f}"
                  "| acc_mlm: {:3.3f}"
                  "| acc_nsp: {:3.3f}".format(
                    steps, 
                    total_steps, 
                    done_tokens / elapsed, 
                    loss, 
                    loss_mlm,
                    loss_nsp,
                    total_correct_mlm / total_denominator,
                    total_correct_nsp  / total_instances))
            
            total_loss, total_loss_mlm, total_loss_nsp = 0., 0., 0.
            total_correct_mlm, total_denominator = 0., 0.
            total_correct_nsp, total_instances = 0., 0.

            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))

        steps += 1


def train_lm(args, gpu_id, rank, loader, model, optimizer):
    model.train()
    start_time = time.time()
    total_loss = 0.
    # Calculate MLM accuracy.
    total_correct, total_denominator = 0., 0. 
    # Calculate NSP accuracy.
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        src, tgt, seg = next(loader_iter)

        if gpu_id is not None:
            src = src.cuda(gpu_id)
            tgt = tgt.cuda(gpu_id)
            seg = seg.cuda(gpu_id)
        
        # Forward.
        loss_info = model(src, tgt, seg)
        loss, correct, denominator = loss_info
        
        # Backward.
        total_loss += loss.item()
        total_correct += correct.item()
        total_denominator += denominator.item()

        loss = loss / args.accumulation_steps
        loss.backward()

        if steps % args.accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()
        
        if steps % args.report_steps == 0  and \
            (not args.dist_train or (args.dist_train and rank == 0)):

            loss = total_loss / args.report_steps

            elapsed = time.time() - start_time

            done_tokens = \
                args.batch_size * src.size(1) * args.report_steps * args.world_size \
                if args.dist_train \
                else args.batch_size * src.size(1) * args.report_steps

            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| acc: {:3.3f}".format(
                    steps, 
                    total_steps, 
                    done_tokens / elapsed, 
                    loss, 
                    total_correct / total_denominator))
            
            total_loss = 0.
            total_correct, total_denominator = 0., 0.

            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))

        steps += 1


def train_bilm(args, gpu_id, rank, loader, model, optimizer):
    model.train()
    start_time = time.time()
    total_loss, total_loss_forward, total_loss_backward = 0., 0., 0.
    # Calculate BiLM accuracy.
    total_correct_forward, total_correct_backward, total_denominator = 0., 0., 0. 
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        src, tgt_forward, tgt_backward, seg = next(loader_iter)

        if gpu_id is not None:
            src = src.cuda(gpu_id)
            tgt_forward = tgt_forward.cuda(gpu_id)
            tgt_backward = tgt_backward.cuda(gpu_id)
            seg = seg.cuda(gpu_id)
        
        # Forward.
        loss_info = model(src, (tgt_forward, tgt_backward), seg)
        loss_forward, loss_backward, correct_forward, correct_backward, denominator = loss_info
        
        # Backward.
        loss = loss_forward + loss_backward
        total_loss += loss.item()
        total_loss_forward += loss_forward.item()
        total_loss_backward += loss_backward.item()
        total_correct_forward += correct_forward.item()
        total_correct_backward += correct_backward.item()
        total_denominator += denominator.item()

        loss = loss / args.accumulation_steps
        loss.backward()

        if steps % args.accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()
        
        if steps % args.report_steps == 0  and \
            (not args.dist_train or (args.dist_train and rank == 0)):

            loss = total_loss / args.report_steps

            elapsed = time.time() - start_time

            done_tokens = \
                args.batch_size * src.size(1) * args.report_steps * args.world_size \
                if args.dist_train \
                else args.batch_size * src.size(1) * args.report_steps

            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| loss_forward {:3.3f}"
                  "| loss_backward {:3.3f}"
                  "| acc_forward: {:3.3f}"
                  "| acc_backward: {:3.3f}".format(
                    steps, 
                    total_steps, 
                    done_tokens / elapsed, 
                    loss,
                    loss_forward,
                    loss_backward,
                    total_correct_forward / total_denominator,
                    total_correct_backward / total_denominator))
            
            total_loss, total_loss_forward, total_loss_backward = 0., 0., 0.
            total_correct_forward, total_correct_backward, total_denominator = 0., 0., 0. 

            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))

        steps += 1


def train_cls(args, gpu_id, rank, loader, model, optimizer):
    model.train()
    start_time = time.time()
    total_loss = 0.
    total_correct, total_instances = 0., 0.
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        src, tgt, seg = next(loader_iter)

        if gpu_id is not None:
            src = src.cuda(gpu_id)
            tgt = tgt.cuda(gpu_id)
            seg = seg.cuda(gpu_id)
        
        # Forward.
        loss_info = model(src, tgt, seg)
        loss, correct = loss_info
        
        # Backward.
        total_loss += loss.item()
        total_correct += correct.item()
        total_instances += src.size(0)

        loss = loss / args.accumulation_steps
        loss.backward()

        if steps % args.accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()
        
        if steps % args.report_steps == 0  and \
            (not args.dist_train or (args.dist_train and rank == 0)):

            loss = total_loss / args.report_steps

            elapsed = time.time() - start_time

            done_tokens = \
                args.batch_size * src.size(1) * args.report_steps * args.world_size \
                if args.dist_train \
                else args.batch_size * src.size(1) * args.report_steps

            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| acc: {:3.3f}".format(
                    steps, 
                    total_steps, 
                    done_tokens / elapsed, 
                    loss, 
                    total_correct / total_instances))
            
            total_loss = 0.
            total_correct = 0.
            total_instances = 0.

            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))

        steps += 1


def train_mlm(args, gpu_id, rank, loader, model, optimizer):
    model.train()
    start_time = time.time()
    total_loss, total_loss_mlm, total_loss_nsp = 0., 0., 0.
    # Calculate MLM accuracy.
    total_correct, total_denominator = 0., 0. 
    # Calculate NSP accuracy.
    total_instances = 0., 0.
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        src, tgt, seg = next(loader_iter)

        if gpu_id is not None:
            src = src.cuda(gpu_id)
            tgt = tgt.cuda(gpu_id)
            seg = seg.cuda(gpu_id)
        
        # Forward.
        loss_info = model(src, tgt, seg)
        loss, correct, denominator = loss_info
        
        # Backward.
        total_loss += loss.item()
        total_correct += correct.item()
        total_denominator += denominator.item()

        loss = loss / args.accumulation_steps
        loss.backward()

        if steps % args.accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()
        
        if steps % args.report_steps == 0  and \
            (not args.dist_train or (args.dist_train and rank == 0)):

            loss = total_loss / args.report_steps

            elapsed = time.time() - start_time

            done_tokens = \
                args.batch_size * src.size(1) * args.report_steps * args.world_size \
                if args.dist_train \
                else args.batch_size * src.size(1) * args.report_steps

            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| acc: {:3.3f}".format(
                    steps, 
                    total_steps, 
                    done_tokens / elapsed, 
                    loss, 
                    total_correct / total_denominator))
            
            total_loss = 0.
            total_correct, total_denominator = 0., 0.

            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))

        steps += 1


def train_nsp(args, gpu_id, rank, loader, model, optimizer):
    model.train()
    start_time = time.time()
    total_loss = 0.
    total_correct, total_instances = 0., 0.
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        src, tgt, seg = next(loader_iter)

        if gpu_id is not None:
            src = src.cuda(gpu_id)
            tgt = tgt.cuda(gpu_id)
            seg = seg.cuda(gpu_id)
        
        # Forward.
        loss_info = model(src, tgt, seg)
        loss, correct = loss_info
        
        # Backward.
        total_loss += loss.item()
        total_correct += correct.item()
        total_instances += src.size(0)

        loss = loss / args.accumulation_steps
        loss.backward()

        if steps % args.accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()
        
        if steps % args.report_steps == 0  and \
            (not args.dist_train or (args.dist_train and rank == 0)):

            loss = total_loss / args.report_steps

            elapsed = time.time() - start_time

            done_tokens = \
                args.batch_size * src.size(1) * args.report_steps * args.world_size \
                if args.dist_train \
                else args.batch_size * src.size(1) * args.report_steps

            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| acc: {:3.3f}".format(
                    steps, 
                    total_steps, 
                    done_tokens / elapsed, 
                    loss, 
                    total_correct / total_instances))
            
            total_loss = 0.
            total_correct = 0.
            total_instances = 0.

            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))

        steps += 1


def train_s2s(args, gpu_id, rank, loader, model, optimizer):
    model.train()
    start_time = time.time()
    total_loss= 0.
    total_correct, total_denominator = 0., 0. 
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        src, tgt, seg = next(loader_iter)

        if gpu_id is not None:
            src = src.cuda(gpu_id)
            tgt = tgt.cuda(gpu_id)
            seg = seg.cuda(gpu_id)
        
        # Forward.
        loss_info = model(src, tgt, seg)
        loss, correct, denominator = loss_info
        
        # Backward.
        total_loss += loss.item()
        total_correct += correct.item()
        total_denominator += denominator.item()

        loss = loss / args.accumulation_steps
        loss.backward()

        if steps % args.accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()
        
        if steps % args.report_steps == 0  and \
            (not args.dist_train or (args.dist_train and rank == 0)):

            loss = total_loss / args.report_steps

            elapsed = time.time() - start_time

            done_tokens = \
                args.batch_size * src.size(1) * args.report_steps * args.world_size \
                if args.dist_train \
                else args.batch_size * src.size(1) * args.report_steps

            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| acc: {:3.3f}".format(
                    steps, 
                    total_steps, 
                    done_tokens / elapsed, 
                    loss, 
                    total_correct / total_denominator))
            
            total_loss = 0.
            total_correct, total_denominator = 0., 0.

            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))

        steps += 1