import torch


def init_env(args):
    if args.dist_train:
        # Initialize multiprocessing distributed training environment.
        args.global_rank = args.gpu_ranks[args.local_rank]
        torch.distributed.init_process_group(backend=args.backend,
                                init_method=args.master_ip,
                                world_size=args.world_size,
                                rank=args.global_rank)
    elif args.single_gpu:
        args.global_rank = None
    else:
        args.global_rank = None

    return None
