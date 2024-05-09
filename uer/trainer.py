import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from uer.initialize import init_env
from uer.model_loader import load_model
from uer.model_saver import save_model
from uer.model_builder import build_model
from uer.utils.logging import init_logger
from uer.utils.optimizers import *
from uer.utils import *
from uer.utils.vocab import Vocab
from uer.utils.seed import set_seed


def init_model(args):
    # Build model.
    model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model = load_model(model, args.pretrained_model_path)
    else:
        # Initialize with normal distribution.
        if args.deep_init:
            scaled_factor = 1 / math.sqrt(2.0 * args.layers_num)
            for n, p in list(model.named_parameters()):
                if "gamma" not in n and "beta" not in n:
                    if "linear_2.weight" in n or "final_linear.weight" in n:
                        p.data.normal_(0, 0.02 * scaled_factor)
                    elif "linear_2.bias" in n or "final_linear.bias" in n:
                        p.data.zero_()
                    else:
                        p.data.normal_(0, 0.02)
        else:
            for n, p in list(model.named_parameters()):
                if "gamma" not in n and "beta" not in n:
                    p.data.normal_(0, 0.02)
    return model


def init_optimizer(args, model):
    # Build optimizer.
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta", "layer_norm"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]

    if args.optimizer in ["adamw"]:
        custom_optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        custom_optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        custom_scheduler = str2scheduler[args.scheduler](custom_optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        custom_scheduler = str2scheduler[args.scheduler](custom_optimizer, args.total_steps*args.warmup)
    elif args.scheduler in ["tri_stage"]:
        custom_scheduler = str2scheduler[args.scheduler](custom_optimizer, args.total_steps*args.warmup, args.total_steps*args.lr_decay, args.total_steps)
    else:
        custom_scheduler = str2scheduler[args.scheduler](custom_optimizer, args.total_steps*args.warmup, args.total_steps)

    return custom_optimizer, custom_scheduler


def train_and_validate(args):
    set_seed(args.seed)

    # Load vocabulary.
    if args.data_processor == "mt":
        args.tgt_tokenizer = str2tokenizer[args.tgt_tokenizer](args, is_src=False)
        args.tgt_vocab = args.tgt_tokenizer.vocab

    args.tokenizer = str2tokenizer[args.tokenizer](args)
    args.vocab = args.tokenizer.vocab

    if args.dist_train:
        # Multiprocessing distributed mode.
        mp.spawn(worker, nprocs=args.ranks_num, args=(args.gpu_ranks, args), daemon=False)
    elif args.single_gpu:
        # Single GPU mode.
        worker(args.local_rank, None, args)
    else:
        # CPU mode.
        worker(None, None, args)


class Trainer(object):
    def __init__(self, args):
        self.current_step = 1
        self.total_steps = args.total_steps
        self.accumulation_steps = args.accumulation_steps
        self.report_steps = args.report_steps
        self.save_checkpoint_steps = args.save_checkpoint_steps

        self.output_model_path = args.output_model_path

        self.start_time = time.time()
        self.total_loss = 0.0

        self.dist_train = args.dist_train
        self.batch_size = args.batch_size
        self.world_size = args.world_size
        self.logger = args.logger

    def forward_propagation(self, batch, model):

        raise NotImplementedError

    def report_and_reset_stats(self):

        raise NotImplementedError

    def train(self, args, local_rank, global_rank, loader, model, optimizer, scheduler):
        model.train()
        loader_iter = iter(loader)
        while True:
            if self.current_step == self.total_steps + 1:
                break
            batch = list(next(loader_iter))
            self.seq_length = batch[0].size(1)
            if local_rank is not None:
                for i in range(len(batch)):
                    batch[i] = batch[i].cuda(local_rank)

            loss = self.forward_propagation(batch, model)

            loss.backward()

            if self.current_step % self.accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            if self.current_step % self.report_steps == 0 and \
                    (not self.dist_train or (self.dist_train and global_rank == 0)):
                self.report_and_reset_stats()
                self.start_time = time.time()
           
            if self.current_step % self.save_checkpoint_steps == 0 and \
                    (not self.dist_train or (self.dist_train and global_rank == 0)):
                save_model(model, self.output_model_path + "-" + str(self.current_step))

            self.current_step += 1


class MlmTrainer(Trainer):
    def __init__(self, args):
        super(MlmTrainer, self).__init__(args)
        self.total_correct = 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        src, tgt, seg = batch
        loss_info = model(src, tgt, seg)
        loss, correct, denominator = loss_info
        self.total_loss += loss.item()
        self.total_correct += correct.item()
        self.total_denominator += denominator.item()
        loss = loss / self.accumulation_steps
        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size
        self.logger.info("| {:8d}/{:8d} steps"
              "| {:8.2f} tokens/s"
              "| loss {:7.2f}"
              "| acc: {:3.3f}".format(
                  self.current_step,
                  self.total_steps,
                  done_tokens / (time.time() - self.start_time),
                  self.total_loss / self.report_steps,
                  self.total_correct / self.total_denominator))

        self.total_loss = 0.0
        self.total_correct = 0.0
        self.total_denominator = 0.0


class BertTrainer(Trainer):
    def __init__(self, args):
        super(BertTrainer, self).__init__(args)
        self.total_loss_sp = 0.0
        self.total_correct_sp = 0.0
        self.total_instances = 0.0

        self.total_loss_mlm = 0.0
        self.total_correct_mlm = 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        src, tgt_mlm, tgt_sp, seg = batch
        tgt = {"mlm": tgt_mlm, "sp": tgt_sp}
        loss_info = model(src, tgt, seg)
        loss_mlm, correct_mlm, denominator = loss_info["mlm"]
        loss_sp, correct_sp = loss_info["sp"]
        loss = loss_mlm + loss_sp
        self.total_loss += loss.item()
        self.total_loss_mlm += loss_mlm.item()
        self.total_loss_sp += loss_sp.item()
        self.total_correct_mlm += correct_mlm.item()
        self.total_correct_sp += correct_sp.item()
        self.total_denominator += denominator.item()
        self.total_instances += src.size(0)
        loss = loss / self.accumulation_steps

        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size

        self.logger.info("| {:8d}/{:8d} steps"
              "| {:8.2f} tokens/s"
              "| loss {:7.2f}"
              "| loss_mlm: {:3.3f}"
              "| loss_sp: {:3.3f}"
              "| acc_mlm: {:3.3f}"
              "| acc_sp: {:3.3f}".format(
                  self.current_step,
                  self.total_steps,
                  done_tokens / (time.time() - self.start_time),
                  self.total_loss / self.report_steps,
                  self.total_loss_mlm / self.report_steps,
                  self.total_loss_sp / self.report_steps,
                  self.total_correct_mlm / self.total_denominator,
                  self.total_correct_sp / self.total_instances))

        self.total_loss, self.total_loss_mlm, self.total_loss_sp = 0.0, 0.0, 0.0
        self.total_correct_mlm, self.total_denominator = 0.0, 0.0
        self.total_correct_sp, self.total_instances = 0.0, 0.0


class AlbertTrainer(BertTrainer):
    pass


class LmTrainer(MlmTrainer):
    pass


class BilmTrainer(Trainer):
    def __init__(self, args):
        super(BilmTrainer, self).__init__(args)
        self.total_loss_forward, self.total_loss_backward = 0.0, 0.0
        self.total_correct_forward, self.total_correct_backward = 0.0, 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        src, tgt_forward, tgt_backward, seg = batch
        loss_info = model(src, (tgt_forward, tgt_backward), seg)
        loss_forward, loss_backward, correct_forward, correct_backward, denominator = loss_info
        loss = loss_forward + loss_backward
        self.total_loss += loss.item()
        self.total_loss_forward += loss_forward.item()
        self.total_loss_backward += loss_backward.item()
        self.total_correct_forward += correct_forward.item()
        self.total_correct_backward += correct_backward.item()
        self.total_denominator += denominator.item()
        loss = loss / self.accumulation_steps
        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size
        self.logger.info("| {:8d}/{:8d} steps"
              "| {:8.2f} tokens/s"
              "| loss {:7.2f}"
              "| loss_forward {:3.3f}"
              "| loss_backward {:3.3f}"
              "| acc_forward: {:3.3f}"
              "| acc_backward: {:3.3f}".format(
                  self.current_step,
                  self.total_steps,
                  done_tokens / (time.time() - self.start_time),
                  self.total_loss / self.report_steps,
                  self.total_loss_forward / self.report_steps,
                  self.total_loss_backward / self.report_steps,
                  self.total_correct_forward / self.total_denominator,
                  self.total_correct_backward / self.total_denominator))

        self.total_loss, self.total_loss_forward, self.total_loss_backward = 0.0, 0.0, 0.0
        self.total_correct_forward, self.total_correct_backward, self.total_denominator = 0.0, 0.0, 0.0


class ClsTrainer(Trainer):
    def __init__(self, args):
        super(ClsTrainer, self).__init__(args)
        self.total_correct = 0.0
        self.total_instances = 0.0

    def forward_propagation(self, batch, model):
        src, tgt, seg = batch
        loss_info = model(src, tgt, seg)
        loss, correct = loss_info
        self.total_loss += loss.item()
        self.total_correct += correct.item()
        self.total_instances += src.size(0)
        loss = loss / self.accumulation_steps
        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size
        self.logger.info("| {:8d}/{:8d} steps"
              "| {:8.2f} tokens/s"
              "| loss {:7.2f}"
              "| acc: {:3.3f}".format(
                  self.current_step,
                  self.total_steps,
                  done_tokens / (time.time() - self.start_time),
                  self.total_loss / self.report_steps,
                  self.total_correct / self.total_instances))

        self.total_loss = 0.0
        self.total_correct = 0.0
        self.total_instances = 0.0


class MtTrainer(Trainer):
    def __init__(self, args):
        super(MtTrainer, self).__init__(args)
        self.total_correct = 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        src, tgt_out, seg, tgt_in, tgt_seg = batch
        loss_info = model(src, tgt_out, seg, tgt_in, tgt_seg)
        loss, correct, denominator = loss_info
        self.total_loss += loss.item()
        self.total_correct += correct.item()
        self.total_denominator += denominator.item()

        loss = loss / self.accumulation_steps

        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size

        self.logger.info("| {:8d}/{:8d} steps"
              "| {:8.2f} tokens/s"
              "| loss {:7.2f}"
              "| acc: {:3.3f}".format(
                  self.current_step,
                  self.total_steps,
                  done_tokens / (time.time() - self.start_time),
                  self.total_loss / self.report_steps,
                  self.total_correct / self.total_denominator))

        self.total_loss = 0.0
        self.total_correct = 0.0
        self.total_denominator = 0.0


class ClsMlmTrainer(Trainer):
    def __init__(self, args):
        super(ClsMlmTrainer, self).__init__(args)
        self.total_loss_cls = 0.0
        self.total_correct_cls = 0.0
        self.total_instances = 0.0

        self.total_loss_mlm = 0.0
        self.total_correct_mlm = 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        src, tgt_mlm, tgt_cls, seg = batch
        tgt = {"mlm": tgt_mlm, "cls": tgt_cls}
        loss_info = model(src, tgt, seg)
        loss_mlm, correct_mlm, denominator = loss_info["mlm"]
        loss_cls, correct_cls = loss_info["cls"]
        loss = loss_mlm + loss_cls
        self.total_loss += loss.item()
        self.total_loss_mlm += loss_mlm.item()
        self.total_loss_cls += loss_cls.item()
        self.total_correct_mlm += correct_mlm.item()
        self.total_correct_cls += correct_cls.item()
        self.total_denominator += denominator.item()
        self.total_instances += src.size(0)
        loss = loss / self.accumulation_steps

        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size

        self.logger.info("| {:8d}/{:8d} steps"
              "| {:8.2f} tokens/s"
              "| loss {:7.2f}"
              "| loss_mlm: {:3.3f}"
              "| loss_cls: {:3.3f}"
              "| acc_mlm: {:3.3f}"
              "| acc_cls: {:3.3f}".format(
                  self.current_step,
                  self.total_steps,
                  done_tokens / (time.time() - self.start_time),
                  self.total_loss / self.report_steps,
                  self.total_loss_mlm / self.report_steps,
                  self.total_loss_cls / self.report_steps,
                  self.total_correct_mlm / self.total_denominator,
                  self.total_correct_cls / self.total_instances))

        self.total_loss, self.total_loss_mlm, self.total_loss_cls = 0.0, 0.0, 0.0
        self.total_correct_mlm, self.total_denominator = 0.0, 0.0
        self.total_correct_cls, self.total_instances = 0.0, 0.0


class T5Trainer(MtTrainer):
    pass


class GsgTrainer(MtTrainer):
    pass


class BartTrainer(MtTrainer):
    pass


class PrefixlmTrainer(MlmTrainer):
    pass


str2trainer = {"bert": BertTrainer, "mlm": MlmTrainer, "lm": LmTrainer,
               "albert": AlbertTrainer, "bilm": BilmTrainer, "cls": ClsTrainer,
               "mt": MtTrainer, "t5": T5Trainer, "gsg": GsgTrainer,
               "bart": BartTrainer, "prefixlm": PrefixlmTrainer, "cls_mlm": ClsMlmTrainer}


def worker(local_rank, gpu_ranks, args):
    """
    Args:
        local_rank: The id of GPU for single GPU mode;
                    The id of process (and GPU) for multiprocessing distributed mode.
        gpu_ranks: List of ranks of each process.
    """
    set_seed(args.seed)

    # Get logger
    args.logger = init_logger(args)

    # Env initialize.
    args.local_rank = local_rank
    init_env(args)
    global_rank = args.global_rank

    # Build model.
    model = init_model(args)

    # Build optimizer.
    custom_optimizer, custom_scheduler = init_optimizer(args, model)

    if local_rank is not None:
        model.cuda(local_rank)
    optimizer = custom_optimizer
    scheduler = custom_scheduler

    if args.dist_train:
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        args.logger.info("Worker %d is training ... " % global_rank)
    else:
        args.logger.info("Worker is training ...")

    if args.dist_train:
        train_loader = str2dataloader[args.data_processor](args, args.dataset_path, args.batch_size, global_rank, args.world_size, local_rank, True)
    else:
        train_loader = str2dataloader[args.data_processor](args, args.dataset_path, args.batch_size, 0, 1, local_rank, True)


    trainer = str2trainer[args.data_processor](args)
    trainer.train(args, local_rank, global_rank, train_loader, model, optimizer, scheduler)
