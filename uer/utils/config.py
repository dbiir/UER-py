import json
import sys
from argparse import Namespace


def load_hyperparam(args):
    """
    Load arguments form argparse and config file
    Priority: default options < config file < command line args
    """
    with open(args.config_path, mode="r", encoding="utf-8") as f:
        param = json.load(f)

    if "deepspeed" in args.__dict__:
        with open(args.deepspeed_config, mode="r", encoding="utf-8") as f:
            args.deepspeed_config_param = json.load(f)

    args_dict = vars(args)

    input_args = {k: args_dict[k] for k in [a[2:] for a in sys.argv if (a[:2] == "--" and "local_rank" not in a)]}
    args_dict.update(param)
    args_dict.update(input_args)
    args = Namespace(**args_dict)

    return args
