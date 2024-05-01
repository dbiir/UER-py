import json
import sys
from argparse import Namespace


def load_hyperparam(args):
    """
    Load arguments form argparse and config file
    Priority: default options < config file < command line args
    """
    with open(args.config_path, mode="r", encoding="utf-8") as f:
        config_args_dict = json.load(f)

    args_dict = vars(args)

    command_line_args_dict = {k: args_dict[k] for k in [
        a[2:] for a in sys.argv if (a[:2] == "--" and "local_rank" not in a)
    ]}
    args_dict.update(config_args_dict)
    args_dict.update(command_line_args_dict)
    args = Namespace(**args_dict)

    return args
