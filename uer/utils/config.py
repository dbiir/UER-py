import json
from argparse import Namespace


def load_hyperparam(args):
    with open(args.config_path, mode="r", encoding="utf-8") as f:
        param = json.load(f)

    args_dict = vars(args)
    args_dict.update(param)
    args = Namespace(**args_dict)

    return args
