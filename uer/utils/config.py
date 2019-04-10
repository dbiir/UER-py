# -*- encoding:utf-8 -*-
import json
import codecs

def load_hyperparam(args):
    with codecs.open(args.config_path, "r", "utf-8") as f:
        param = json.load(f)
    args.emb_size = param["emb_size"]
    args.hidden_size = param["hidden_size"]
    args.feedforward_size = param["feedforward_size"]
    args.heads_num = param["heads_num"]
    args.layers_num = param["layers_num"]
    args.dropout = param["dropout"]
    return args

def save_hyperparam(args):
    parameters = {
        "emb_size": args.emb_size,
        "feedforward_size": args.feedforward_size,
        "hidden_size": args.hidden_size,
        "heads_num": args.heads_num,
        "layers_num": args.layers_num,
        "dropout": args.dropout,
        "adaptive": None
        }
    with codecs.open(args.config_path, "w", "utf-8") as f:
        json.dump(parameters, f)

