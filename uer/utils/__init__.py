from uer.utils.tokenizers import CharTokenizer
from uer.utils.tokenizers import SpaceTokenizer
from uer.utils.tokenizers import BertTokenizer
from uer.utils.data import *
from uer.utils.act_fun import *


str2tokenizer = {"char": CharTokenizer, "space": SpaceTokenizer, "bert": BertTokenizer}
str2dataset = {"bert": BertDataset, "lm": LmDataset, "mlm": MlmDataset,
               "bilm": BilmDataset, "albert": AlbertDataset, "mt": MtDataset,
               "t5": T5Dataset, "cls": ClsDataset}
str2dataloader = {"bert": BertDataLoader, "lm": LmDataLoader, "mlm": MlmDataLoader,
                  "bilm": BilmDataLoader, "albert": AlbertDataLoader, "mt": MtDataLoader,
                  "t5": T5DataLoader, "cls": ClsDataLoader}

str2act = {"gelu": gelu, "gelu_fast": gelu_fast, "relu": relu, "silu": silu, "linear": linear}

__all__ = ["CharTokenizer", "SpaceTokenizer", "BertTokenizer", "str2tokenizer",
           "BertDataset", "LmDataset", "MlmDataset", "BilmDataset",
           "AlbertDataset", "MtDataset", "T5Dataset", "ClsDataset", "str2dataset",
           "BertDataLoader", "LmDataLoader", "MlmDataLoader", "BilmDataLoader",
           "AlbertDataLoader", "MtDataLoader", "T5DataLoader", "ClsDataLoader", "str2dataloader",
           "gelu", "gelu_fast", "relu", "silu", "str2act"]
