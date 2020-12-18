from uer.utils.tokenizers import CharTokenizer
from uer.utils.tokenizers import SpaceTokenizer
from uer.utils.tokenizers import BertTokenizer
from uer.utils.data import *


str2tokenizer = {"char": CharTokenizer, "space": SpaceTokenizer, "bert": BertTokenizer}
str2dataset = {"bert": BertDataset, "lm": LmDataset, "mlm": MlmDataset,
               "bilm": BilmDataset, "albert": AlbertDataset}
str2dataloader = {"bert": BertDataLoader, "lm": LmDataLoader, "mlm": MlmDataLoader,
                  "bilm": BilmDataLoader, "albert": AlbertDataLoader}

__all__ = ["CharTokenizer", "SpaceTokenizer", "BertTokenizer", "str2tokenizer",
           "BertDataset", "LmDataset", "MlmDataset", "BilmDataset",
           "AlbertDataset", "str2dataset",
           "BertDataLoader", "LmDataLoader", "MlmDataLoader", "BilmDataLoader",
           "AlbertDataLoader", "str2dataloader"]

