from uer.utils.tokenizers import CharTokenizer
from uer.utils.tokenizers import SpaceTokenizer
from uer.utils.tokenizers import BertTokenizer
from uer.utils.data import *
from uer.utils.act_fun import *
from uer.utils.optimizers import *


str2tokenizer = {"char": CharTokenizer, "space": SpaceTokenizer, "bert": BertTokenizer}
str2dataset = {"bert": BertDataset, "lm": LmDataset, "mlm": MlmDataset,
               "bilm": BilmDataset, "albert": AlbertDataset, "seq2seq": Seq2seqDataset,
               "t5": T5Dataset, "cls": ClsDataset, "prefixlm": PrefixlmDataset}
str2dataloader = {"bert": BertDataLoader, "lm": LmDataLoader, "mlm": MlmDataLoader,
                  "bilm": BilmDataLoader, "albert": AlbertDataLoader, "seq2seq": Seq2seqDataLoader,
                  "t5": T5DataLoader, "cls": ClsDataLoader, "prefixlm": PrefixlmDataLoader}

str2act = {"gelu": gelu, "gelu_fast": gelu_fast, "relu": relu, "silu": silu, "linear": linear}

str2optimizer = {"adamw": AdamW, "lamb": Lamb, "adabelief": AdaBelief}

str2schedule = {"warmupconstant": WarmupConstantSchedule, "warmuplinear": WarmupLinearSchedule,
                "warmupcosine": WarmupCosineSchedule, "warmupcosinerestart": WarmupCosineWithHardRestartsSchedule}

__all__ = ["CharTokenizer", "SpaceTokenizer", "BertTokenizer", "str2tokenizer",
           "BertDataset", "LmDataset", "MlmDataset", "BilmDataset",
           "AlbertDataset", "Seq2seqDataset", "T5Dataset", "ClsDataset",
           "PrefixlmDataset", "str2dataset",
           "BertDataLoader", "LmDataLoader", "MlmDataLoader", "BilmDataLoader",
           "AlbertDataLoader", "Seq2seqDataLoader", "T5DataLoader", "ClsDataLoader",
           "PrefixlmDataLoader", "str2dataloader",
           "gelu", "gelu_fast", "relu", "silu", "linear", "str2act",
           "AdamW", "Lamb", "AdaBelief", "str2optimizer",
           "WarmupConstantSchedule", "WarmupLinearSchedule", "WarmupCosineSchedule",
           "WarmupCosineWithHardRestartsSchedule", "str2schedule"]
