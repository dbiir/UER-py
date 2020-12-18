from uer.targets.bert_target import BertTarget
from uer.targets.mlm_target import MlmTarget
from uer.targets.lm_target import LmTarget
from uer.targets.bilm_target import BilmTarget
from uer.targets.albert_target import AlbertTarget


str2target = {"bert": BertTarget, "mlm": MlmTarget, "lm": LmTarget,
               "bilm": BilmTarget, "albert": AlbertTarget}

__all__ = ["BertTarget", "MlmTarget", "LmTarget", "BilmTarget", "AlbertTarget",
           "str2target"]

