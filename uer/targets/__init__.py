from uer.targets.bert_target import BertTarget
from uer.targets.mlm_target import MlmTarget
from uer.targets.lm_target import LmTarget
from uer.targets.bilm_target import BilmTarget
from uer.targets.albert_target import AlbertTarget
from uer.targets.mt_target import MtTarget


str2target = {"bert": BertTarget, "mlm": MlmTarget, "lm": LmTarget,
               "bilm": BilmTarget, "albert": AlbertTarget, "mt": MtTarget}

__all__ = ["BertTarget", "MlmTarget", "LmTarget", "BilmTarget", "AlbertTarget",
           "MtTarget", "str2target"]

