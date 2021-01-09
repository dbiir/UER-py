from uer.targets.mlm_target import MlmTarget
from uer.targets.lm_target import LmTarget
from uer.targets.bert_target import BertTarget
from uer.targets.cls_target import ClsTarget
from uer.targets.bilm_target import BilmTarget
from uer.targets.albert_target import AlbertTarget
from uer.targets.mt_target import MtTarget
from uer.targets.t5_target import T5Target


str2target = {"bert": BertTarget, "mlm": MlmTarget, "lm": LmTarget,
              "bilm": BilmTarget, "albert": AlbertTarget, "mt": MtTarget,
              "t5": T5Target, "cls": ClsTarget}

__all__ = ["BertTarget", "MlmTarget", "LmTarget", "BilmTarget", "AlbertTarget",
           "MtTarget", "T5Target", "ClsTarget", "str2target"]

