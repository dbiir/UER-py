import torch
import torch.nn as nn
from uer.targets import *


class ViltTarget(BertTarget):
    """
    BERT exploits masked language modeling (MLM)
    and next sentence prediction (NSP) for pretraining.
    """
    pass