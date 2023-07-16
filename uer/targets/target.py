import torch.nn as nn


class Target(nn.Module):
    def __init__(self):
        super(Target, self).__init__()
        self.target_name_list = []
        self.loss_info = {}

    def update(self, target, target_name):
        setattr(self, target_name, target)
        self.target_name_list.append(target_name)

    def forward(self, memory_bank, tgt, seg):
        self.loss_info = {}
        for i, target_name in enumerate(self.target_name_list):
            target = getattr(self, target_name)
            if len(self.target_name_list) > 1:
                self.loss_info[self.target_name_list[i]] = target(memory_bank, tgt[self.target_name_list[i]], seg)
            else:
                self.loss_info = target(memory_bank, tgt, seg)

        return self.loss_info
