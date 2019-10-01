# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn


def count_lines(file_path):
    lines_num = 0
    with open(file_path, mode="r", encoding="utf-8") as f:
        for line in f:
            lines_num += 1
    return lines_num


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]
