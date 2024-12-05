import torch
from torch import Tensor

from model import BlazePalm


def get_class_num(raw_t: Tensor):
    raw_t = blaze_palm.forward(img)
    raw_t = raw_t.clamp(-100, 100)
    scores = raw_t.sigmoid().squeeze(dim=-1)
    scores = torch.where(scores >= 0.7, scores, 0)
    scores = scores.permute(0, 2, 1)
    max_, ind = torch.max(scores, dim=2)

    return max_, torch.arg




img = torch.randn(1, 3, 256, 256)
blaze_palm = BlazePalm(classes_num=4)
raw_tensor = blaze_palm.forward(img)
clazz_num = get_class_num(raw_tensor)

print("result")
