import torch
from torch import nn


class Loss(nn.Module):

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):

