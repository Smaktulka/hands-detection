import torch
import torch.nn.functional as F
from torch import nn


class BlazeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super(BlazeBlock, self).__init__()

        self.stride = stride
        self.channel_pad = out_channels - in_channels
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._define_layers()

    def _define_layers(self):
        if self.stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=self.stride, stride=self.stride)
            padding = 0
        else:
            padding = (self.kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                      kernel_size=self.kernel_size, stride=self.stride, padding=padding,
                      groups=self.in_channels, bias=True),
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.out_channels),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.act(self.convs(h) + x)
