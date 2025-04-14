import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import blaze
from blaze import DEFAULT_CLASSES_NUM
from blaze_block import BlazeBlock


class BlazeModel(nn.Module):
    def __init__(self) -> None:
        super(BlazeModel, self).__init__()

        self.dbox_list = None
        self.num_anchors = 896
        self.x_scale = 128.0
        self.y_scale = 128.0
        self.h_scale = 128.0
        self.w_scale = 128.0
        self.min_score_thresh = 0.75
        self._define_layers()

    def _define_layers(self) -> None:
        self.backbone1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2, padding=0, bias=True),
            nn.ReLU(inplace=True),

            BlazeBlock(24, 24),
            BlazeBlock(24, 28),
            BlazeBlock(28, 32, stride=2),
            BlazeBlock(32, 36),
            BlazeBlock(36, 42),
            BlazeBlock(42, 48, stride=2),
            BlazeBlock(48, 56),
            BlazeBlock(56, 64),
            BlazeBlock(64, 72),
            BlazeBlock(72, 80),
            BlazeBlock(80, 88),
        )

        self.backbone2 = nn.Sequential(
            BlazeBlock(88, 96, stride=2),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
        )

        self.classifier_8 = nn.Conv2d(88, 2 * DEFAULT_CLASSES_NUM, 1, bias=True)
        self.classifier_16 = nn.Conv2d(96, 6 * DEFAULT_CLASSES_NUM, 1, bias=True)

        self.regressor_8 = nn.Conv2d(88, 8, 1, bias=True)
        self.regressor_16 = nn.Conv2d(96, 24, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (1, 2, 1, 2), "constant", 0)
        b = x.shape[0]

        x = self.backbone1(x)
        h = self.backbone2(x)

        c1 = self.classifier_8(x)
        c1 = c1.permute(0, 2, 3, 1)
        c1 = c1.reshape(b, -1, DEFAULT_CLASSES_NUM)
        c2 = self.classifier_16(h)
        c2 = c2.permute(0, 2, 3, 1)
        c2 = c2.reshape(b, -1, DEFAULT_CLASSES_NUM)
        c = torch.cat((c1, c2), dim=1)

        r1 = self.regressor_8(x)
        r1 = r1.permute(0, 2, 3, 1)
        r1 = r1.reshape(b, -1, 4)
        r2 = self.regressor_16(h)
        r2 = r2.permute(0, 2, 3, 1)
        r2 = r2.reshape(b, -1, 4)

        r = torch.cat((r1, r2), dim=1)
        return torch.cat([r, c], dim=2)

    def load_weights(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_anchors(self, path: str) -> None:
        self.dbox_list = torch.tensor(np.load(path), dtype=torch.float32, device=blaze.DEVICE)
        assert (self.dbox_list.ndimension() == 2)
        assert (self.dbox_list.shape[0] == self.num_anchors)
        assert (self.dbox_list.shape[1] == 4)
