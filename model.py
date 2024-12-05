import torch
import torch.nn as nn
import torch.nn.functional as F

from blaze_block import BlazeBlock


class BlazePalm(nn.Module):
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def __init__(self, classes_num, batch_size):
        super(BlazePalm, self).__init__()
        self.classes_num = classes_num
        self.coordinations_num = 4
        self.min_score_thresh = 0.7
        self.batch_size = batch_size

        self._define_layers()

    def _define_layers(self):
        self.backbone1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.ReLU(inplace=True),

            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 32),

            BlazeBlock(32, 64, stride=2),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),

            BlazeBlock(64, 128, stride=2),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),

        )

        self.backbone2 = nn.Sequential(
            BlazeBlock(128, 256, stride=2),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
        )

        self.backbone3 = nn.Sequential(
            BlazeBlock(256, 256, stride=2),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
        )

        self.conv_transpose_256 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2,
                                                     padding=0,
                                                     bias=True)
        self.blaze_256 = BlazeBlock(256, 256)

        self.conv_transpose_128 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2,
                                                     padding=0,
                                                     bias=True)
        self.blaze_128 = BlazeBlock(128, 128)

        self.classifier_32 = nn.Conv2d(128, 2 * self.classes_num, 1, bias=True)
        self.classifier_16 = nn.Conv2d(256, 2 * self.classes_num, 1, bias=True)
        self.classifier_8 = nn.Conv2d(256, 6 * self.classes_num, 1, bias=True)

        self.regressor_32 = nn.Conv2d(128, 2 * self.coordinations_num, 1, bias=True)
        self.regressor_16 = nn.Conv2d(256, 2 * self.coordinations_num, 1, bias=True)
        self.regressor_8 = nn.Conv2d(256, 6 * self.coordinations_num, 1, bias=True)

        # self.fc = nn.Linear(2944 * self.classes_num, self.classes_num)

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        x_32 = F.pad(x, (0, 1, 0, 1), "constant", 0)

        x_32 = self.backbone1(x_32)  # (batch_size, 128, 32, 32)
        y_16 = self.backbone2(x_32)  # (batch_size, 256, 16, 16)
        z_8 = self.backbone3(y_16)  # (batch_size, 256, 8, 8)

        y_16 = y_16 + F.relu(self.conv_transpose_256(z_8), True)  # (batch_size, 256, 16, 16)
        y_16 = self.blaze_256(y_16)  # (batch_size, 256, 16, 16)

        x_32 = x_32 + F.relu(self.conv_transpose_128(y_16))  # (batch_size, 128, 32, 32)
        x_32 = self.blaze_128(x_32)  # (batch_size, 128, 32, 32)

        c_8 = self.classifier_8(z_8)  # (batch_size, 6 * classes_num, 8, 8)
        c_16 = self.classifier_16(y_16)  # (batch_size, 2 * classes_num, 16, 16)
        c_32 = self.classifier_32(x_32)  # (batch_size, 2 * classes_num, 32, 32)

        c_8 = c_8.permute(0, 2, 3, 1)
        c_16 = c_16.permute(0, 2, 3, 1)
        c_32 = c_32.permute(0, 2, 3, 1)

        c_8 = c_8.reshape(batch_size, -1, self.classes_num)
        c_16 = c_16.reshape(batch_size, -1, self.classes_num)
        c_32 = c_32.reshape(batch_size, -1, self.classes_num)

        c = torch.cat((c_32, c_16, c_8), dim=1)

        r_8 = self.regressor_8(z_8)
        r_16 = self.regressor_16(y_16)
        r_32 = self.regressor_32(x_32)

        r_8 = r_8.permute(0, 2, 3, 1)
        r_16 = r_16.permute(0, 2, 3, 1)
        r_32 = r_32.permute(0, 2, 3, 1)

        r_8 = r_8.reshape(batch_size, -1, 4)
        r_16 = r_16.reshape(batch_size, -1, 4)
        r_32 = r_32.reshape(batch_size, -1, 4)

        r = torch.cat((r_32, r_16, r_8), dim=1)

        return torch.cat([r, c], dim=2)
        # print("forward c")
        # print(c)
        # scores = c.clamp(-100, 100)
        # print("scores")
        # print(scores)
        # scores = torch.nn.functional.softmax(scores, dim=-1).squeeze()
        # scores = scores.sigmoid().squeeze(dim=-1)
        # print("softmax")
        # print(scores)
        # scores = torch.where(scores >= 0.7, scores, 0)
        # scores = scores.permute(0, 2, 1)
        # print("after where")
        # print(scores)
        # max_, ind = torch.max(scores, dim=2)
        #
        # return max_

