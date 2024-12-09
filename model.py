import torch
import torch.nn as nn
from torchinfo import summary

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(),
        )

        self.skip = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        z = self.model(x)
        s = self.skip(x)
        return z + s

class BBoxModel(nn.Module):
    def __init__(self):
        super().__init__()

        # self.model = MobileNetV4("MobileNetV4ConvSmall")

        self.model = nn.Sequential(
            ResBlock(4, 32),
            nn.MaxPool2d(2),  # 128 -> 64
            nn.GroupNorm(2, 32),
            ResBlock(32, 64),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.GroupNorm(4, 64),
            ResBlock(64, 128),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.GroupNorm(8, 128),
            ResBlock(128, 256),
            nn.MaxPool2d(2),  # 16 -> 8
            nn.GroupNorm(16, 256),
            ResBlock(256, 512),
            nn.MaxPool2d(2),  # 8 -> 4
            nn.GroupNorm(32, 512),
            ResBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.Flatten(),
            nn.Linear(4 * 4 * 512, 64),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(),
        )

        self.bbox_output = nn.Sequential(
            nn.Linear(64, 4),
            nn.Sigmoid()
        )
        self.label_output = nn.Sequential(
            nn.Linear(64, 2)
        )

    def forward(self, x, large_mask, medium_mask, small_mask):

        x = torch.cat([x, large_mask, medium_mask, small_mask], dim=1)

        z = self.model(x)

        bbox_out = self.bbox_output(z)
        label_out = self.label_output(z)
        return bbox_out, label_out


if __name__ == '__main__':
    model = BBoxModel()
    batch_size = 576
    num_preds = 1
    img_res = 128
    summary(model, input_size=((num_preds * batch_size, 1, img_res, img_res), (num_preds * batch_size, 1, img_res, img_res), (num_preds * batch_size, 1, img_res, img_res), (num_preds * batch_size, 1, img_res, img_res)), device='meta')
