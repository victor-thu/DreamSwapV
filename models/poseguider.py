from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .motion_module import zero_module

class PoseGuider(torch.nn.Module):
    def __init__(self):
        super(PoseGuider, self).__init__()
        self.pose_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=16, eps=1e-6, affine=True),
            torch.nn.SiLU(),

            torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=16, eps=1e-6, affine=True),
            torch.nn.SiLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_features=32, eps=1e-6, affine=True),
            torch.nn.SiLU(),

            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=32, eps=1e-6, affine=True),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_features=64, eps=1e-6, affine=True),
            torch.nn.SiLU(),

            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=64, eps=1e-6, affine=True),
            torch.nn.SiLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_features=128, eps=1e-6, affine=True),
            torch.nn.SiLU(),
        )

        self.projector = torch.nn.Conv2d(128, 320, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.pose_encoder(x)
        x = self.projector(x)
        return x

    def _init_weight(self):
        for m in self.pose_encoder.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.projector.weight.data.fill_(0)
        self.projector.bias.data.fill_(0)

class PoseGuiderWoBN(torch.nn.Module):
    def __init__(
        self,
        conditioning_embedding_channels: int = 320,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 96, 256),
    ):
        super().__init__()
        self.conv_in = torch.nn.Conv2d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                torch.nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.blocks.append(
                torch.nn.Conv2d(
                    channel_in, channel_out, kernel_size=3, padding=1, stride=2
                )
            )

        self.conv_out = zero_module(
            torch.nn.Conv2d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.conv_out.weight.data.fill_(0)
        self.conv_out.bias.data.fill_(0)
