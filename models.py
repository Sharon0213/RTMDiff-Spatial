"""Conditional U-Net for RTMDiff-Spatial."""

import torch
from torch import nn


class TwoWaysModule(object):
    pass


class TwoWaysSequential(nn.Module):

    def __init__(self, *modules):
        super().__init__()
        self.module_list = nn.ModuleList(modules)

    def forward(self, inputs: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        h = inputs
        for m in self.module_list:
            if isinstance(m, TwoWaysModule):
                h = m(h, conditions)
            else:
                h = m(h)
        return h


class ResBlock_time(nn.Module, TwoWaysModule):

    def __init__(self, num_channels: int, condition_dim: int):
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(num_channels),
            nn.GELU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.GELU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, num_channels),
        )

    def forward(self, inputs: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:

        h = inputs + self.residual(inputs)
        t = conditions[:, 0]
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, h.shape[-2], h.shape[-1])
        h = h + emb
        return h


class ConditionalUNet(nn.Module):

    def __init__(
            self,
            num_channels: int,
            condition_dim: int = 512,
            base_channels: int = 64,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.base_channels = base_channels
        self.condition_dim = condition_dim

        self.conv_in = nn.Conv2d(num_channels+condition_dim, base_channels, kernel_size=3, stride=1, padding=1)
        self.encoder_list = nn.ModuleList([
            # stage 1
            TwoWaysSequential(
                ResBlock_time(1 * base_channels, condition_dim),
                ResBlock_time(1 * base_channels, condition_dim),
            ),
            # stage 2
            TwoWaysSequential(
                nn.Conv2d(base_channels, 2 * base_channels, kernel_size=2, stride=2, padding=0),
                ResBlock_time(2 * base_channels, condition_dim),
                ResBlock_time(2 * base_channels, condition_dim),
            ),
            # stage 3
            TwoWaysSequential(
                nn.Conv2d(2 * base_channels, 4 * base_channels, kernel_size=2, stride=2, padding=0),
                ResBlock_time(4 * base_channels, condition_dim),
                ResBlock_time(4 * base_channels, condition_dim),
            ),
            # stage 4
            TwoWaysSequential(
                nn.Conv2d(4 * base_channels, 8 * base_channels, kernel_size=2, stride=2, padding=0),
                ResBlock_time(8 * base_channels, condition_dim),
                ResBlock_time(8 * base_channels, condition_dim),
            ),
        ])
        self.middle = TwoWaysSequential(
            nn.Conv2d(8 * base_channels, 32 * base_channels, kernel_size=2, stride=2, padding=0),
            ResBlock_time(32 * base_channels, condition_dim),
            ResBlock_time(32 * base_channels, condition_dim),
            nn.ConvTranspose2d(32 * base_channels, 8 * base_channels, kernel_size=2, stride=2, padding=0),
        )
        self.decoder_list = nn.ModuleList([
            # stage 4
            TwoWaysSequential(
                nn.Conv2d(2 * 8 * base_channels, 8 * base_channels, kernel_size=1, stride=1, padding=0),
                ResBlock_time(8 * base_channels, condition_dim),
                ResBlock_time(8 * base_channels, condition_dim),
                nn.ConvTranspose2d(8 * base_channels, 4 * base_channels, kernel_size=2, stride=2, padding=0),
            ),
            # stage 3
            TwoWaysSequential(
                nn.Conv2d(2 * 4 * base_channels, 4 * base_channels, kernel_size=1, stride=1, padding=0),
                ResBlock_time(4 * base_channels, condition_dim),
                ResBlock_time(4 * base_channels, condition_dim),
                nn.ConvTranspose2d(4 * base_channels, 2 * base_channels, kernel_size=2, stride=2, padding=0),
            ),
            # stage 2
            TwoWaysSequential(
                nn.Conv2d(2 * 2 * base_channels, 2 * base_channels, kernel_size=1, stride=1, padding=0),
                ResBlock_time(2 * base_channels, condition_dim),
                ResBlock_time(2 * base_channels, condition_dim),
                nn.ConvTranspose2d(2 * base_channels, 1 * base_channels, kernel_size=2, stride=2, padding=0),
            ),
            # stage 1
            TwoWaysSequential(
                nn.Conv2d(2 * 1 * base_channels, 1 * base_channels, kernel_size=1, stride=1, padding=0),
                ResBlock_time(1 * base_channels, condition_dim),
                ResBlock_time(1 * base_channels, condition_dim),
            ),
        ])
        self.conv_out = nn.Conv2d(base_channels, num_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        c = conditions[:, 1:, :]
        c = torch.swapdims(c, 1, 2)
        c = c.view(c.shape[0], c.shape[1], inputs.shape[2], inputs.shape[3])
        inputs = torch.cat((inputs, c), axis=1)
        h = self.conv_in(inputs)
        skip_list = []
        for m in self.encoder_list:
            h = m(h, conditions)
            skip_list.insert(0, h)
        h = self.middle(h, conditions)
        for m, skip in zip(self.decoder_list, skip_list):
            h = torch.concat([skip, h], dim=1)
            h = m(h, conditions)
        h = self.conv_out(h)
        return h
