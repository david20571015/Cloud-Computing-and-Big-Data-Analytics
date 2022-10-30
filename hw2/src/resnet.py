from typing import Type

import torch
from torch import nn


class Stem(nn.Module):

    def __init__(self, in_channels=3, out_channels=64) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(7, 7),
                stride=(1, 1),
                padding='same',
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
        )

    def forward(self, x: torch.Tensor):
        return self.blocks(x)


class BasicBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        inner_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            nn.Conv2d(
                in_channels,
                inner_channels,
                kernel_size=(3, 3),
                padding='same',
            ),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inner_channels,
                out_channels,
                kernel_size=(3, 3),
                padding='same',
            ),
            nn.BatchNorm2d(out_channels),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        return self.blocks(x) + self.shortcut(x)


class BottleNeckBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        inner_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            nn.Conv2d(
                in_channels,
                inner_channels,
                kernel_size=(1, 1),
            ),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inner_channels,
                inner_channels,
                kernel_size=(3, 3),
                padding='same',
            ),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inner_channels,
                out_channels,
                kernel_size=(1, 1),
            ),
            nn.BatchNorm2d(out_channels),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        return self.blocks(x) + self.shortcut(x)


class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        inner_channels: int,
        num_blocks: int,
        block_type: Type[BasicBlock | BottleNeckBlock],
    ) -> None:
        super().__init__()

        if block_type == BasicBlock:
            out_channels = inner_channels
        elif block_type == BottleNeckBlock:
            out_channels = inner_channels * 4
        else:
            raise ValueError("block_type must be BasicBlock or BottleNeckBlock")

        self.blocks = nn.Sequential(
            block_type(
                in_channels,
                inner_channels,
                out_channels,
            ),
            *[
                block_type(
                    out_channels,
                    inner_channels,
                    out_channels,
                ) for _ in range(num_blocks - 1)
            ],
        )

        self.downsample = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(2, 2),
            stride=(2, 2),
        )

    def forward(self, x: torch.Tensor):
        x = self.blocks(x)
        x = self.downsample(x)
        return x


class Head(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x: torch.Tensor):
        return self.blocks(x)


class ResNet(nn.Module):

    def __init__(
        self,
        block_type: Type[BasicBlock | BottleNeckBlock],
        in_channels: list[int],
        depths: list[int],
        num_classes: int,
    ) -> None:
        super().__init__()

        self.stem = Stem(in_channels=3, out_channels=64)

        out_channels = in_channels if block_type == BasicBlock else [
            ch * 4 for ch in in_channels
        ]

        self.blocks = nn.Sequential(
            ResidualBlock(
                in_channels=64,
                inner_channels=in_channels[0],
                num_blocks=depths[0],
                block_type=block_type,
            ),
            ResidualBlock(
                in_channels=out_channels[0],
                inner_channels=in_channels[1],
                num_blocks=depths[1],
                block_type=block_type,
            ),
            ResidualBlock(
                in_channels=out_channels[1],
                inner_channels=in_channels[2],
                num_blocks=depths[2],
                block_type=block_type,
            ),
            ResidualBlock(
                in_channels=out_channels[2],
                inner_channels=in_channels[3],
                num_blocks=depths[3],
                block_type=block_type,
            ),
        )

        self.head = Head(
            in_channels=out_channels[3],
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.blocks(x)
        # x = self.head(x)
        return x


class ResNet18(ResNet):

    def __init__(self, num_classes: int) -> None:
        super().__init__(
            block_type=BasicBlock,
            in_channels=[64, 128, 256, 512],
            depths=[2, 2, 2, 2],
            num_classes=num_classes,
        )


class ResNet34(ResNet):

    def __init__(self, num_classes: int) -> None:
        super().__init__(
            block_type=BasicBlock,
            in_channels=[64, 128, 256, 512],
            depths=[3, 4, 6, 3],
            num_classes=num_classes,
        )


class ResNet50(ResNet):

    def __init__(self, num_classes: int) -> None:
        super().__init__(
            block_type=BottleNeckBlock,
            in_channels=[64, 128, 256, 512],
            depths=[3, 4, 6, 3],
            num_classes=num_classes,
        )


class ResNet101(ResNet):

    def __init__(self, num_classes: int) -> None:
        super().__init__(
            block_type=BottleNeckBlock,
            in_channels=[64, 128, 256, 512],
            depths=[3, 4, 23, 3],
            num_classes=num_classes,
        )


class ResNet152(ResNet):

    def __init__(self, num_classes: int) -> None:
        super().__init__(
            block_type=BottleNeckBlock,
            in_channels=[64, 128, 256, 512],
            depths=[3, 8, 36, 3],
            num_classes=num_classes,
        )
