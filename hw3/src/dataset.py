import os
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
import torchvision


class MnistDataset(Dataset):

    def __init__(
        self,
        root: str = os.path.join('data', 'mnist'),
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.image_files = list(Path(root).glob("*.png"))
        self.transform = transform

    def __getitem__(self, index: int) -> torch.Tensor:
        image_file = self.image_files[index]

        image = torchvision.io.read_image(
            str(image_file), torchvision.io.image.ImageReadMode.RGB)

        if self.transform is not None:
            image = self.transform(image)

        # map [0, 255] to [-1, 1]
        return image.float() / 127.5 - 1.0

    def __len__(self) -> int:
        return len(self.image_files)


class GaussianNoiseDataset(Dataset):

    def __init__(self, shape, length, mean=0, std=1) -> None:
        super().__init__()
        self.shape = shape
        self.length = length
        self.mean = mean
        self.std = std

    def __getitem__(self, index) -> torch.Tensor:
        return self.mean + self.std * torch.randn(self.shape)

    def __len__(self) -> int:
        return self.length


if __name__ == '__main__':
    dataset = MnistDataset()
    print(dataset[0].shape)  # torch.Size([3, 28, 28])
    print(dataset[0].dtype)  # torch.float32
    print(dataset[0].max(), dataset[0].min())  #tensor(1.) tensor(-1.)
