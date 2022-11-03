from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.datasets import VisionDataset


class UnlabeledDataset(VisionDataset):

    def __init__(self, root: str, transform: Optional[Callable] = None) -> None:
        super().__init__(root, None, transform, None)

        self.image_files = list(Path(root).glob("*.jpg"))

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_file = self.image_files[index]

        image = torchvision.io.read_image(
            str(image_file), torchvision.io.image.ImageReadMode.GRAY)

        if self.transform is not None:
            trans_image = self.transform(image)
        else:
            trans_image = image

        label = torch.tensor(int(image_file.stem), dtype=torch.int16)

        return image.float() / 255., trans_image.float() / 255., label

    def __len__(self) -> int:
        return len(self.image_files)


class EmbeddingDataset(Dataset):

    def __init__(self, root: str) -> None:
        super().__init__()

        self.image_files = list(Path(root).glob("*.jpg"))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_file = self.image_files[index]

        image = torchvision.io.read_image(
            str(image_file), torchvision.io.image.ImageReadMode.GRAY)

        label = torch.tensor(int(image_file.stem), dtype=torch.int16)

        return image.float() / 255., label

    def __len__(self) -> int:
        return len(self.image_files)
