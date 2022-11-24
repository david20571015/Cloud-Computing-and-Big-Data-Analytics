import pathlib
import shutil
from typing import Any

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import utils
from torchvision.transforms.functional import resize

from src.diffusion import DiffusionSampler


class Logger(object):

    def __init__(
        self,
        name: str,
        root='./logs',
        config_file='./config.yaml',
    ) -> None:
        self.log_dir = pathlib.Path(root) / name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(config_file, self.log_dir / 'config.yaml')

        self.writer = SummaryWriter(log_dir=self.log_dir / 'tensorboard')

    def get_writer(self) -> SummaryWriter:
        return self.writer

    def add_scalar(self, tag: str, value: Any, step: int) -> None:
        self.writer.add_scalar(tag, value, step, double_precision=True)

    @torch.no_grad()
    def save_state(
        self,
        epoch: int,
        state_dict: dict[str, Any],
        prefix: str = 'ckpt',
    ) -> None:
        model_path = self.log_dir / 'weights' / f'{prefix}_{epoch}.pth'
        model_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(state_dict, model_path)
        shutil.copyfile(model_path, self.log_dir / 'weights' / 'latest.pth')

    @torch.no_grad()
    def save_image(
            self,
            epoch: int,
            sampler: DiffusionSampler,
            sample_x_T: torch.Tensor,
            num_row: int = 8,
            single_image_size=(28, 28),
            prefix: str = 'sample',
    ) -> None:
        sample_path = self.log_dir / 'samples' / f'{prefix}_{epoch}.png'
        sample_path.parent.mkdir(parents=True, exist_ok=True)

        images = sampler.grid_sample(sample_x_T, num_row)
        images = (images + 1.0) / 2.0  # map [-1, 1] to [0, 1]
        images = resize(images, single_image_size)
        grid = utils.make_grid(images, nrow=num_row)

        utils.save_image(grid, sample_path)
        self.writer.add_image(prefix, grid, epoch)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
