import pathlib
import shutil
from typing import Any

import torch
from torch.utils.tensorboard.writer import SummaryWriter


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

    def save(
        self,
        state_dict: dict[str, Any],
        name: str = 'model',
    ) -> None:
        model_path = self.log_dir / 'weights' / name
        model_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(state_dict, model_path)
        shutil.copyfile(model_path, self.log_dir / 'weights' / 'latest.pth')

    def __del__(self) -> None:
        self.writer.close()
