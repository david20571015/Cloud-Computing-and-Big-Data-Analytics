import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import yaml

from src.dataset import GaussianNoiseDataset
from src.diffusion import DiffusionSampler
from src.models import UNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l',
        '--logdir',
        type=str,
        required=True,
        help='log dir path, e.g. ./logs/2022-01-01_00-00-00',
    )
    parser.add_argument(
        '-w',
        '--weight',
        type=str,
        default='latest',
        help='file name of model weight, e.g. ckpt_100 (default: latest)',
    )
    args = parser.parse_args()

    with open(Path(args.logdir) / 'config.yaml', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    Path('images').mkdir(exist_ok=True)

    # prepare dataset
    input_shape = (3, 28, 28)
    dataset = GaussianNoiseDataset(input_shape, 10000)
    dataloader = DataLoader(dataset,
                            batch_size=128,
                            num_workers=torch.get_num_threads(),
                            pin_memory=True)

    # prepare model
    model = UNet(input_shape=input_shape, **config['model'])
    checkpoint = torch.load(
        Path(args.logdir) / 'weights' / f'{args.weight}.pth')
    model.load_state_dict(checkpoint['ema_model'])

    sampler = DiffusionSampler(
        model, time_steps=config['model']['time_steps']).to(DEVICE)
    sampler.eval()

    image_id = 1
    with torch.no_grad():
        assert sampler(dataset[0][None, ...].to(DEVICE)).shape == (1, 3, 28, 28)

        for x_T in tqdm(dataloader, dynamic_ncols=True):
            x_T = x_T.to(DEVICE)
            x_0 = sampler(x_T)

            for image in x_0:
                save_image(image, Path('images') / f'{image_id:05d}.png')
                image_id += 1


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
