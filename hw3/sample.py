import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize
from torchvision.utils import save_image
from tqdm import tqdm
import yaml

from src.dataset import GaussianNoiseDataset
from src.diffusion import DdimSampler
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
    parser.add_argument(
        '-n',
        '--num_samples',
        type=int,
        default=10000,
        help='number of samples (default: 10000)',
    )
    parser.add_argument(
        '--ddim',
        action='store_true',
        help='use ddim sampler',
    )
    parser.add_argument(
        '-s',
        '--step',
        type=int,
        default=10,
        help=('number of ddim sampling steps, affect if `--ddim` is set '
              '(default: 10)'),
    )
    parser.add_argument(
        '--eta',
        type=float,
        default=0.0,
        help=('linearity between ddim sampling and ddpm sampling, 0.0 for pure '
              'ddim sampliing, 1.0 for pure ddpm sampling, affect if `--ddim` '
              'is set (default: 0.0)'),
    )
    args = parser.parse_args()

    with open(Path(args.logdir) / 'config.yaml', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    save_dir = Path(args.logdir) / (f'images_ddim_{args.eta}_step_{args.step}'
                                    if args.ddim else 'images')
    save_dir.mkdir(exist_ok=True)

    # prepare dataset
    input_shape = (3, 32, 32)
    dataset = GaussianNoiseDataset(input_shape, args.num_samples)
    dataloader = DataLoader(dataset,
                            batch_size=512,
                            num_workers=torch.get_num_threads(),
                            pin_memory=True)

    # prepare model
    model = UNet(input_shape=input_shape, **config['model']).to(DEVICE)
    checkpoint = torch.load(
        Path(args.logdir) / 'weights' / f'{args.weight}.pth')
    model.load_state_dict(checkpoint['ema_model'])

    if args.ddim:
        sampler = DdimSampler(model,
                              time_steps=config['model']['time_steps'],
                              sample_steps=args.step).to(DEVICE)
    else:
        sampler = DiffusionSampler(
            model, time_steps=config['model']['time_steps']).to(DEVICE)
    sampler.eval()

    image_id = 1
    with torch.no_grad():
        assert sampler(dataset[0][None, ...].to(DEVICE)).shape == (1, 3, 32, 32)

        for x_T in tqdm(dataloader, dynamic_ncols=True):
            x_T = x_T.to(DEVICE)

            if args.ddim:
                x_0 = sampler(x_T, eta=args.eta)
            else:
                x_0 = sampler(x_T)
            x_0 = resize(x_0, [28, 28])

            for image in x_0:
                save_image(image, save_dir / f'{image_id:05d}.png')
                image_id += 1


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
