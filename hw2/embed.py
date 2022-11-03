import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import EmbeddingDataset
from src.utils import create_encoder


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

    dataset = EmbeddingDataset('data/unlabeled')
    dataloader = DataLoader(dataset,
                            batch_size=64,
                            shuffle=False,
                            num_workers=torch.get_num_threads(),
                            pin_memory=True)

    encoder = create_encoder(config['model']).cuda()
    checkpoint = torch.load(
        Path(args.logdir) / 'weights' / f'{args.weight}.pth')
    encoder.load_state_dict(checkpoint['encoder'])
    encoder.eval()

    embeddings = torch.zeros(len(dataset), 512)

    with torch.no_grad():
        for images, label in tqdm(dataloader):
            images = images.cuda()
            features = encoder(images)

            embeddings[label.long()] = features.cpu()

    np.save('embeddings.npy', embeddings.numpy())


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True  # type: ignore
    main()
