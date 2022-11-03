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
        '--log-name',
        type=str,
        required=True,
        help='log dir name',
    )
    args = parser.parse_args()

    with open(Path(args.log_name) / 'config.yaml', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset = EmbeddingDataset('data/unlabeled')
    dataloader = DataLoader(dataset,
                            batch_size=64,
                            shuffle=False,
                            num_workers=torch.get_num_threads(),
                            pin_memory=True)

    encoder = create_encoder(config['model']).cuda()
    checkpoint = torch.load(Path(args.log_name) / 'weights' / 'latest.pth')
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
