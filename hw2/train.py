import argparse
import datetime

import torch
import torchvision
import torchvision.transforms as T
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.dataset import UnlabeledDataset
from src.logger import Logger
from src.utils import creaete_projector, create_encoder, knn_classify, nt_xent


def create_dataloaders(batch_size):
    train_dataset = UnlabeledDataset(
        'data/unlabeled',
        transform=T.Compose([
            T.RandomResizedCrop(96),
            T.RandomHorizontalFlip(),
            T.RandomApply(
                [T.ColorJitter(0.8, 0.8, 0.8, 0.2)],  # type: ignore
                p=0.8),
            T.GaussianBlur(9, sigma=(0.1, 2.0))
        ]))
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=torch.get_num_threads(),
                                  pin_memory=True)

    valid_dataset = ImageFolder('data/test',
                                transform=T.Lambda(lambda x: x / 255.),
                                loader=lambda x: torchvision.io.read_image(
                                    x, torchvision.io.image.ImageReadMode.GRAY))
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=torch.get_num_threads(),
                                  pin_memory=True)

    return train_dataloader, valid_dataloader


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    dataloader: DataLoader,
    accu_steps: int,
):
    loss_sum = 0
    loss_num = 0

    model.train()

    with tqdm(dataloader) as pbar:
        n_iters = 0

        for image, trans_image, _ in pbar:
            image, trans_image = image.cuda(), trans_image.cuda()
            u = model(image)
            v = model(trans_image)

            loss = nt_xent(u, v)
            loss_sum += loss.item()
            loss_num += 1
            pbar.set_postfix_str(f'loss: {loss:.6f}')

            loss = loss / accu_steps
            loss.backward()

            n_iters += 1

            if n_iters % accu_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()

    return loss_sum / loss_num


def valid_step(model, dataloader):
    model.eval()

    with torch.no_grad():
        embedding = []
        classes = []
        for image, cls in dataloader:
            image = image.cuda()
            embedding.append(model(image))
            classes.append(cls)
        embedding = torch.cat(embedding, dim=0)
        classes = torch.cat(classes, dim=0)
        acc = knn_classify(embedding, classes, 128)
        print(f'Test Accuracy: {acc:.5f}')

    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='./config.yaml',
        help='config file path',
    )
    parser.add_argument(
        '-l',
        '--log-name',
        type=str,
        default=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        help='log dir name',
    )
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    logger = Logger(args.log_name, config_file=args.config)

    batch_size = config['train']['batch_size']
    accu_steps = config['train']['update_batch_size'] // batch_size

    train_dataloader, valid_dataloader = create_dataloaders(batch_size)

    encoder = create_encoder(config['model'])
    projector = creaete_projector(config['model'])
    model = nn.Sequential(encoder, projector).cuda()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['train']['lr'],
                                 weight_decay=config['train']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 10, 2)

    epochs = config['train']['epochs']

    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1}/{epochs}')

        train_loss = train_step(
            model,
            optimizer,
            scheduler,
            train_dataloader,
            accu_steps,
        )
        logger.get_writer().add_scalar('loss/train', train_loss, epoch + 1)

        valid_acc = valid_step(model, valid_dataloader)
        logger.get_writer().add_scalar('acc/valid', valid_acc, epoch + 1)

        if (epoch + 1) % config['train']['save_freq'] == 0:
            logger.save(
                {
                    'encoder': encoder.state_dict(),
                    'projector': projector.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch + 1,
                },
                f'ckpt_{epoch+1}.pth',
            )


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True  # type: ignore
    main()
