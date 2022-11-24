import argparse
import copy
import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
import yaml

from src.dataset import MnistDataset
from src.diffusion import DiffusionSampler
from src.diffusion import DiffusionTrainer
from src.logger import Logger
from src.models import UNet
from src.utils import ExponentialMovingAverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='./config.yaml',
        help='config file path (default: ./config.yaml)',
    )
    parser.add_argument(
        '-l',
        '--logdir',
        type=str,
        default=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        help='log dir name (default: current time)',
    )
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    logger = Logger(args.logdir, config_file=args.config)

    # prepare dataset
    dataset = MnistDataset(
        transform=T.Resize((32, 32), InterpolationMode.NEAREST))
    dataloader = DataLoader(dataset,
                            batch_size=config['train']['batch_size'],
                            shuffle=True,
                            num_workers=torch.get_num_threads(),
                            pin_memory=True)

    sample_noise = torch.randn(8, *dataset[0].shape).to(DEVICE)

    # prepare model
    model = UNet(input_shape=dataset[0].shape, **config['model'])
    trainer = DiffusionTrainer(
        model, time_steps=config['model']['time_steps']).to(DEVICE)
    sampler = DiffusionSampler(
        model, time_steps=config['model']['time_steps']).to(DEVICE)

    ema_model = copy.deepcopy(model)
    ema_sampler = DiffusionSampler(
        ema_model, time_steps=config['model']['time_steps']).to(DEVICE)

    ema = ExponentialMovingAverage(model,
                                   ema_model,
                                   decay=config['train']['ema_decay'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])

    for epoch in range(1, config['train']['epochs'] + 1):

        # training
        trainer.train()
        with tqdm(
                dataloader,
                desc=f'Epoch {epoch} / {config["train"]["epochs"]}',
                dynamic_ncols=True,
        ) as pbar:
            for x_0 in pbar:
                x_0 = x_0.to(DEVICE)

                optimizer.zero_grad()
                loss = trainer(x_0)
                loss.backward()
                optimizer.step()
                ema.step()

                logger.add_scalar('loss/train', loss, epoch)
                pbar.set_postfix_str(f'loss: {loss:.6f}')

        # save
        if epoch % config['train']['save_freq'] == 0 or epoch == 1:
            states = {
                'epoch': epoch,
                'model': model.state_dict(),
                'ema_model': ema_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'sample_noise': sample_noise,
            }
            logger.save_state(epoch, states)

            sampler.eval()
            logger.save_image(epoch, sampler, sample_noise, prefix='org')
            ema_sampler.eval()
            logger.save_image(epoch, ema_sampler, sample_noise, prefix='ema')

    logger.close()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
