import datetime
import json
import os

import tensorflow as tf
import yaml

from src.dataset import get_dataset
from src.model import create_model
from src.utilis import test, train


def main(config):
    dataset = get_dataset(config['dataset']['train_path'], 'train',
                          config['train']['batch_size'])

    train_size = int(dataset.cardinality().numpy() * 0.9)
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    model = create_model(
        input_shape=(None, config['preprocess']['crop_height'],
                     config['preprocess']['crop_width'], 3),
        num_classes=config['model']['num_classes'],
    )
    model.summary()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config['train']['learning_rate'])

    train_writer = tf.summary.create_file_writer(  # type: ignore
        config['train_log_dir'])
    test_writer = tf.summary.create_file_writer(  # type: ignore
        config['test_log_dir'])

    for epoch in range(1, config['train']['num_epochs'] + 1):
        print(f'Epoch {epoch}/{config["train"]["num_epochs"]}:')

        train(model, optimizer, train_dataset, train_writer, epoch)
        test(model, test_dataset, test_writer, epoch)


if __name__ == '__main__':
    print('CUDA: ', tf.config.list_physical_devices('GPU'))

    with open('config.yaml', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg['log_dir'] = os.path.join(cfg['log_dir'], current_time)
    os.makedirs(cfg['log_dir'], exist_ok=True)

    with open(os.path.join(cfg['log_dir'], 'config.yaml'),
              mode='w',
              encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    cfg['train_log_dir'] = os.path.join(cfg['log_dir'], 'tensorboard', 'train')
    cfg['test_log_dir'] = os.path.join(cfg['log_dir'], 'tensorboard', 'test')

    main(cfg)
