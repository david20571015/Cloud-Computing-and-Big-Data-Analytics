import datetime
import os

import tensorflow as tf
import yaml

from src.dataset import get_dataset
from src.model import create_model
from src.utilis import test, train


def main(config):
    dataset = get_dataset(config['train_data_path'], 'train',
                          config['batch_size'])

    train_size = int(dataset.cardinality().numpy() * 0.9)
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    model = create_model(
        input_shape=(None, config['crop_height'], config['crop_width'], 3),
        num_classes=config['num_classes'],
    )
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])

    train_writer = tf.summary.create_file_writer(  # type: ignore
        config['train_log_dir'])
    test_writer = tf.summary.create_file_writer(  # type: ignore
        config['test_log_dir'])

    for epoch in range(1, config['num_epochs'] + 1):
        print(f'Epoch {epoch}/{config["num_epochs"]}:')

        train(model, optimizer, train_dataset, train_writer, epoch)
        test(model, test_dataset, test_writer, epoch)


if __name__ == '__main__':
    print('CUDA: ', tf.config.list_physical_devices('GPU'))

    with open('config.yaml', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg['log_path'] = os.path.join(cfg['log_path'], current_time)
    cfg['train_log_dir'] = os.path.join(cfg['log_path'], 'tensorboard', 'train')
    cfg['test_log_dir'] = os.path.join(cfg['log_path'], 'tensorboard', 'test')

    main(cfg)
