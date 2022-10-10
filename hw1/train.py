import argparse
import datetime
import json
import os

import tensorflow as tf
import yaml

from src.dataset import get_train_valid_dataset
from src.model import create_model
from src.utilis import test, train


def resume_model_and_opt(config, option):
    if not os.path.isdir(option.resume_training):
        raise ValueError(f'Invalid path: {config["train"]["dir"]}.')

    with open(
            os.path.join(option.resume_training, 'model.json'),
            mode='r',
            encoding='utf-8',
    ) as file:
        model = tf.keras.models.model_from_json(json.load(file))

    optimizer = tf.keras.optimizers.Adam()

    ckpt = tf.train.Checkpoint(net=model, optimizer=optimizer)
    ckpd_dir = os.path.join(option.resume_training, 'weights')
    ckpt.restore(tf.train.latest_checkpoint(ckpd_dir)).assert_consumed()

    return model, optimizer


def main(config, option):
    train_dataset, valid_dataset = get_train_valid_dataset(
        config['dataset']['train_path'],
        config['train']['batch_size'],
        config['train']['train_ratio'],
        config['preprocess']['timesteps'],
        config['preprocess']['crop_height'],
        config['preprocess']['crop_width'],
        config['preprocess']['sample_rate'],
    )

    if option.resume_training:
        model, optimizer = resume_model_and_opt(config, option)
    else:
        model = create_model(
            input_shape=(config['preprocess']['timesteps'],
                         config['preprocess']['crop_height'],
                         config['preprocess']['crop_width'], 3),
            num_classes=config['model']['num_classes'],
        )
        with open(
                os.path.join(config['log_dir'], 'model.json'),
                mode='w',
                encoding='utf-8',
        ) as file:
            json.dump(model.to_json(), file)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config['train']['learning_rate'])

    model.summary()  # type: ignore

    train_writer = tf.summary.create_file_writer(  # type: ignore
        os.path.join(config['log_dir'], 'tensorboard', 'train'))
    test_writer = tf.summary.create_file_writer(  # type: ignore
        os.path.join(config['log_dir'], 'tensorboard', 'test'))

    ckpt = tf.train.Checkpoint(net=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(
        ckpt,
        os.path.join(config['log_dir'], 'weights'),
        max_to_keep=5,
    )

    for epoch in range(1, config['train']['num_epochs'] + 1):
        print(f'Epoch {epoch}/{config["train"]["num_epochs"]}:')

        train(model, optimizer, train_dataset, train_writer, epoch)
        test(model, valid_dataset, test_writer, epoch)
        manager.save()


if __name__ == '__main__':
    print('CUDA: ', tf.config.list_physical_devices('GPU'))

    with open('config.yaml', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg['log_dir'] = os.path.join(cfg['train']['log_dir'], current_time)
    os.makedirs(cfg['log_dir'], exist_ok=True)

    with open(os.path.join(cfg['log_dir'], 'config.yaml'),
              mode='w',
              encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--resume-training',
        '-r',
        default='',
        type=str,
        help='Path to the log directory of the training to resume.',
    )
    args = parser.parse_args()

    main(cfg, args)
