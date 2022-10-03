import json
import os

import tensorflow as tf
import yaml

from src.dataset import get_dataset


def infer(model, data):
    pred = model(data, training=False)
    return tf.argmax(pred, axis=-1)


def main(config):
    dataset = get_dataset(config['dataset']['test_path'], 'test',
                          config['infer']['batch_size'])

    with open(config['model_struct_path'], 'r', encoding='utf-8') as file:
        model = tf.keras.models.model_from_json(json.load(file))
    model.summary()  # type: ignore

    ckpt = tf.train.Checkpoint(net=model)
    ckpt.restore(
        tf.train.latest_checkpoint(
            os.path.join(config['infer']['dir'], 'weights'))).expect_partial()

    with open('prediction.csv', 'w', encoding='utf-8') as file:
        print('name,label', file=file)

        progbar = tf.keras.utils.Progbar(dataset.cardinality().numpy())

        for data, filename in dataset:
            pred = infer(model, data)

            for name, label in zip(filename, pred):
                print(f'{name.numpy().decode("utf-8")},{label}', file=file)

            progbar.add(1)


if __name__ == '__main__':
    with open('config.yaml', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg['model_struct_path'] = os.path.join(cfg['infer']['dir'], 'model.json')

    main(cfg)
