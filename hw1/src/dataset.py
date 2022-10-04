import os
from typing import Any

import tensorflow as tf
import tensorflow_io as tfio
from sklearn.model_selection import train_test_split


@tf.function
def _decode_video(
    video_path: str,
    height: int = 90,
    weight: int = 90,
    sample_rate: int = 1,
) -> tf.Tensor:
    """Decode video to tensor.

    Returns:
        tf.Tensor: shape=(frames, height, width, channels), dtype=tf.float32
        where height and width are 90, and the value range is [-1.0, 1.0].

    """

    video = tf.io.read_file(video_path)
    video = tfio.experimental.ffmpeg.decode_video(video)
    video = video[::sample_rate, ...]
    video = tf.image.resize_with_crop_or_pad(video, height, weight)

    # [0, 255] -> [-1.0, 1.0]
    video = tf.cast(video, tf.float32) / 128.0 - 1.0  # type: ignore
    return video


def _get_dataset(
    video_data_list: list[str],
    remain_data_list: list[Any],
    batch_size: int = 32,
    shuffle=True,
    height: int = 90,
    weight: int = 90,
    sample_rate: int = 1,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(video_data_list),
        tf.data.Dataset.from_tensor_slices(remain_data_list),
    ))

    if shuffle:
        dataset = dataset.shuffle(len(video_data_list))

    dataset = dataset.map(
        lambda x, y: (_decode_video(x, height, weight, sample_rate), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def get_train_valid_dataset(
    path: str,
    batch_size: int = 32,
    train_ratio=0.9,
    height: int = 90,
    weight: int = 90,
    sample_rate: int = 1,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:

    if not os.path.isdir(path):
        raise ValueError(f'Invalid path: {path}')

    file_list = tf.io.gfile.glob(os.path.join(path, '*', '*.mp4'))
    train_file_list, valid_file_list = train_test_split(file_list,
                                                        train_size=train_ratio)
    train_class_list, valid_class_list = [
        int(os.path.basename(os.path.dirname(f))) for f in train_file_list
    ], [int(os.path.basename(os.path.dirname(f))) for f in valid_file_list]

    print(f'Found {len(train_file_list)} training videos '
          f'and {len(valid_file_list)} validation videos.')

    train_dataset = _get_dataset(
        train_file_list,
        train_class_list,
        batch_size,
        height=height,
        weight=weight,
        sample_rate=sample_rate,
    )
    valid_dataset = _get_dataset(
        valid_file_list,
        valid_class_list,
        batch_size,
        shuffle=False,
        height=height,
        weight=weight,
        sample_rate=sample_rate,
    )

    return train_dataset, valid_dataset


def get_test_dataset(
    path: str,
    batch_size: int = 32,
    height: int = 90,
    weight: int = 90,
    sample_rate: int = 1,
) -> tf.data.Dataset:

    if not os.path.isdir(path):
        raise ValueError(f'Invalid path: {path}')

    file_list = tf.io.gfile.glob(os.path.join(path, '*.mp4'))
    filename_list = [os.path.basename(f) for f in file_list]
    print(f'Found {len(file_list)} testing videos.')

    dataset = _get_dataset(
        file_list,
        filename_list,
        batch_size,
        shuffle=False,
        height=height,
        weight=weight,
        sample_rate=sample_rate,
    )

    return dataset
