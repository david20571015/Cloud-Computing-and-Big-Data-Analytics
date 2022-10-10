import os
from typing import Any, Optional

import tensorflow as tf
import tensorflow_io as tfio
from sklearn.model_selection import train_test_split


@tf.function
def _decode_video(
    video_path: str,
    timesteps: Optional[int] = None,
    height: int = 90,
    width: int = 90,
    sample_rate: int = 1,
    data_augment: bool = False,
) -> tf.Tensor:
    """Decode video to tensor.

    Returns:
        tf.Tensor: shape=(frames, height, width, channels), dtype=tf.float32
        where height and width are 90, and the value range is [-1.0, 1.0].

    """

    video = tf.io.read_file(video_path)
    video = tfio.experimental.ffmpeg.decode_video(video)
    video = video[::sample_rate, ...]
    video = tf.image.resize_with_pad(video, height, width)

    if data_augment:
        video = tf.image.random_hue(video, 0.2)
        video = tf.image.random_saturation(video, 0.8, 1.2)
        video = tf.image.random_brightness(video, 0.2)

    if timesteps is not None:
        video_timesetps = tf.shape(video)[0]
        if video_timesetps > timesteps:
            video = video[:timesteps, ...]  # type: ignore
        else:
            video = tf.pad(
                video,
                [[0, timesteps - video_timesetps], [0, 0], [0, 0], [0, 0]],
                'CONSTANT',
                -1,
            )

    # [0, 255] -> [0.0, 1.0]
    video = tf.cast(video, tf.float32) / 255.0  # type: ignore
    return video


def _get_dataset(
    video_data_list: list[str],
    remain_data_list: list[Any],
    batch_size: int = 32,
    shuffle=True,
    timesteps: Optional[int] = None,
    height: int = 90,
    width: int = 90,
    sample_rate: int = 1,
    data_augment: bool = False,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(video_data_list),
        tf.data.Dataset.from_tensor_slices(remain_data_list),
    ))

    if shuffle:
        dataset = dataset.shuffle(len(video_data_list))

    dataset = dataset.map(
        lambda x, y:
        (_decode_video(x, timesteps, height, width, sample_rate, data_augment), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if timesteps is None:
        dataset = dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(batch_size))
    else:
        dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def get_train_valid_dataset(
    path: str,
    batch_size: int = 32,
    train_ratio=0.9,
    timesteps: Optional[int] = None,
    height: int = 90,
    width: int = 90,
    sample_rate: int = 1,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:

    if not os.path.isdir(path):
        raise ValueError(f'Invalid path: {path}')

    file_list = tf.io.gfile.glob(os.path.join(path, '*', '*.mp4'))
    train_file_list, valid_file_list = train_test_split(file_list,
                                                        train_size=train_ratio,
                                                        random_state=42)
    train_class_list, valid_class_list = [
        int(os.path.basename(os.path.dirname(f))) for f in train_file_list
    ], [int(os.path.basename(os.path.dirname(f))) for f in valid_file_list]

    print(f'Found {len(train_file_list)} training videos '
          f'and {len(valid_file_list)} validation videos.')

    train_dataset = _get_dataset(
        train_file_list,
        train_class_list,
        batch_size,
        timesteps=timesteps,
        height=height,
        width=width,
        sample_rate=sample_rate,
        data_augment=True,
    )
    valid_dataset = _get_dataset(
        valid_file_list,
        valid_class_list,
        batch_size,
        shuffle=False,
        timesteps=timesteps,
        height=height,
        width=width,
        sample_rate=sample_rate,
    )

    return train_dataset, valid_dataset


def get_test_dataset(
    path: str,
    batch_size: int = 32,
    timesteps: Optional[int] = None,
    height: int = 90,
    width: int = 90,
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
        timesteps=timesteps,
        height=height,
        width=width,
        sample_rate=sample_rate,
    )

    return dataset
