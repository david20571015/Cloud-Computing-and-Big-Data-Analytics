import os

import tensorflow as tf
import tensorflow_io as tfio


@tf.function
def _decode_video(video_path: str) -> tf.Tensor:
    """Decode video to tensor.

    Returns:
        tf.Tensor: shape=(frames, height, width, channels), dtype=tf.float32
        where height and width are 90, and the value range is [-1.0, 1.0].

    """

    video = tf.io.read_file(video_path)
    video = tfio.experimental.ffmpeg.decode_video(video)
    video = tf.image.resize_with_crop_or_pad(video, 90, 90)

    # [0, 255] -> [-1.0, 1.0]
    video = tf.cast(video, tf.float32) / 128.0 - 1.0  # type: ignore
    return video


def get_dataset(
    path: str,
    ds_type: str,
    batch_size: int = 32,
) -> tf.data.Dataset:

    if not os.path.isdir(path):
        raise ValueError(f'Invalid path: {path}')

    if ds_type not in ['train', 'test']:
        raise ValueError('ds_type must be "train" or "test"')

    if ds_type == 'train':
        file_list = tf.io.gfile.glob(os.path.join(path, '*', '*.mp4'))
        class_list = [
            int(os.path.basename(os.path.dirname(f))) for f in file_list
        ]
        print(f'Found {len(file_list)} training videos.')

        dataset = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(file_list),
            tf.data.Dataset.from_tensor_slices(class_list),
        ))
        dataset = dataset.cache().shuffle(len(file_list))
        dataset = dataset.map(lambda x, y: (_decode_video(x), y),
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(batch_size))

    else:  # test
        file_list = tf.io.gfile.glob(os.path.join(path, '*.mp4'))
        filename_list = [os.path.basename(f) for f in file_list]
        print(f'Found {len(file_list)} testing videos.')

        dataset = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(file_list),
            tf.data.Dataset.from_tensor_slices(filename_list),
        ))
        dataset = dataset.map(lambda x, y: (_decode_video(x), y),
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(batch_size))

    return dataset.prefetch(tf.data.AUTOTUNE)
