import tensorflow as tf

from src.resnet import resnet_backbone


def conv_block(filters: int, repeat: int):
    layers = []
    for _ in range(repeat):
        layers.append(
            tf.keras.layers.Conv2D(
                filters,
                3,
                padding='same',
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
        layers.append(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.BatchNormalization()))
    layers.append(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D()))
    layers.append(tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.2)))
    return tf.keras.Sequential(layers)


def vgg19_backbone():
    return tf.keras.Sequential([
        conv_block(64, 2),
        conv_block(128, 2),
        conv_block(256, 4),
        conv_block(512, 4),
        conv_block(512, 4),
    ])


def attention_block(inputs, timesteps):
    keys = tf.keras.layers.Permute((2, 1))(inputs)
    keys = tf.keras.layers.Dense(timesteps, activation='softmax')(keys)
    keys = tf.keras.layers.Permute((2, 1))(keys)
    return tf.keras.layers.Multiply()([inputs, keys])


def create_model(input_shape: tuple, num_classes: int):
    inputs = tf.keras.Input(input_shape)

    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.RandomFlip('horizontal'))(inputs)

    outputs = resnet_backbone('resnet18')(outputs)
    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Flatten())(outputs)
    outputs = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            64, kernel_regularizer=tf.keras.regularizers.l2(0.0005)))(outputs)
    outputs = tf.keras.layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.0005))(outputs)
    outputs = tf.keras.layers.Dropout(0.5)(outputs)
    outputs = tf.keras.layers.Dense(
        num_classes,
        kernel_regularizer=tf.keras.regularizers.l2(0.0005))(outputs)

    return tf.keras.Model(inputs, outputs)
