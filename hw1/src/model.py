import tensorflow as tf


def conv_block(filters: int, repeat: int):
    layers = []
    for _ in range(repeat):
        layers.append(
            tf.keras.layers.Conv2D(
                filters,
                3,
                padding='same',
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
        layers.append(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.BatchNormalization()))
        layers.append(
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.2)))
    layers.append(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D()))
    return tf.keras.Sequential(layers)


def vgg19_backend():
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

    outputs = vgg19_backend()(outputs)
    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Flatten())(outputs)
    outputs = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            128,
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)))(outputs)
    outputs = attention_block(outputs, input_shape[0])
    outputs = tf.keras.layers.Flatten()(outputs)
    outputs = tf.keras.layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.0001))(outputs)
    outputs = tf.keras.layers.Dropout(0.5)(outputs)
    outputs = tf.keras.layers.Dense(
        num_classes,
        kernel_regularizer=tf.keras.regularizers.l2(0.0001))(outputs)

    return tf.keras.Model(inputs, outputs)
