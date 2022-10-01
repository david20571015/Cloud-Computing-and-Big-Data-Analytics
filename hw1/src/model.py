import tensorflow as tf


def time_distributed_conv_block(filters, kernel_size):
    return tf.keras.Sequential([
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters, kernel_size, padding='same')),
        tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()),
        tf.keras.layers.ReLU(),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D()),
    ])


def create_model(input_shape=(None, 90, 90, 3), num_classes=39):
    inputs = tf.keras.layers.Input(shape=input_shape)

    outputs = time_distributed_conv_block(32, 3)(inputs)
    outputs = time_distributed_conv_block(32, 3)(outputs)
    outputs = time_distributed_conv_block(32, 3)(outputs)
    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Flatten())(outputs)
    outputs = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True))(outputs)
    outputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(outputs)
    outputs = tf.keras.layers.Dense(num_classes)(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs)
