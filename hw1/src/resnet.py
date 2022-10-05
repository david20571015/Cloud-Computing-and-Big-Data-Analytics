import tensorflow as tf

Kernel = tuple[int, int]  # (filters, kernel_size)
Spec = list[tuple[list[Kernel], int]]

RESNET_SPEC: dict[str, Spec] = {
    'resnet18': [
        ([(64, 3), (64, 3)], 2),
        ([(128, 3), (128, 3)], 2),
        ([(256, 3), (256, 3)], 2),
        ([(512, 3), (512, 3)], 2),
    ],
    'resnet34': [
        ([(64, 3), (64, 3)], 3),
        ([(128, 3), (128, 3)], 4),
        ([(256, 3), (256, 3)], 6),
        ([(512, 3), (512, 3)], 3),
    ],
    'resnet50': [
        ([(64, 1), (64, 3), (256, 1)], 3),
        ([(128, 1), (128, 3), (512, 1)], 4),
        ([(256, 1), (256, 3), (1024, 1)], 6),
        ([(512, 1), (512, 3), (2048, 1)], 3),
    ],
    'resnet101': [
        ([(64, 1), (64, 3), (256, 1)], 3),
        ([(128, 1), (128, 3), (512, 1)], 4),
        ([(256, 1), (256, 3), (1024, 1)], 23),
        ([(512, 1), (512, 3), (2048, 1)], 3),
    ],
    'resnet152': [
        ([(64, 1), (64, 3), (256, 1)], 3),
        ([(128, 1), (128, 3), (512, 1)], 8),
        ([(256, 1), (256, 3), (1024, 1)], 36),
        ([(512, 1), (512, 3), (2048, 1)], 3),
    ],
}


class ResidualBlock(tf.keras.Model):

    def __init__(self, kernels: list[Kernel]):
        super().__init__()

        layers = []
        for filters, kernel_size in kernels:
            layers.append(
                tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Conv2D(filters, kernel_size,
                                           padding='same')))
            layers.append(
                tf.keras.layers.TimeDistributed(
                    tf.keras.layers.BatchNormalization()))
            layers.append(tf.keras.layers.ReLU())
        layers.pop()  # Remove last ReLU

        self.conv_layers = tf.keras.Sequential(layers)

        self.identity = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(
                filters=kernels[-1][0],
                kernel_size=1,
                padding='same',
            ))

    def call(self, inputs, training=False, mask=None):
        conv_output = self.conv_layers(inputs, training=training)
        identity_output = self.identity(inputs)
        outputs = tf.keras.layers.add([conv_output, identity_output])
        return tf.keras.layers.ReLU()(outputs)


class ConvBlock(tf.keras.Model):

    def __init__(self, kernels: list[Kernel], repeat: int):
        super().__init__()

        self.downsample = tf.keras.layers.TimeDistributed(
            tf.keras.layers.MaxPool2D(3, strides=2, padding='same'))

        layers = []
        for _ in range(repeat):
            layers.append(ResidualBlock(kernels))
        self.residual_blocks = tf.keras.Sequential(layers)

    def call(self, inputs, training=False, mask=None):
        outputs = self.downsample(inputs)
        return self.residual_blocks(outputs, training=training)


class ResNet(tf.keras.Model):

    def __init__(self, output_features: int, spec: Spec):
        super().__init__()

        self.head = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(64, 7, strides=2, padding='same'))
        self.conv_blocks = tf.keras.Sequential(
            [ConvBlock(kernels, repeat) for kernels, repeat in spec])
        self.pool = tf.keras.layers.TimeDistributed(
            tf.keras.layers.GlobalAveragePooling2D())
        self.classifier = tf.keras.layers.Dense(output_features)

    def call(self, inputs, training=False, mask=None):
        outputs = self.head(inputs)
        outputs = self.conv_blocks(outputs, training=training)
        outputs = self.pool(outputs)
        return self.classifier(outputs)


def create_lstm_resnet(
    input_shape,
    num_classes: int,
    name: str,
):
    inputs = tf.keras.Input(shape=input_shape)
    outputs = ResNet(128, RESNET_SPEC[name])(inputs)
    outputs = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))(outputs)
    outputs = tf.keras.layers.Dense(num_classes)(outputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
