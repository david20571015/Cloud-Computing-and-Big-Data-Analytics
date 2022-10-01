import tensorflow as tf


@tf.function
def train_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_metric: tf.keras.metrics.Mean,
    acc_metric: tf.keras.metrics.SparseCategoricalAccuracy,
    data: tf.Tensor,
    label: tf.Tensor,
):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    with tf.GradientTape() as tape:
        logits = model(data, training=True)
        loss = loss_fn(label, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    loss_metric.update_state(loss)
    acc_metric.update_state(label, logits)


def train(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    dataset: tf.data.Dataset,
    summary_writer: tf.summary.SummaryWriter,  # type: ignore
    epoch: int,
):

    loss_metric = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    train_progbar = tf.keras.utils.Progbar(
        dataset.cardinality().numpy(),
        stateful_metrics=[loss_metric.name, acc_metric.name],
    )

    for data, label in dataset:
        train_step(model, optimizer, loss_metric, acc_metric, data, label)
        values = [
            (loss_metric.name, loss_metric.result()),  # pylint: disable=not-callable
            (acc_metric.name, acc_metric.result()),  # pylint: disable=not-callable
        ]
        train_progbar.add(1, values=values)

    with summary_writer.as_default():
        tf.summary.scalar('loss', loss_metric.result(), step=epoch)  # pylint: disable=not-callable
        tf.summary.scalar('accuracy', acc_metric.result(), step=epoch)  # pylint: disable=not-callable

    loss_metric.reset_states()
    acc_metric.reset_states()


@tf.function
def test_step(
    model: tf.keras.Model,
    loss_metric: tf.keras.metrics.Mean,
    acc_metric: tf.keras.metrics.SparseCategoricalAccuracy,
    data: tf.Tensor,
    label: tf.Tensor,
):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    logits = model(data, training=False)
    loss = loss_fn(label, logits)

    loss_metric.update_state(loss)
    acc_metric.update_state(label, logits)


def test(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    summary_writer: tf.summary.SummaryWriter,  # type: ignore
    epoch: int,
):
    loss_metric = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    test_progbar = tf.keras.utils.Progbar(
        dataset.cardinality().numpy(),
        stateful_metrics=[loss_metric.name, acc_metric.name],
    )

    for data, label in dataset:
        test_step(model, loss_metric, acc_metric, data, label)
        values = [
            (loss_metric.name, loss_metric.result()),  # pylint: disable=not-callable
            (acc_metric.name, acc_metric.result()),  # pylint: disable=not-callable
        ]
        test_progbar.add(1, values=values)

    with summary_writer.as_default():
        tf.summary.scalar('loss', loss_metric.result(), step=epoch)  # pylint: disable=not-callable
        tf.summary.scalar('accuracy', acc_metric.result(), step=epoch)  # pylint: disable=not-callable

    loss_metric.reset_states()
    acc_metric.reset_states()
