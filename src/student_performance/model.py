"""TensorFlow/Keras ANN architecture."""

from __future__ import annotations


def build_ann(input_size: int, task: str, learning_rate: float = 0.001):
    import tensorflow as tf

    if task not in {"classification", "regression"}:
        raise ValueError("task must be classification or regression")
    output_activation = "sigmoid" if task == "classification" else "linear"
    loss = "binary_crossentropy" if task == "classification" else "mse"
    metrics = ["accuracy"] if task == "classification" else ["mae"]

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_size,), name="student_features"),
            tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, activation=output_activation, name="prediction"),
        ],
        name=f"student_performance_{task}",
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss, metrics=metrics)
    return model

