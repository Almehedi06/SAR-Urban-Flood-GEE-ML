from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from .utils import label_maps


def train(X_train, y_train, classes: list[int], cfg: dict[str, Any], X_val=None, y_val=None) -> dict[str, Any]:
    if X_val is None or y_val is None:
        raise ValueError("cnn_1d needs validation data for early stopping.")

    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError as exc:
        raise ImportError("Install the 'cnn' extra to train CNN models.") from exc

    random_state = int(cfg.get("random_state", 42))
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    X_train = np.asarray(X_train, dtype="float32")
    X_val = np.asarray(X_val, dtype="float32")
    y_train = np.asarray(y_train, dtype="int32")
    y_val = np.asarray(y_val, dtype="int32")

    label_to_index, _ = label_maps(classes)
    y_train_index = np.array([label_to_index[int(label)] for label in y_train], dtype="int32")
    y_val_index = np.array([label_to_index[int(label)] for label in y_val], dtype="int32")

    scaler = StandardScaler()
    X_train_cnn = scaler.fit_transform(X_train).astype("float32").reshape(-1, X_train.shape[1], 1)
    X_val_cnn = scaler.transform(X_val).astype("float32").reshape(-1, X_train.shape[1], 1)

    class_weights = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y_train_index)
    class_weight = {idx: float(weight) for idx, weight in enumerate(class_weights)}

    model = keras.Sequential(
        [
            layers.Input(shape=(X_train.shape[1], 1)),
            layers.Conv1D(
                int(cfg.get("filters_1", 32)),
                kernel_size=int(cfg.get("kernel_size", 2)),
                padding="same",
                activation="relu",
            ),
            layers.BatchNormalization(),
            layers.Conv1D(
                int(cfg.get("filters_2", 64)),
                kernel_size=int(cfg.get("kernel_size", 2)),
                padding="same",
                activation="relu",
            ),
            layers.GlobalAveragePooling1D(),
            layers.Dense(int(cfg.get("dense_units", 64)), activation="relu"),
            layers.Dropout(float(cfg.get("dropout", 0.30))),
            layers.Dense(len(classes), activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(cfg.get("learning_rate", 0.001))),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(cfg.get("patience", 20)),
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5),
    ]
    history = model.fit(
        X_train_cnn,
        y_train_index,
        validation_data=(X_val_cnn, y_val_index),
        epochs=int(cfg.get("epochs", 200)),
        batch_size=int(cfg.get("batch_size", 64)),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=int(cfg.get("verbose", 0)),
    )
    return {"kind": "cnn_1d", "model": model, "scaler": scaler, "history": pd.DataFrame(history.history)}


def predict(model_info: dict[str, Any], X, classes: list[int], flood_class: int) -> tuple[np.ndarray, np.ndarray]:
    label_to_index, index_to_label = label_maps(classes)
    model = model_info["model"]
    scaler = model_info["scaler"]
    X_arr = np.asarray(X, dtype="float32")
    X_cnn = scaler.transform(X_arr).astype("float32").reshape(-1, X_arr.shape[1], 1)
    probabilities = model.predict(X_cnn, batch_size=4096, verbose=0)
    pred_index = np.argmax(probabilities, axis=1).astype("int32")
    labels = np.array([index_to_label[int(idx)] for idx in pred_index], dtype="int32")
    flood_col = label_to_index[flood_class]
    return labels, probabilities[:, flood_col].astype("float32")


__all__ = ["predict", "train"]
