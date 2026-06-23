#!/usr/bin/env python3
"""Strong augmentation fine-tuning for the RPS DenseNet121 base model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf


IMG_SIZE = 64
NUM_CLASSES = 3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def collect_files(data_dir: Path) -> dict[str, list[Path]]:
    files_by_class = {}
    for cls_idx in range(NUM_CLASSES):
        class_dir = data_dir / str(cls_idx)
        files = sorted(
            p
            for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if not files:
            raise FileNotFoundError(f"No image files found in {class_dir}")
        files_by_class[str(cls_idx)] = files
    return files_by_class


def split_files(files_by_class: dict[str, list[Path]], seed: int):
    rng = np.random.default_rng(seed)
    split = {"train": [], "val": [], "test": []}
    split_counts = {}
    for cls, files in files_by_class.items():
        idx = np.arange(len(files))
        rng.shuffle(idx)
        n_holdout = int(math.ceil(len(files) * 0.20))
        holdout_idx = idx[:n_holdout]
        train_idx = idx[n_holdout:]
        n_val = n_holdout // 2
        val_idx = holdout_idx[:n_val]
        test_idx = holdout_idx[n_val:]
        for part, part_idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
            for i in part_idx:
                split[part].append((str(files[i]), int(cls)))
        split_counts[cls] = {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        }
    for part in split:
        rng.shuffle(split[part])
    return split, split_counts


def load_images(items: list[tuple[str, int]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.empty((len(items), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    y = np.empty((len(items),), dtype=np.int64)
    for i, (path, label) in enumerate(items):
        image_bytes = tf.io.read_file(path)
        image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE], method="bilinear")
        x[i] = image.numpy().astype(np.float32)
        y[i] = label
    return x, y


def make_augmentation(profile: str) -> tf.keras.Sequential:
    if profile == "strong_flip_light25":
        layers = [
            tf.keras.layers.RandomFlip("horizontal", name="aug_flip"),
            tf.keras.layers.RandomRotation(0.12, fill_mode="nearest", name="aug_rotation"),
            tf.keras.layers.RandomTranslation(0.12, 0.12, fill_mode="nearest", name="aug_translation"),
            tf.keras.layers.RandomZoom(0.15, fill_mode="nearest", name="aug_zoom"),
            tf.keras.layers.RandomContrast(0.25, name="aug_contrast"),
            tf.keras.layers.RandomBrightness(0.25, value_range=(0.0, 255.0), name="aug_brightness"),
            tf.keras.layers.GaussianNoise(3.0, name="aug_noise"),
        ]
    elif profile == "medium_flip_light20":
        layers = [
            tf.keras.layers.RandomFlip("horizontal", name="aug_flip"),
            tf.keras.layers.RandomRotation(0.08, fill_mode="nearest", name="aug_rotation"),
            tf.keras.layers.RandomTranslation(0.08, 0.08, fill_mode="nearest", name="aug_translation"),
            tf.keras.layers.RandomZoom(0.10, fill_mode="nearest", name="aug_zoom"),
            tf.keras.layers.RandomContrast(0.20, name="aug_contrast"),
            tf.keras.layers.RandomBrightness(0.20, value_range=(0.0, 255.0), name="aug_brightness"),
            tf.keras.layers.GaussianNoise(2.0, name="aug_noise"),
        ]
    else:
        raise ValueError(f"Unknown augmentation profile: {profile}")
    return tf.keras.Sequential(layers, name=f"augmentation_{profile}")


def build_deploy_model(name: str) -> tf.keras.Model:
    dense_net = tf.keras.applications.DenseNet121(
        weights=None,
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = tf.keras.applications.densenet.preprocess_input(inputs)
    x = dense_net(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d_2")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="dense_2")(x)
    return tf.keras.Model(inputs, outputs, name=name)


def build_train_model(profile: str, learning_rate: float) -> tf.keras.Model:
    dense_net = tf.keras.applications.DenseNet121(
        weights=None,
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = make_augmentation(profile)(inputs)
    x = tf.keras.applications.densenet.preprocess_input(x)
    x = dense_net(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d_2")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="dense_2")(x)
    model = tf.keras.Model(inputs, outputs, name=f"rps_densenet121_{profile}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def transfer_from_saved_model(base_keras: Path, target: tf.keras.Model) -> dict:
    source = tf.keras.models.load_model(str(base_keras), compile=False)
    source_dense = source.get_layer("densenet121")
    target_dense = target.get_layer("densenet121")
    target_dense.set_weights(source_dense.get_weights())

    source_head = source.get_layer("dense_2")
    target_head = target.get_layer("dense_2")
    target_head.set_weights(source_head.get_weights())
    return {
        "source": str(base_keras),
        "transferred": ["densenet121", "dense_2"],
    }


def copy_train_weights_to_deploy(train_model: tf.keras.Model, deploy_model: tf.keras.Model) -> None:
    deploy_model.get_layer("densenet121").set_weights(train_model.get_layer("densenet121").get_weights())
    deploy_model.get_layer("dense_2").set_weights(train_model.get_layer("dense_2").get_weights())


class BestByValAccuracyLoss(tf.keras.callbacks.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.best_val_accuracy = -1.0
        self.best_val_loss = np.inf
        self.best_epoch = 0
        self.best_weights = None

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        val_accuracy = float(logs.get("val_accuracy", -1.0))
        val_loss = float(logs.get("val_loss", np.inf))
        better = (
            val_accuracy > self.best_val_accuracy + 1e-12
            or (
                abs(val_accuracy - self.best_val_accuracy) <= 1e-12
                and val_loss < self.best_val_loss
            )
        )
        if better:
            self.best_val_accuracy = val_accuracy
            self.best_val_loss = val_loss
            self.best_epoch = epoch + 1
            self.best_weights = self.model.get_weights()


def write_history(history: tf.keras.callbacks.History, path: Path) -> None:
    keys = list(history.history.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for i in range(len(history.history[keys[0]])):
            writer.writerow([i + 1] + [history.history[key][i] for key in keys])


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        matrix[int(truth), int(pred)] += 1
    return matrix.tolist()


def evaluate_keras(model: tf.keras.Model, x: np.ndarray, y: np.ndarray, batch_size: int) -> dict:
    loss, accuracy = model.evaluate(x, y, batch_size=batch_size, verbose=0)
    pred = np.argmax(model.predict(x, batch_size=batch_size, verbose=0), axis=1)
    correct = int(np.sum(pred == y))
    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "correct": correct,
        "total": int(len(y)),
        "confusion_matrix": confusion_matrix(y, pred),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-keras", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=["medium_flip_light20", "strong_flip_light25"], default="strong_flip_light25")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files_by_class = collect_files(args.data_dir)
    split, split_counts = split_files(files_by_class, args.seed)
    x_train, y_train = load_images(split["train"])
    x_val, y_val = load_images(split["val"])
    x_test, y_test = load_images(split["test"])

    train_model = build_train_model(args.profile, args.learning_rate)
    transfer_info = transfer_from_saved_model(args.base_keras, train_model)
    initial_test = evaluate_keras(train_model, x_test, y_test, args.batch_size)

    best = BestByValAccuracyLoss()
    callbacks = [
        best,
        tf.keras.callbacks.CSVLogger(str(args.output_dir / "strong_aug_training_log.csv")),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=False,
            verbose=1,
        ),
    ]
    start = time.time()
    history = train_model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed_sec = time.time() - start
    write_history(history, args.output_dir / "strong_aug_history.csv")

    if best.best_weights is not None:
        train_model.set_weights(best.best_weights)

    deploy_model = build_deploy_model("03_aug_geometric_brightness10")
    copy_train_weights_to_deploy(train_model, deploy_model)
    deploy_model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    train_eval = evaluate_keras(deploy_model, x_train, y_train, args.batch_size)
    val_eval = evaluate_keras(deploy_model, x_val, y_val, args.batch_size)
    test_eval = evaluate_keras(deploy_model, x_test, y_test, args.batch_size)

    deploy_path = args.output_dir / f"04_aug_{args.profile}.deploy.best.keras"
    deploy_model.save(deploy_path)
    train_path = args.output_dir / f"04_aug_{args.profile}.train_with_aug.best.keras"
    train_model.save(train_path)
    with (args.output_dir / "model_summary.txt").open("w", encoding="utf-8") as f:
        deploy_model.summary(print_fn=lambda line: f.write(line + "\n"))

    metrics = {
        "tensorflow": tf.__version__,
        "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "seed": args.seed,
        "profile": args.profile,
        "epochs_requested": args.epochs,
        "epochs_ran": len(history.history.get("loss", [])),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "elapsed_sec": elapsed_sec,
        "split_counts": split_counts,
        "base_keras": str(args.base_keras),
        "transfer": transfer_info,
        "selection": {
            "method": "max_val_accuracy_tie_min_val_loss",
            "best_epoch": int(best.best_epoch),
            "best_val_accuracy": float(best.best_val_accuracy),
            "best_val_loss": float(best.best_val_loss),
        },
        "initial_test_eval": initial_test,
        "selected_deploy_keras": str(deploy_path),
        "selected_train_aug_keras": str(train_path),
        "train_eval": train_eval,
        "val_eval": val_eval,
        "test_eval": test_eval,
    }
    (args.output_dir / "strong_aug_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
