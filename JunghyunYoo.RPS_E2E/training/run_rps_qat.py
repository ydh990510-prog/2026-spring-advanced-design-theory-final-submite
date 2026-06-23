#!/usr/bin/env python3
"""Quantization-aware fine-tuning for the RPS DenseNet121 model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
import tf_keras as keras


IMG_SIZE = 64
NUM_CLASSES = 3
CLASS_NAMES = ["scissors", "rock", "paper"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def collect_files(data_dir: Path) -> dict[str, list[Path]]:
    files_by_class = {}
    for cls_idx in range(NUM_CLASSES):
        class_dir = data_dir / str(cls_idx)
        files = sorted(
            [
                p
                for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ]
        )
        if not files:
            raise FileNotFoundError(f"No files found in {class_dir}")
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


def preprocess_np(x: np.ndarray) -> np.ndarray:
    return keras.applications.densenet.preprocess_input(x.copy()).astype(np.float32)


def build_nested_source_model() -> keras.Model:
    dense_net = keras.applications.DenseNet121(
        weights=None,
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = keras.applications.densenet.preprocess_input(inputs)
    x = dense_net(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d_2")(x)
    x = keras.layers.Dropout(0.2, name="dropout_2")(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax", name="dense_2")(x)
    return keras.Model(inputs, outputs, name="03_aug_geometric_brightness10")


def build_flat_core_model() -> keras.Model:
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="preprocessed_image")
    base = keras.applications.DenseNet121(
        weights=None,
        include_top=False,
        input_tensor=inputs,
    )
    x = base.output
    x = keras.layers.GlobalAveragePooling2D(name="qat_global_average_pooling")(x)
    x = keras.layers.Dropout(0.2, name="qat_dropout")(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax", name="qat_predictions")(x)
    return keras.Model(inputs, outputs, name="rps_densenet121_core")


def transfer_weights(base_keras: Path, target: keras.Model) -> dict:
    source = build_nested_source_model()
    source.load_weights(str(base_keras))
    dense_source = source.get_layer("densenet121")

    transferred = []
    skipped = []
    for layer in target.layers:
        if not layer.weights:
            continue
        source_layer = None
        if layer.name in [l.name for l in dense_source.layers]:
            source_layer = dense_source.get_layer(layer.name)
        elif layer.name == "qat_predictions":
            source_layer = source.get_layer("dense_2")
        if source_layer is None:
            skipped.append(layer.name)
            continue
        try:
            source_weights = source_layer.get_weights()
            if [w.shape for w in source_weights] == [w.shape for w in layer.get_weights()]:
                layer.set_weights(source_weights)
                transferred.append(layer.name)
            else:
                skipped.append(layer.name)
        except Exception:
            skipped.append(layer.name)
    return {"transferred": transferred, "skipped": skipped}


def representative_raw(x_train: np.ndarray, limit: int):
    samples = x_train[: min(limit, len(x_train))]

    def gen():
        for sample in samples:
            yield [sample[None, ...].astype(np.float32)]

    return gen


def convert_full_int8(model: keras.Model, output_path: Path, x_train_raw: np.ndarray, rep_count: int) -> dict:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_raw(x_train_raw, rep_count)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output_path.write_bytes(converter.convert())
    return {
        "path": str(output_path),
        "size_mb": output_path.stat().st_size / 1024 / 1024,
    }


def prepare_tflite_input(sample: np.ndarray, input_detail: dict) -> np.ndarray:
    dtype = input_detail["dtype"]
    if dtype == np.float32:
        return sample[None, ...].astype(np.float32)
    scale, zero_point = input_detail.get("quantization", (0.0, 0))
    values = sample.astype(np.float32)
    if scale:
        values = np.round(values / scale + zero_point)
    if dtype == np.int8:
        values = np.clip(values, -128, 127)
    elif dtype == np.uint8:
        values = np.clip(values, 0, 255)
    return values.astype(dtype)[None, ...]


def decode_tflite_output(output: np.ndarray, output_detail: dict) -> np.ndarray:
    output = np.squeeze(output)
    if output_detail["dtype"] in (np.int8, np.uint8):
        scale, zero_point = output_detail.get("quantization", (0.0, 0))
        if scale:
            output = (output.astype(np.float32) - zero_point) * scale
    return output.astype(np.float32)


def evaluate_tflite(path: Path, x_test_raw: np.ndarray, y_test: np.ndarray) -> dict:
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    correct = 0
    for sample, label in zip(x_test_raw, y_test):
        interpreter.set_tensor(input_detail["index"], prepare_tflite_input(sample, input_detail))
        interpreter.invoke()
        scores = decode_tflite_output(interpreter.get_tensor(output_detail["index"]), output_detail)
        pred = int(np.argmax(scores))
        label = int(label)
        correct += int(pred == label)
        confusion[label, pred] += 1
    return {
        "accuracy": correct / len(y_test),
        "correct": int(correct),
        "total": int(len(y_test)),
        "confusion_matrix": confusion.tolist(),
        "input_dtype": str(input_detail["dtype"]),
        "input_quantization": input_detail.get("quantization", (0.0, 0)),
        "output_dtype": str(output_detail["dtype"]),
        "output_quantization": output_detail.get("quantization", (0.0, 0)),
    }


def write_history(history: keras.callbacks.History, path: Path) -> None:
    keys = list(history.history.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for i in range(len(history.history[keys[0]])):
            writer.writerow([i + 1] + [history.history[key][i] for key in keys])


class BestWeights(keras.callbacks.Callback):
    def __init__(self, monitor: str = "val_loss") -> None:
        super().__init__()
        self.monitor = monitor
        self.best_value = np.inf
        self.best_epoch = 0
        self.best_weights = None

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        value = logs.get(self.monitor)
        if value is None:
            return
        value = float(value)
        if value < self.best_value:
            self.best_value = value
            self.best_epoch = epoch + 1
            self.best_weights = self.model.get_weights()


class TFLiteValCheckpoint(keras.callbacks.Callback):
    def __init__(
        self,
        output_dir: Path,
        x_train_raw: np.ndarray,
        x_val_raw: np.ndarray,
        y_val: np.ndarray,
        rep_count: int,
        keep_epoch_tflite: bool = False,
    ) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.x_train_raw = x_train_raw
        self.x_val_raw = x_val_raw
        self.y_val = y_val
        self.rep_count = rep_count
        self.keep_epoch_tflite = keep_epoch_tflite
        self.epoch_dir = output_dir / "epoch_tflite"
        self.csv_path = output_dir / "epoch_tflite_val.csv"
        self.best_tflite_path = output_dir / "best_by_tflite_val.tflite"
        self.best_weights = None
        self.best_epoch = 0
        self.best_accuracy = -1.0
        self.best_correct = -1
        self.best_val_loss = np.inf
        self.rows: list[dict] = []

    def on_train_begin(self, logs: dict | None = None) -> None:
        self.epoch_dir.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "keras_loss",
                    "keras_accuracy",
                    "keras_val_loss",
                    "keras_val_accuracy",
                    "tflite_val_accuracy",
                    "tflite_val_correct",
                    "tflite_val_total",
                    "tflite_size_mb",
                    "is_best",
                    "elapsed_sec",
                ]
            )

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        epoch_num = epoch + 1
        t0 = time.time()
        epoch_path = self.epoch_dir / f"epoch_{epoch_num:03d}.full_int8.tflite"
        deploy_model = make_deploy_model(self.model)
        tflite_info = convert_full_int8(deploy_model, epoch_path, self.x_train_raw, self.rep_count)
        eval_info = evaluate_tflite(epoch_path, self.x_val_raw, self.y_val)
        elapsed_sec = time.time() - t0

        val_loss = float(logs.get("val_loss", np.inf))
        accuracy = float(eval_info["accuracy"])
        correct = int(eval_info["correct"])
        better = (
            accuracy > self.best_accuracy + 1e-12
            or (
                abs(accuracy - self.best_accuracy) <= 1e-12
                and (correct > self.best_correct or val_loss < self.best_val_loss)
            )
        )
        if better:
            self.best_accuracy = accuracy
            self.best_correct = correct
            self.best_val_loss = val_loss
            self.best_epoch = epoch_num
            self.best_weights = self.model.get_weights()
            shutil.copyfile(epoch_path, self.best_tflite_path)

        row = {
            "epoch": epoch_num,
            "keras_loss": float(logs.get("loss", np.nan)),
            "keras_accuracy": float(logs.get("accuracy", np.nan)),
            "keras_val_loss": val_loss,
            "keras_val_accuracy": float(logs.get("val_accuracy", np.nan)),
            "tflite_val_accuracy": accuracy,
            "tflite_val_correct": correct,
            "tflite_val_total": int(eval_info["total"]),
            "tflite_size_mb": float(tflite_info["size_mb"]),
            "is_best": bool(better),
            "elapsed_sec": elapsed_sec,
        }
        self.rows.append(row)
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    row["epoch"],
                    row["keras_loss"],
                    row["keras_accuracy"],
                    row["keras_val_loss"],
                    row["keras_val_accuracy"],
                    row["tflite_val_accuracy"],
                    row["tflite_val_correct"],
                    row["tflite_val_total"],
                    row["tflite_size_mb"],
                    int(row["is_best"]),
                    row["elapsed_sec"],
                ]
            )

        if not self.keep_epoch_tflite and not better:
            epoch_path.unlink(missing_ok=True)
        print(
            "[tflite-val] "
            f"epoch={epoch_num} acc={accuracy:.6f} correct={correct}/{eval_info['total']} "
            f"best_epoch={self.best_epoch} best_acc={self.best_accuracy:.6f} "
            f"elapsed={elapsed_sec:.2f}s"
        )


def make_deploy_model(qat_core: keras.Model) -> keras.Model:
    raw = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = keras.applications.densenet.preprocess_input(raw)
    out = qat_core(x)
    return keras.Model(raw, out, name="rps_qat_raw_input_deploy")


def strip_qat_weights_to_core(qat_model: keras.Model) -> tuple[keras.Model, dict]:
    stripped = build_flat_core_model()
    qat_layers = {layer.name: layer for layer in qat_model.layers}
    inner_layers = {}
    wrappers_by_inner_name = {}
    for layer in qat_model.layers:
        inner = getattr(layer, "layer", None)
        if inner is not None:
            inner_layers[inner.name] = inner
            wrappers_by_inner_name[inner.name] = layer

    transferred = []
    skipped = []

    def select_matching_weights(source_layer, target_weights):
        source_weights = source_layer.get_weights()
        selected = []
        used = set()
        for target_weight in target_weights:
            match_idx = None
            for idx, source_weight in enumerate(source_weights):
                if idx in used:
                    continue
                if source_weight.shape == target_weight.shape:
                    match_idx = idx
                    break
            if match_idx is None:
                return None, [list(w.shape) for w in source_weights]
            used.add(match_idx)
            selected.append(source_weights[match_idx])
        return selected, [list(w.shape) for w in source_weights]

    for layer in stripped.layers:
        if not layer.weights:
            continue
        source = inner_layers.get(layer.name) or qat_layers.get(layer.name)
        if source is None:
            skipped.append(layer.name)
            continue
        target_weights = layer.get_weights()
        source_weights = source.get_weights()
        if [w.shape for w in source_weights] == [w.shape for w in target_weights]:
            layer.set_weights(source_weights)
            transferred.append(layer.name)
            continue

        wrapper = wrappers_by_inner_name.get(layer.name)
        if wrapper is not None:
            matching_weights, wrapper_shapes = select_matching_weights(wrapper, target_weights)
            if matching_weights is not None:
                layer.set_weights(matching_weights)
                transferred.append({"layer": layer.name, "source": "quantize_wrapper_shape_match"})
                continue

        skipped.append(
            {
                "layer": layer.name,
                "source_shapes": [list(w.shape) for w in source_weights],
                "wrapper_shapes": wrapper_shapes if wrapper is not None else None,
                "target_shapes": [list(w.shape) for w in target_weights],
            }
        )
    return stripped, {"transferred_count": len(transferred), "transferred": transferred, "skipped": skipped}


def make_qat_model(core: keras.Model, tfmot_module, qat_scope: str = "all") -> tuple[keras.Model, dict]:
    """Apply QAT to supported high-cost layers.

    DenseNet contains BatchNormalization and merge layers that the default
    TFMOT registry does not quantize directly. Annotating Conv2D/Dense keeps
    the expensive layers quantization-aware while leaving structural layers
    unchanged.
    """

    annotated_names = []

    def in_scope(layer) -> bool:
        if qat_scope == "all":
            return isinstance(layer, (keras.layers.Conv2D, keras.layers.Dense))
        if qat_scope == "head":
            return isinstance(layer, keras.layers.Dense) and layer.name == "qat_predictions"
        if qat_scope == "conv5_head":
            return (
                isinstance(layer, keras.layers.Dense)
                and layer.name == "qat_predictions"
            ) or (
                isinstance(layer, keras.layers.Conv2D)
                and (layer.name.startswith("conv5_") or layer.name == "conv5_block16_2_conv")
            )
        raise ValueError(f"Unsupported qat_scope: {qat_scope}")

    def annotate_if_supported(layer):
        if in_scope(layer):
            annotated_names.append(layer.name)
            return tfmot_module.quantization.keras.quantize_annotate_layer(layer)
        return layer

    annotated_model = keras.models.clone_model(core, clone_function=annotate_if_supported)
    annotated_model.set_weights(core.get_weights())
    with tfmot_module.quantization.keras.quantize_scope():
        qat_model = tfmot_module.quantization.keras.quantize_apply(annotated_model)
    return qat_model, {
        "strategy": f"partial_qat_{qat_scope}",
        "annotated_count": len(annotated_names),
        "annotated_names": annotated_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-keras", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--rep-count", type=int, default=300)
    parser.add_argument("--per-epoch-tflite-val", action="store_true")
    parser.add_argument("--keep-epoch-tflite", action="store_true")
    parser.add_argument("--reduce-lr-on-plateau", action="store_true")
    parser.add_argument("--reduce-lr-factor", type=float, default=0.1)
    parser.add_argument("--reduce-lr-patience", type=int, default=5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--strip-qat-to-ptq", action="store_true")
    parser.add_argument("--qat-scope", choices=["all", "head", "conv5_head"], default="all")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import tensorflow_model_optimization as tfmot

    files_by_class = collect_files(args.data_dir)
    split, split_counts = split_files(files_by_class, args.seed)
    x_train_raw, y_train = load_images(split["train"])
    x_val_raw, y_val = load_images(split["val"])
    x_test_raw, y_test = load_images(split["test"])

    x_train = preprocess_np(x_train_raw)
    x_val = preprocess_np(x_val_raw)
    x_test = preprocess_np(x_test_raw)

    core = build_flat_core_model()
    transfer_info = transfer_weights(args.base_keras, core)
    core.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    float_eval = core.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=0)

    qat_model, qat_info = make_qat_model(core, tfmot, args.qat_scope)
    qat_model.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    initial_qat_eval = qat_model.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=0)

    best_weights = BestWeights(monitor="val_loss")
    tflite_val_checkpoint = None
    callbacks = [
        best_weights,
        keras.callbacks.CSVLogger(str(args.output_dir / "qat_training_log.csv")),
    ]
    if args.reduce_lr_on_plateau:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=args.reduce_lr_factor,
                patience=args.reduce_lr_patience,
                min_lr=args.min_learning_rate,
                verbose=1,
            )
        )
    if args.per_epoch_tflite_val:
        tflite_val_checkpoint = TFLiteValCheckpoint(
            output_dir=args.output_dir,
            x_train_raw=x_train_raw,
            x_val_raw=x_val_raw,
            y_val=y_val,
            rep_count=args.rep_count,
            keep_epoch_tflite=args.keep_epoch_tflite,
        )
        callbacks.append(tflite_val_checkpoint)

    start = time.time()
    history = qat_model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed_sec = time.time() - start
    write_history(history, args.output_dir / "qat_history.csv")

    selection_method = "keras_val_loss"
    final_tflite_path = args.output_dir / "03_aug_geometric_brightness10.qat_full_int8_io.tflite"
    if tflite_val_checkpoint is not None and tflite_val_checkpoint.best_weights is not None:
        qat_model.set_weights(tflite_val_checkpoint.best_weights)
        selection_method = "tflite_val_accuracy"
    elif best_weights.best_weights is not None:
        qat_model.set_weights(best_weights.best_weights)
    qat_eval = qat_model.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=0)

    if (
        tflite_val_checkpoint is not None
        and tflite_val_checkpoint.best_tflite_path.exists()
    ):
        shutil.copyfile(tflite_val_checkpoint.best_tflite_path, final_tflite_path)
        tflite_info = {
            "path": str(final_tflite_path),
            "size_mb": final_tflite_path.stat().st_size / 1024 / 1024,
            "source": str(tflite_val_checkpoint.best_tflite_path),
        }
    else:
        deploy_model = make_deploy_model(qat_model)
        tflite_info = convert_full_int8(
            deploy_model,
            final_tflite_path,
            x_train_raw,
            args.rep_count,
        )
    tflite_eval = evaluate_tflite(Path(tflite_info["path"]), x_test_raw, y_test)

    metrics = {
        "tensorflow": tf.__version__,
        "tfmot": tfmot.__version__,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "rep_count": args.rep_count,
        "per_epoch_tflite_val": args.per_epoch_tflite_val,
        "reduce_lr_on_plateau": args.reduce_lr_on_plateau,
        "reduce_lr_factor": args.reduce_lr_factor,
        "reduce_lr_patience": args.reduce_lr_patience,
        "min_learning_rate": args.min_learning_rate,
        "strip_qat_to_ptq": args.strip_qat_to_ptq,
        "qat_scope": args.qat_scope,
        "split_counts": split_counts,
        "transfer": {
            "transferred_count": len(transfer_info["transferred"]),
            "skipped_count": len(transfer_info["skipped"]),
            "skipped": transfer_info["skipped"],
        },
        "qat_annotation": qat_info,
        "float_core_test_loss": float(float_eval[0]),
        "float_core_test_accuracy": float(float_eval[1]),
        "initial_qat_test_loss": float(initial_qat_eval[0]),
        "initial_qat_test_accuracy": float(initial_qat_eval[1]),
        "elapsed_sec": elapsed_sec,
        "best_epoch_by_val_loss": int(np.argmin(history.history["val_loss"]) + 1),
        "best_epoch_by_callback": int(best_weights.best_epoch),
        "selection_method": selection_method,
        "best_val_loss_history": float(np.min(history.history["val_loss"])),
        "best_val_accuracy_history": float(np.max(history.history["val_accuracy"])),
        "qat_test_loss": float(qat_eval[0]),
        "qat_test_accuracy": float(qat_eval[1]),
        "qat_tflite": tflite_info,
        "qat_full_int8_tflite_eval": tflite_eval,
    }
    if tflite_val_checkpoint is not None:
        metrics["tflite_val_selection"] = {
            "best_epoch": int(tflite_val_checkpoint.best_epoch),
            "best_accuracy": float(tflite_val_checkpoint.best_accuracy),
            "best_correct": int(tflite_val_checkpoint.best_correct),
            "best_val_loss": float(tflite_val_checkpoint.best_val_loss),
            "csv": str(tflite_val_checkpoint.csv_path),
            "rows": tflite_val_checkpoint.rows,
        }
    if args.strip_qat_to_ptq:
        stripped_core, strip_info = strip_qat_weights_to_core(qat_model)
        stripped_core.save_weights(str(args.output_dir / "qat_selected_stripped_core.weights.h5"))
        stripped_deploy = make_deploy_model(stripped_core)
        stripped_path = args.output_dir / "03_aug_geometric_brightness10.qat_stripped_ptq_full_int8_io.tflite"
        stripped_tflite_info = convert_full_int8(
            stripped_deploy,
            stripped_path,
            x_train_raw,
            args.rep_count,
        )
        metrics["qat_stripped_ptq"] = {
            "weight_transfer": strip_info,
            "tflite": stripped_tflite_info,
            "val_eval": evaluate_tflite(stripped_path, x_val_raw, y_val),
            "test_eval": evaluate_tflite(stripped_path, x_test_raw, y_test),
        }
    (args.output_dir / "qat_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
