#!/usr/bin/env python3
"""Build DenseNet prefix TFLite models for stage-level latency profiling."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf


IMG_SIZE = 64
NUM_CLASSES = 3


ENDPOINTS = [
    ("nested", "pool1", "00_stem_pool1"),
    ("nested", "conv2_block3_concat", "01_dense2_block03"),
    ("nested", "conv2_block6_concat", "02_dense2_block06"),
    ("nested", "pool2_pool", "03_transition2_pool"),
    ("nested", "conv3_block4_concat", "04_dense3_block04"),
    ("nested", "conv3_block8_concat", "05_dense3_block08"),
    ("nested", "conv3_block12_concat", "06_dense3_block12"),
    ("nested", "pool3_pool", "07_transition3_pool"),
    ("nested", "conv4_block6_concat", "08_dense4_block06"),
    ("nested", "conv4_block12_concat", "09_dense4_block12"),
    ("nested", "conv4_block18_concat", "10_dense4_block18"),
    ("nested", "conv4_block24_concat", "11_dense4_block24"),
    ("nested", "pool4_pool", "12_transition4_pool"),
    ("nested", "conv5_block4_concat", "13_dense5_block04"),
    ("nested", "conv5_block8_concat", "14_dense5_block08"),
    ("nested", "conv5_block12_concat", "15_dense5_block12"),
    ("nested", "conv5_block16_concat", "16_dense5_block16"),
    ("nested", "relu", "17_final_relu"),
    ("full", "global_average_pooling2d_2", "18_global_average_pool"),
    ("full", "dense_2", "19_softmax_head"),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


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
    for cls, files in files_by_class.items():
        idx = np.arange(len(files))
        rng.shuffle(idx)
        n_holdout = int(math.ceil(len(files) * 0.20))
        train_idx = idx[n_holdout:]
        for i in train_idx:
            split["train"].append((str(files[i]), int(cls)))
    rng.shuffle(split["train"])
    return split


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


def random_indices(n: int, count: int, seed: int) -> list[int]:
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    return [int(i) for i in idx[: min(count, n)]]


def representative_raw(samples: np.ndarray):
    def gen():
        for sample in samples:
            yield [sample[None, ...].astype(np.float32)]

    return gen


def make_prefix_model(full_model: tf.keras.Model, kind: str, layer_name: str, label: str) -> tf.keras.Model:
    if kind == "full":
        output = full_model.get_layer(layer_name).output
        return tf.keras.Model(full_model.input, output, name=f"prefix_{label}")

    dense_net = full_model.get_layer("densenet121")
    prefix_core = tf.keras.Model(
        dense_net.input,
        dense_net.get_layer(layer_name).output,
        name=f"densenet_core_{label}",
    )
    raw = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = tf.keras.applications.densenet.preprocess_input(raw)
    y = prefix_core(x)
    return tf.keras.Model(raw, y, name=f"prefix_{label}")


def convert_int8(model: tf.keras.Model, path: Path, x_rep: np.ndarray) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_raw(x_rep)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    path.write_bytes(converter.convert())


def inspect_tflite(path: Path) -> dict:
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    try:
        ops = Counter(op.get("op_name", "") for op in interpreter._get_ops_details())
    except Exception as exc:
        ops = Counter({"error": repr(exc)})
    return {
        "path": str(path),
        "size_mb": path.stat().st_size / 1024 / 1024,
        "input_shape": input_detail["shape"].tolist(),
        "input_dtype": str(input_detail["dtype"]),
        "input_quantization": input_detail.get("quantization", (0.0, 0)),
        "output_shape": output_detail["shape"].tolist(),
        "output_dtype": str(output_detail["dtype"]),
        "output_quantization": output_detail.get("quantization", (0.0, 0)),
        "op_counts": dict(ops),
        "delegate_count": int(ops.get("DELEGATE", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-keras", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--rep-count", type=int, default=120)
    parser.add_argument("--rep-trial-seed", type=int, default=31)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files_by_class = collect_files(args.data_dir)
    split = split_files(files_by_class, args.seed)
    x_train_raw, y_train = load_images(split["train"])
    rep_seed = args.seed * 1000 + args.rep_count * 10 + args.rep_trial_seed
    rep_idx = random_indices(len(x_train_raw), args.rep_count, rep_seed)
    x_rep = x_train_raw[rep_idx]

    full_model = tf.keras.models.load_model(str(args.base_keras), compile=False)
    dense_net = full_model.get_layer("densenet121")

    manifest = {
        "tensorflow": tf.__version__,
        "base_keras": str(args.base_keras),
        "data_dir": str(args.data_dir),
        "seed": args.seed,
        "rep_count": int(len(x_rep)),
        "rep_trial_seed": args.rep_trial_seed,
        "rep_class_counts": {
            str(cls): int(np.sum(y_train[rep_idx] == cls)) for cls in range(NUM_CLASSES)
        },
        "endpoints": [],
    }

    available_layers = {layer.name for layer in dense_net.layers} | {layer.name for layer in full_model.layers}
    for kind, layer_name, label in ENDPOINTS:
        if layer_name not in available_layers:
            raise ValueError(f"Missing endpoint layer {layer_name}; available sample={sorted(available_layers)[:20]}")
        print(f"[build] {label} <= {kind}:{layer_name}", flush=True)
        model = make_prefix_model(full_model, kind, layer_name, label)
        out_path = args.output_dir / f"{label}.full_int8_io.tflite"
        convert_int8(model, out_path, x_rep)
        info = inspect_tflite(out_path)
        info.update({"label": label, "kind": kind, "layer_name": layer_name})
        manifest["endpoints"].append(info)
        print(json.dumps(info, indent=2), flush=True)

    (args.output_dir / "prefix_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
