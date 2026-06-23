#!/usr/bin/env python3
"""Create FP16/full-integer TFLite variants and evaluate test accuracy."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf


IMG_SIZE = 64
NUM_CLASSES = 3


def load_image(path: Path) -> np.ndarray:
    image_bytes = tf.io.read_file(str(path))
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE], method="bilinear")
    return image.numpy().astype(np.float32)


def representative_generator(data_root: Path, items: list[list]) -> Callable[[], object]:
    def gen():
        for rel_path, _label in items:
            sample = load_image(data_root / rel_path)
            yield [sample[None, ...]]

    return gen


def build_inference_model() -> tf.keras.Model:
    """Rebuild the training model without random augmentation.

    The saved .keras file was produced by an older Keras path layout that Pi's
    Keras 3 loader cannot deserialize. The augmentation layers are inference
    no-ops, so rebuilding the inference graph and loading the weights preserves
    the deployed model behavior.
    """
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = tf.keras.applications.densenet.preprocess_input(inputs)
    dense_net = tf.keras.applications.DenseNet121(
        weights=None,
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    x = dense_net(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="03_aug_geometric_brightness10")
    return model


def load_model_with_fallback(keras_path: Path) -> tuple[tf.keras.Model, str]:
    try:
        return tf.keras.models.load_model(keras_path, compile=False), "load_model"
    except Exception:
        model = build_inference_model()
        model.load_weights(keras_path)
        return model, "rebuilt_inference_model_load_weights"


def convert_fp16(model: tf.keras.Model, output_path: Path) -> dict:
    start = time.time()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    output_path.write_bytes(converter.convert())
    return {
        "path": str(output_path),
        "size_mb": output_path.stat().st_size / 1024 / 1024,
        "elapsed_sec": time.time() - start,
    }


def convert_full_integer(
    model: tf.keras.Model,
    output_path: Path,
    rep_fn: Callable[[], object],
    io_dtype: tf.dtypes.DType,
) -> dict:
    start = time.time()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_fn
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = io_dtype
    converter.inference_output_type = io_dtype
    output_path.write_bytes(converter.convert())
    return {
        "path": str(output_path),
        "size_mb": output_path.stat().st_size / 1024 / 1024,
        "elapsed_sec": time.time() - start,
        "io_dtype": io_dtype.name,
    }


def prepare_input(sample: np.ndarray, detail: dict) -> np.ndarray:
    dtype = detail["dtype"]
    if dtype == np.float32:
        return sample[None, ...].astype(np.float32)
    if dtype == np.float16:
        return sample[None, ...].astype(np.float16)

    scale, zero_point = detail.get("quantization", (0.0, 0))
    if not scale:
        return sample[None, ...].astype(dtype)
    quantized = np.round(sample / scale + zero_point)
    if dtype == np.int8:
        quantized = np.clip(quantized, -128, 127)
    elif dtype == np.uint8:
        quantized = np.clip(quantized, 0, 255)
    return quantized.astype(dtype)[None, ...]


def decode_output(output: np.ndarray, detail: dict) -> np.ndarray:
    output = np.squeeze(output)
    dtype = detail["dtype"]
    if dtype in (np.int8, np.uint8):
        scale, zero_point = detail.get("quantization", (0.0, 0))
        if scale:
            output = (output.astype(np.float32) - zero_point) * scale
    return output.astype(np.float32)


def evaluate_tflite(model_path: Path, data_root: Path, test_items: list[list]) -> dict:
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    correct = 0
    confusion = np.zeros((3, 3), dtype=np.int64)
    start = time.time()
    for rel_path, label in test_items:
        sample = load_image(data_root / rel_path)
        interpreter.set_tensor(input_detail["index"], prepare_input(sample, input_detail))
        interpreter.invoke()
        output = decode_output(interpreter.get_tensor(output_detail["index"]), output_detail)
        pred = int(np.argmax(output))
        label = int(label)
        correct += int(pred == label)
        confusion[label, pred] += 1

    return {
        "path": str(model_path),
        "accuracy": correct / len(test_items),
        "correct": correct,
        "total": len(test_items),
        "confusion_matrix": confusion.tolist(),
        "elapsed_sec": time.time() - start,
        "input_dtype": str(input_detail["dtype"]),
        "output_dtype": str(output_detail["dtype"]),
        "input_quantization": input_detail.get("quantization", (0.0, 0)),
        "output_quantization": output_detail.get("quantization", (0.0, 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keras-model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-name", default="03_aug_geometric_brightness10")
    parser.add_argument("--existing-tflite", nargs="*", default=[])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rep_items = manifest["representative"]
    test_items = manifest["test"]

    model, model_load_method = load_model_with_fallback(args.keras_model)
    rep_fn = representative_generator(args.data_root, rep_items)

    conversions: dict[str, dict] = {}
    variants: dict[str, Path] = {}
    for existing in args.existing_tflite:
        path = Path(existing)
        variants[path.stem] = path

    try:
        fp16_path = args.output_dir / f"{args.base_name}.fp16_weight.tflite"
        conversions["fp16_weight"] = convert_fp16(model, fp16_path)
        variants["fp16_weight"] = fp16_path
    except Exception as exc:
        conversions["fp16_weight"] = {"error": repr(exc)}

    try:
        int8_path = args.output_dir / f"{args.base_name}.full_int8_io.tflite"
        conversions["full_int8_io"] = convert_full_integer(model, int8_path, rep_fn, tf.int8)
        variants["full_int8_io"] = int8_path
    except Exception as exc:
        conversions["full_int8_io"] = {"error": repr(exc)}

    try:
        uint8_path = args.output_dir / f"{args.base_name}.full_uint8_io.tflite"
        conversions["full_uint8_io"] = convert_full_integer(
            model, uint8_path, representative_generator(args.data_root, rep_items), tf.uint8
        )
        variants["full_uint8_io"] = uint8_path
    except Exception as exc:
        conversions["full_uint8_io"] = {"error": repr(exc)}

    evaluations = {}
    for name, path in variants.items():
        try:
            evaluations[name] = evaluate_tflite(path, args.data_root, test_items)
        except Exception as exc:
            evaluations[name] = {"path": str(path), "error": repr(exc)}

    result = {
        "tensorflow": tf.__version__,
        "keras_model": str(args.keras_model),
        "model_load_method": model_load_method,
        "manifest": str(args.manifest),
        "representative_count": len(rep_items),
        "test_count": len(test_items),
        "conversions": conversions,
        "evaluations": evaluations,
    }
    output_json = args.output_dir / "quantization_eval.json"
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
