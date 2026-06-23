#!/usr/bin/env python3
"""Focused PTQ calibration sweep for recovering full-int8 RPS accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

import run_rps_qat as rps


def representative_raw(samples: np.ndarray):
    def gen():
        for sample in samples:
            yield [sample[None, ...].astype(np.float32)]

    return gen


def convert_full_int8_with_options(
    model,
    output_path: Path,
    x_rep: np.ndarray,
    new_quantizer: bool | None,
    float_io: bool,
) -> dict:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_raw(x_rep)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if new_quantizer is not None:
        converter._experimental_new_quantizer = bool(new_quantizer)
    if not float_io:
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    output_path.write_bytes(converter.convert())
    return {
        "path": str(output_path),
        "size_mb": output_path.stat().st_size / 1024 / 1024,
        "new_quantizer": new_quantizer,
        "float_io": float_io,
    }


def random_indices(n: int, count: int, seed: int) -> list[int]:
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    return [int(i) for i in idx[: min(count, n)]]


def balanced_indices(y: np.ndarray, count: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    by_class = {}
    for cls in range(rps.NUM_CLASSES):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        by_class[cls] = list(idx)
    selected = []
    cursor = {cls: 0 for cls in range(rps.NUM_CLASSES)}
    while len(selected) < count and any(cursor[cls] < len(by_class[cls]) for cls in by_class):
        for cls in range(rps.NUM_CLASSES):
            if len(selected) >= count:
                break
            pos = cursor[cls]
            if pos < len(by_class[cls]):
                selected.append(int(by_class[cls][pos]))
                cursor[cls] += 1
    return selected


def make_brightness_representatives(
    x_train_raw: np.ndarray,
    indices: list[int],
    count: int,
) -> np.ndarray:
    factors = [0.85, 1.0, 1.15]
    out = []
    for idx in indices:
        for factor in factors:
            out.append(np.clip(x_train_raw[idx] * factor, 0, 255).astype(np.float32))
            if len(out) >= count:
                return np.stack(out, axis=0)
    return np.stack(out, axis=0)


def class_counts(y_train: np.ndarray, indices: list[int]) -> dict[str, int]:
    return {str(cls): int(np.sum(y_train[indices] == cls)) for cls in range(rps.NUM_CLASSES)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-keras", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--counts", nargs="+", type=int, default=[60, 120, 180, 240, 300, 360])
    parser.add_argument("--trial-seeds", nargs="+", type=int, default=[7, 17, 31])
    args = parser.parse_args()

    rps.set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files_by_class = rps.collect_files(args.data_dir)
    split, split_counts = rps.split_files(files_by_class, args.seed)
    x_train_raw, y_train = rps.load_images(split["train"])
    x_val_raw, y_val = rps.load_images(split["val"])
    x_test_raw, y_test = rps.load_images(split["test"])

    model = rps.build_nested_source_model()
    model.load_weights(str(args.base_keras))

    variants = []
    for count in args.counts:
        count = min(count, len(x_train_raw))
        for trial_seed in args.trial_seeds:
            seed = args.seed * 1000 + count * 10 + trial_seed
            variants.append(
                {
                    "name": f"random_c{count}_s{trial_seed}",
                    "indices": random_indices(len(x_train_raw), count, seed),
                    "x_rep": None,
                    "mode": "random",
                    "new_quantizer": None,
                    "float_io": False,
                }
            )
        variants.append(
            {
                "name": f"balanced_c{count}",
                "indices": balanced_indices(y_train, count, args.seed + count),
                "x_rep": None,
                "mode": "balanced",
                "new_quantizer": None,
                "float_io": False,
            }
        )

    for count in [120, 180, 240, 300]:
        seed = args.seed * 1000 + count * 10 + 17
        base_indices = random_indices(len(x_train_raw), max(1, count // 3 + 2), seed)
        variants.append(
            {
                "name": f"brightness_mild_c{count}_s17",
                "indices": base_indices,
                "x_rep": make_brightness_representatives(x_train_raw, base_indices, count),
                "mode": "brightness_mild",
                "new_quantizer": None,
                "float_io": False,
            }
        )

    for count in [120, 240, 300]:
        seed = args.seed * 1000 + count * 10 + 17
        idx = random_indices(len(x_train_raw), count, seed)
        variants.append(
            {
                "name": f"old_quantizer_c{count}_s17",
                "indices": idx,
                "x_rep": None,
                "mode": "random_old_quantizer",
                "new_quantizer": False,
                "float_io": False,
            }
        )
        variants.append(
            {
                "name": f"float_io_c{count}_s17",
                "indices": idx,
                "x_rep": None,
                "mode": "random_float_io",
                "new_quantizer": None,
                "float_io": True,
            }
        )

    results = {
        "seed": args.seed,
        "split_counts": split_counts,
        "base_keras": str(args.base_keras),
        "variants": {},
    }
    best_name = None
    best_val = -1.0
    for variant in variants:
        name = variant["name"]
        x_rep = variant["x_rep"]
        if x_rep is None:
            x_rep = x_train_raw[variant["indices"]]
        suffix = "float_io" if variant["float_io"] else "full_int8_io"
        out_path = args.output_dir / f"03_aug_geometric_brightness10.ptq_{name}_{suffix}.tflite"
        info = convert_full_int8_with_options(
            model,
            out_path,
            x_rep,
            variant["new_quantizer"],
            variant["float_io"],
        )
        val_eval = rps.evaluate_tflite(out_path, x_val_raw, y_val)
        test_eval = rps.evaluate_tflite(out_path, x_test_raw, y_test)
        if val_eval["accuracy"] > best_val:
            best_val = float(val_eval["accuracy"])
            best_name = name
        results["variants"][name] = {
            "mode": variant["mode"],
            "representative_count": int(len(x_rep)),
            "representative_class_counts": class_counts(y_train, variant["indices"]),
            "tflite": info,
            "val_eval": val_eval,
            "test_eval": test_eval,
        }
        print(json.dumps({name: results["variants"][name]}, indent=2))

    results["best_by_val"] = best_name
    (args.output_dir / "ptq_focused_sweep.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
