#!/usr/bin/env python3
"""PTQ representative calibration sweep for the RPS DenseNet model."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

import run_rps_qat as rps


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


def shuffled_indices(n: int, count: int, seed: int) -> list[int]:
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    return [int(i) for i in idx[: min(count, n)]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-keras", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--rep-counts", nargs="+", type=int, default=[300, 600, 1000, 2172])
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
    for count in args.rep_counts:
        count = min(count, len(x_train_raw))
        variants.append((f"rep{count}_shuffled", shuffled_indices(len(x_train_raw), count, args.seed + count)))
        variants.append((f"rep{count}_balanced", balanced_indices(y_train, count, args.seed + count)))

    results = {
        "seed": args.seed,
        "split_counts": split_counts,
        "base_keras": str(args.base_keras),
        "variants": {},
    }
    for name, indices in variants:
        x_rep = x_train_raw[indices]
        out_path = args.output_dir / f"03_aug_geometric_brightness10.ptq_{name}_full_int8_io.tflite"
        info = rps.convert_full_int8(model, out_path, x_rep, len(x_rep))
        results["variants"][name] = {
            "representative_count": len(indices),
            "representative_class_counts": {
                str(cls): int(np.sum(y_train[indices] == cls)) for cls in range(rps.NUM_CLASSES)
            },
            "tflite": info,
            "val_eval": rps.evaluate_tflite(out_path, x_val_raw, y_val),
            "test_eval": rps.evaluate_tflite(out_path, x_test_raw, y_test),
        }
        print(json.dumps({name: results["variants"][name]}, indent=2))

    (args.output_dir / "ptq_calibration_sweep.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
