#!/usr/bin/env python3
"""Pruning experiments for RPS DenseNet with TFLite/XNNPACK sparse path."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
import tf_keras as keras

import run_rps_qat as rps


def build_prune_compatible_core() -> keras.Model:
    inputs = keras.Input(shape=(rps.IMG_SIZE, rps.IMG_SIZE, 3), name="preprocessed_image")
    base = keras.applications.DenseNet121(
        weights=None,
        include_top=False,
        input_tensor=inputs,
    )
    x = base.output
    x = keras.layers.GlobalAveragePooling2D(keepdims=True, name="prune_global_average_pooling")(x)
    x = keras.layers.Flatten(name="prune_flatten")(x)
    x = keras.layers.Dropout(0.2, name="prune_dropout")(x)
    outputs = keras.layers.Dense(rps.NUM_CLASSES, activation="softmax", name="prune_predictions")(x)
    return keras.Model(inputs, outputs, name="rps_densenet121_prune_core")


def transfer_base_weights(base_keras: Path, target: keras.Model) -> dict:
    source = rps.build_nested_source_model()
    source.load_weights(str(base_keras))
    dense_source = source.get_layer("densenet121")
    dense_source_names = {layer.name for layer in dense_source.layers}
    transferred = []
    skipped = []
    for layer in target.layers:
        if not layer.weights:
            continue
        source_layer = None
        if layer.name in dense_source_names:
            source_layer = dense_source.get_layer(layer.name)
        elif layer.name == "prune_predictions":
            source_layer = source.get_layer("dense_2")
        if source_layer is None:
            skipped.append(layer.name)
            continue
        source_weights = source_layer.get_weights()
        target_weights = layer.get_weights()
        if [w.shape for w in source_weights] == [w.shape for w in target_weights]:
            layer.set_weights(source_weights)
            transferred.append(layer.name)
        else:
            skipped.append(
                {
                    "layer": layer.name,
                    "source_shapes": [list(w.shape) for w in source_weights],
                    "target_shapes": [list(w.shape) for w in target_weights],
                }
            )
    return {"transferred": transferred, "skipped": skipped}


def make_deploy_model(core: keras.Model) -> keras.Model:
    raw = keras.Input(shape=(rps.IMG_SIZE, rps.IMG_SIZE, 3), name="image")
    x = keras.applications.densenet.preprocess_input(raw)
    out = core(x)
    return keras.Model(raw, out, name="rps_pruned_raw_input_deploy")


def random_indices(n: int, count: int, seed: int) -> list[int]:
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    return [int(i) for i in idx[: min(count, n)]]


def representative_raw(samples: np.ndarray):
    def gen():
        for sample in samples:
            yield [sample[None, ...].astype(np.float32)]

    return gen


def convert_sparse_int8(model: keras.Model, output_path: Path, x_rep_raw: np.ndarray) -> dict:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT, tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
    converter.representative_dataset = representative_raw(x_rep_raw)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output_path.write_bytes(converter.convert())
    return {
        "path": str(output_path),
        "size_mb": output_path.stat().st_size / 1024 / 1024,
        "optimizations": ["DEFAULT", "EXPERIMENTAL_SPARSITY"],
    }


def convert_sparse_float(model: keras.Model, output_path: Path) -> dict:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
    output_path.write_bytes(converter.convert())
    return {
        "path": str(output_path),
        "size_mb": output_path.stat().st_size / 1024 / 1024,
        "optimizations": ["EXPERIMENTAL_SPARSITY"],
    }


def write_history(history: keras.callbacks.History, path: Path) -> None:
    keys = list(history.history.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for i in range(len(history.history[keys[0]])):
            writer.writerow([i + 1] + [history.history[key][i] for key in keys])


def layer_sparsity(model: keras.Model) -> dict:
    rows = {}
    total_zero = 0
    total_count = 0
    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue
        zero = 0
        count = 0
        for weight in weights:
            if not np.issubdtype(weight.dtype, np.floating):
                continue
            zero += int(np.sum(weight == 0))
            count += int(weight.size)
        if count:
            rows[layer.name] = {
                "zeros": zero,
                "count": count,
                "sparsity": zero / count,
            }
            total_zero += zero
            total_count += count
    return {
        "total_zeros": total_zero,
        "total_count": total_count,
        "total_sparsity": total_zero / total_count if total_count else 0.0,
        "layers": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-keras", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sparsities", nargs="+", type=float, default=[0.5, 0.75])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--rep-count", type=int, default=120)
    parser.add_argument("--rep-trial-seed", type=int, default=31)
    args = parser.parse_args()

    rps.set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import tensorflow_model_optimization as tfmot

    files_by_class = rps.collect_files(args.data_dir)
    split, split_counts = rps.split_files(files_by_class, args.seed)
    x_train_raw, y_train = rps.load_images(split["train"])
    x_val_raw, y_val = rps.load_images(split["val"])
    x_test_raw, y_test = rps.load_images(split["test"])
    x_train = rps.preprocess_np(x_train_raw)
    x_val = rps.preprocess_np(x_val_raw)
    x_test = rps.preprocess_np(x_test_raw)

    rep_seed = args.seed * 1000 + args.rep_count * 10 + args.rep_trial_seed
    rep_indices = random_indices(len(x_train_raw), args.rep_count, rep_seed)
    x_rep_raw = x_train_raw[rep_indices]

    baseline_core = build_prune_compatible_core()
    transfer_info = transfer_base_weights(args.base_keras, baseline_core)
    baseline_core.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    baseline_float_eval = baseline_core.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=0)
    baseline_deploy = make_deploy_model(baseline_core)
    baseline_int8_path = args.output_dir / "baseline_prune_compatible_ptq_c120_s31_full_int8_io.tflite"
    baseline_tflite = convert_sparse_int8(baseline_deploy, baseline_int8_path, x_rep_raw)
    baseline_tflite_eval = rps.evaluate_tflite(baseline_int8_path, x_test_raw, y_test)

    results = {
        "tensorflow": tf.__version__,
        "tfmot": tfmot.__version__,
        "seed": args.seed,
        "sparsities": args.sparsities,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "split_counts": split_counts,
        "representative": {
            "count": int(len(rep_indices)),
            "trial_seed": args.rep_trial_seed,
            "seed": rep_seed,
            "class_counts": {
                str(cls): int(np.sum(y_train[rep_indices] == cls)) for cls in range(rps.NUM_CLASSES)
            },
        },
        "transfer": {
            "transferred_count": len(transfer_info["transferred"]),
            "skipped_count": len(transfer_info["skipped"]),
            "skipped": transfer_info["skipped"],
        },
        "baseline": {
            "float_test_loss": float(baseline_float_eval[0]),
            "float_test_accuracy": float(baseline_float_eval[1]),
            "tflite": baseline_tflite,
            "tflite_test_eval": baseline_tflite_eval,
        },
        "variants": {},
    }

    steps_per_epoch = int(math.ceil(len(x_train) / args.batch_size))
    for sparsity in args.sparsities:
        name = f"sparsity_{int(round(sparsity * 100))}"
        out_dir = args.output_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        core = build_prune_compatible_core()
        transfer_base_weights(args.base_keras, core)
        pruning_params = {
            "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=min(0.25, sparsity),
                final_sparsity=sparsity,
                begin_step=0,
                end_step=steps_per_epoch * args.epochs,
            ),
            "pruning_policy": tfmot.sparsity.keras.PruneForLatencyOnXNNPack(),
        }
        policy = "PruneForLatencyOnXNNPack"
        try:
            pruned = tfmot.sparsity.keras.prune_low_magnitude(core, **pruning_params)
        except ValueError as exc:
            policy = "generic_prune_low_magnitude_fallback"
            pruning_params.pop("pruning_policy", None)
            pruned = tfmot.sparsity.keras.prune_low_magnitude(core, **pruning_params)
            results["variants"][name] = {"policy_error": str(exc)}

        pruned.compile(
            optimizer=keras.optimizers.Adam(args.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        t0 = time.time()
        history = pruned.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=2,
            callbacks=[tfmot.sparsity.keras.UpdatePruningStep()],
        )
        elapsed = time.time() - t0
        write_history(history, out_dir / "pruning_history.csv")
        pruned_eval = pruned.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=0)
        export_core = tfmot.sparsity.keras.strip_pruning(pruned)
        export_core.save_weights(str(out_dir / "stripped_pruned_core.weights.h5"))
        deploy = make_deploy_model(export_core)

        sparse_int8_path = out_dir / f"03_aug_geometric_brightness10.pruned_{name}_sparse_full_int8_io.tflite"
        sparse_float_path = out_dir / f"03_aug_geometric_brightness10.pruned_{name}_sparse_float.tflite"
        sparse_int8_info = convert_sparse_int8(deploy, sparse_int8_path, x_rep_raw)
        sparse_float_info = convert_sparse_float(deploy, sparse_float_path)
        sparse_int8_eval = rps.evaluate_tflite(sparse_int8_path, x_test_raw, y_test)
        sparse_float_eval = rps.evaluate_tflite(sparse_float_path, x_test_raw, y_test)

        variant_result = results["variants"].get(name, {})
        variant_result.update(
            {
                "policy": policy,
                "target_sparsity": sparsity,
                "elapsed_sec": elapsed,
                "keras_test_loss": float(pruned_eval[0]),
                "keras_test_accuracy": float(pruned_eval[1]),
                "best_val_loss": float(np.min(history.history["val_loss"])),
                "best_val_accuracy": float(np.max(history.history["val_accuracy"])),
                "sparsity": layer_sparsity(export_core),
                "sparse_int8_tflite": sparse_int8_info,
                "sparse_int8_test_eval": sparse_int8_eval,
                "sparse_float_tflite": sparse_float_info,
                "sparse_float_test_eval": sparse_float_eval,
            }
        )
        results["variants"][name] = variant_result
        (args.output_dir / "pruning_xnnpack_results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps({name: variant_result}, indent=2, ensure_ascii=False))

    (args.output_dir / "pruning_xnnpack_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
