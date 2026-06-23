#!/usr/bin/env python3
"""Deep profiling helpers for RPS TFLite inference on Raspberry Pi."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * pct / 100.0
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": round(statistics.fmean(values), 5),
        "median_ms": round(statistics.median(values), 5),
        "p90_ms": round(percentile(values, 90), 5),
        "p95_ms": round(percentile(values, 95), 5),
        "p99_ms": round(percentile(values, 99), 5),
        "min_ms": round(min(values), 5),
        "max_ms": round(max(values), 5),
        "std_ms": round(statistics.pstdev(values), 5) if len(values) > 1 else 0.0,
    }


def interpreter_classes():
    classes = []
    try:
        from tflite_runtime.interpreter import Interpreter

        classes.append(("tflite_runtime", Interpreter, {}))
    except Exception:
        pass

    try:
        import tensorflow as tf

        classes.append(("tensorflow", tf.lite.Interpreter, {}))
        try:
            no_delegate = tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            classes.append(
                (
                    "tensorflow_no_default_delegates",
                    tf.lite.Interpreter,
                    {"experimental_op_resolver_type": no_delegate},
                )
            )
        except Exception:
            pass
    except Exception:
        pass
    return classes


def make_input(input_details: dict, kind: str) -> np.ndarray:
    shape = tuple(int(v) for v in input_details["shape"])
    dtype = input_details["dtype"]
    if kind == "zeros":
        data = np.zeros(shape, dtype=np.float32)
    elif kind == "ones":
        data = np.ones(shape, dtype=np.float32) * 127.0
    else:
        rng = np.random.default_rng(20260618)
        data = rng.uniform(0.0, 255.0, size=shape).astype(np.float32)

    if dtype == np.float32:
        return data.astype(np.float32)
    if dtype == np.uint8:
        return np.clip(data, 0, 255).astype(np.uint8)
    if dtype == np.int8:
        scale, zero_point = input_details.get("quantization", (0.0, 0))
        if scale:
            data = np.round(data / scale + zero_point)
        return np.clip(data, -128, 127).astype(np.int8)
    return data.astype(dtype)


def run_microbenchmark(
    model_path: Path,
    interpreter_name: str,
    Interpreter: Any,
    extra_kwargs: dict[str, Any],
    num_threads: int | None,
    runs: int,
    warmup: int,
    input_kind: str,
) -> dict[str, Any]:
    kwargs = dict(extra_kwargs)
    if num_threads is not None:
        kwargs["num_threads"] = num_threads
    interpreter = Interpreter(model_path=str(model_path), **kwargs)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_tensor = make_input(input_details, input_kind)

    for _ in range(warmup):
        interpreter.set_tensor(input_details["index"], input_tensor)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details["index"])

    buckets: dict[str, list[float]] = defaultdict(list)
    for _ in range(runs):
        total_start = time.perf_counter_ns()

        t0 = time.perf_counter_ns()
        interpreter.set_tensor(input_details["index"], input_tensor)
        t1 = time.perf_counter_ns()
        interpreter.invoke()
        t2 = time.perf_counter_ns()
        output = interpreter.get_tensor(output_details["index"])
        t3 = time.perf_counter_ns()
        output = np.squeeze(output).astype(np.float32)
        t4 = time.perf_counter_ns()
        _label = int(np.argmax(output))
        _confidence = float(output[_label])
        t5 = time.perf_counter_ns()

        buckets["set_tensor"].append((t1 - t0) / 1_000_000)
        buckets["invoke"].append((t2 - t1) / 1_000_000)
        buckets["get_tensor"].append((t3 - t2) / 1_000_000)
        buckets["squeeze_cast"].append((t4 - t3) / 1_000_000)
        buckets["argmax_confidence"].append((t5 - t4) / 1_000_000)
        buckets["predict_total"].append((t5 - total_start) / 1_000_000)

    total = statistics.fmean(buckets["predict_total"])
    return {
        "model": str(model_path),
        "model_size_mb": round(model_path.stat().st_size / 1024 / 1024, 4),
        "interpreter": interpreter_name,
        "num_threads": num_threads if num_threads is not None else "default",
        "input_kind": input_kind,
        "input_shape": input_details["shape"].tolist(),
        "input_dtype": str(input_details["dtype"]),
        "output_shape": output_details["shape"].tolist(),
        "output_dtype": str(output_details["dtype"]),
        "runs": runs,
        "warmup": warmup,
        "timing": {
            name: {
                **summarize(values),
                "mean_pct_of_predict": round(statistics.fmean(values) / total * 100.0, 3)
                if total
                else 0.0,
            }
            for name, values in buckets.items()
        },
    }


def tensor_shape(tensor: Any) -> list[int]:
    return [int(tensor.Shape(i)) for i in range(tensor.ShapeLength())]


def tensor_type_name(schema_fb: Any, t: int) -> str:
    names = {
        value: key
        for key, value in schema_fb.TensorType.__dict__.items()
        if key.isupper() and isinstance(value, int)
    }
    return names.get(t, str(t))


def builtin_name_map(schema_fb: Any) -> dict[int, str]:
    return {
        value: key
        for key, value in schema_fb.BuiltinOperator.__dict__.items()
        if key.isupper() and isinstance(value, int)
    }


def estimate_macs(op_name: str, in_shapes: list[list[int]], out_shapes: list[list[int]]) -> int:
    try:
        if op_name == "CONV_2D" and len(in_shapes) >= 2 and out_shapes:
            filt = in_shapes[1]
            out = out_shapes[0]
            if len(filt) == 4 and len(out) == 4:
                out_h, out_w, out_c = out[1], out[2], out[3]
                out_ch, kh, kw, in_ch = filt
                return int(out_h * out_w * out_ch * kh * kw * in_ch)
        if op_name == "DEPTHWISE_CONV_2D" and len(in_shapes) >= 2 and out_shapes:
            filt = in_shapes[1]
            out = out_shapes[0]
            if len(filt) == 4 and len(out) == 4:
                out_h, out_w, out_c = out[1], out[2], out[3]
                _, kh, kw, _ = filt
                return int(out_h * out_w * out_c * kh * kw)
        if op_name == "FULLY_CONNECTED" and len(in_shapes) >= 2:
            weights = in_shapes[1]
            if len(weights) == 2:
                return int(weights[0] * weights[1])
        if op_name in {"AVERAGE_POOL_2D", "MAX_POOL_2D"} and in_shapes and out_shapes:
            out = out_shapes[0]
            if len(out) == 4:
                return int(math.prod(out))
    except Exception:
        return 0
    return 0


def analyze_graph(model_path: Path, output_dir: Path) -> dict[str, Any]:
    import tensorflow as tf
    from tensorflow.lite.python import schema_py_generated as schema_fb

    analyzer_text = io.StringIO()
    with redirect_stdout(analyzer_text):
        tf.lite.experimental.Analyzer.analyze(model_path=str(model_path), gpu_compatibility=False)
    analyzer_path = output_dir / f"{model_path.stem}.analyzer.txt"
    analyzer_path.write_text(analyzer_text.getvalue(), encoding="utf-8")

    buf = model_path.read_bytes()
    model = schema_fb.Model.GetRootAsModel(buf, 0)
    names = builtin_name_map(schema_fb)
    op_codes = []
    for i in range(model.OperatorCodesLength()):
        code = model.OperatorCodes(i)
        builtin = int(code.BuiltinCode())
        custom = code.CustomCode()
        op_codes.append(custom.decode() if custom else names.get(builtin, str(builtin)))

    subgraph = model.Subgraphs(0)
    tensors = []
    tensor_bytes_by_type = Counter()
    type_counts = Counter()
    for i in range(subgraph.TensorsLength()):
        t = subgraph.Tensors(i)
        shape = tensor_shape(t)
        t_type = tensor_type_name(schema_fb, int(t.Type()))
        tensors.append(
            {
                "index": i,
                "name": t.Name().decode(errors="replace") if t.Name() else "",
                "shape": shape,
                "type": t_type,
            }
        )
        type_counts[t_type] += 1

    rows = []
    op_counts = Counter()
    macs_by_op = Counter()
    for i in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(i)
        op_name = op_codes[op.OpcodeIndex()]
        inputs = [int(op.Inputs(j)) for j in range(op.InputsLength())]
        outputs = [int(op.Outputs(j)) for j in range(op.OutputsLength())]
        in_shapes = [tensors[idx]["shape"] for idx in inputs if idx >= 0 and idx < len(tensors)]
        out_shapes = [tensors[idx]["shape"] for idx in outputs if idx >= 0 and idx < len(tensors)]
        macs = estimate_macs(op_name, in_shapes, out_shapes)
        op_counts[op_name] += 1
        macs_by_op[op_name] += macs
        rows.append(
            {
                "index": i,
                "op": op_name,
                "inputs": inputs,
                "outputs": outputs,
                "input_shapes": in_shapes,
                "output_shapes": out_shapes,
                "estimated_macs": macs,
                "output_name": tensors[outputs[0]]["name"] if outputs and outputs[0] >= 0 else "",
            }
        )

    ops_csv = output_dir / f"{model_path.stem}.ops.csv"
    with ops_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "op",
                "inputs",
                "outputs",
                "input_shapes",
                "output_shapes",
                "estimated_macs",
                "output_name",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    total_macs = sum(row["estimated_macs"] for row in rows)
    top_macs = sorted(rows, key=lambda r: r["estimated_macs"], reverse=True)[:20]
    summary = {
        "model": str(model_path),
        "model_size_mb": round(model_path.stat().st_size / 1024 / 1024, 4),
        "subgraphs": model.SubgraphsLength(),
        "operators_total": len(rows),
        "op_counts": dict(op_counts.most_common()),
        "tensor_type_counts": dict(type_counts.most_common()),
        "estimated_total_macs": int(total_macs),
        "estimated_total_mmacs": round(total_macs / 1_000_000, 3),
        "estimated_macs_by_op": {
            key: int(value) for key, value in macs_by_op.most_common()
        },
        "top_macs_ops": top_macs,
        "analyzer_txt": str(analyzer_path),
        "ops_csv": str(ops_csv),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = interpreter_classes()
    results = []
    for model in args.model:
        model_path = Path(model)
        graph_summary = analyze_graph(model_path, output_dir)
        model_results = {"graph": graph_summary, "benchmarks": []}
        for interpreter_name, Interpreter, extra_kwargs in classes:
            if interpreter_name == "tensorflow_no_default_delegates":
                thread_options = [1]
                runs = max(100, args.runs // 3)
                warmup = max(20, args.warmup // 2)
            else:
                thread_options = [1, 2, 4]
                runs = args.runs
                warmup = args.warmup
            for threads in thread_options:
                for input_kind in ["zeros", "random"]:
                    try:
                        result = run_microbenchmark(
                            model_path,
                            interpreter_name,
                            Interpreter,
                            extra_kwargs,
                            threads,
                            runs,
                            warmup,
                            input_kind,
                        )
                        model_results["benchmarks"].append(result)
                        print(json.dumps(result, ensure_ascii=False))
                    except Exception as exc:
                        print(
                            json.dumps(
                                {
                                    "model": str(model_path),
                                    "interpreter": interpreter_name,
                                    "num_threads": threads,
                                    "input_kind": input_kind,
                                    "error": f"{type(exc).__name__}: {exc}",
                                },
                                ensure_ascii=False,
                            )
                        )
        results.append(model_results)

    out_json = output_dir / "inference_deep_profile.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
