#!/usr/bin/env python3
"""Detailed Raspberry Pi RPS camera/TFLite timing breakdown."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pi_realtime_rps import (
    DEFAULT_LABELS,
    StableState,
    center_crop,
    dequantize_output,
    load_interpreter,
    open_frame_source,
    quantize_input,
    update_stable_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detailed timing breakdown for RPS TFLite.")
    parser.add_argument("--model", action="append", required=True, help="Path to .tflite model.")
    parser.add_argument("--camera", default="0", help="Camera index/path.")
    parser.add_argument("--camera-backend", choices=("auto", "opencv", "rpicam"), default="auto")
    parser.add_argument("--capture-width", type=int, default=640)
    parser.add_argument("--capture-height", type=int, default=480)
    parser.add_argument("--capture-fps", type=int, default=30)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--roi-scale", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--stable-frames", type=int, default=8)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument(
        "--dashboard-draw",
        action="store_true",
        help="Also time dashboard image composition, without cv2.imshow.",
    )
    parser.add_argument("--json-out", help="Optional output JSON path.")
    return parser.parse_args()


def now_ns() -> int:
    return time.perf_counter_ns()


def add_ms(bucket: dict[str, list[float]], name: str, start_ns: int, end_ns: int) -> None:
    bucket[name].append((end_ns - start_ns) / 1_000_000.0)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100.0
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    mean = statistics.fmean(values)
    return {
        "mean_ms": round(mean, 4),
        "median_ms": round(statistics.median(values), 4),
        "p90_ms": round(percentile(values, 90), 4),
        "p95_ms": round(percentile(values, 95), 4),
        "p99_ms": round(percentile(values, 99), 4),
        "min_ms": round(min(values), 4),
        "max_ms": round(max(values), 4),
        "std_ms": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
    }


def summarize_all(times: dict[str, list[float]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    total_mean = statistics.fmean(times["total_loop"]) if times.get("total_loop") else 0.0
    for name, values in times.items():
        item = summarize(values)
        if total_mean and item:
            item["mean_pct_of_loop"] = round(item["mean_ms"] / total_mean * 100.0, 2)
        summary[name] = item
    if total_mean:
        summary["derived"] = {
            "e2e_fps_from_total_loop_mean": round(1000.0 / total_mean, 3),
            "camera_bound_fps_from_read_mean": round(
                1000.0 / statistics.fmean(times["camera_read"]), 3
            )
            if times.get("camera_read")
            else 0.0,
        }
    return summary


def prepare_input(frame: np.ndarray, input_details: dict, roi_scale: float, times: dict[str, list[float]]):
    input_shape = input_details["shape"]
    input_height = int(input_shape[1])
    input_width = int(input_shape[2])

    t0 = now_ns()
    crop, roi = center_crop(frame, roi_scale)
    t1 = now_ns()
    image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    t2 = now_ns()
    image = cv2.resize(image, (input_width, input_height), interpolation=cv2.INTER_AREA)
    t3 = now_ns()
    image = quantize_input(image, input_details)
    t4 = now_ns()
    input_tensor = np.expand_dims(image, axis=0)
    t5 = now_ns()

    add_ms(times, "preprocess_center_crop", t0, t1)
    add_ms(times, "preprocess_bgr_to_rgb", t1, t2)
    add_ms(times, "preprocess_resize", t2, t3)
    add_ms(times, "preprocess_quantize_or_cast", t3, t4)
    add_ms(times, "preprocess_expand_dims", t4, t5)
    add_ms(times, "preprocess_total", t0, t5)
    return input_tensor, roi


def run_predict(
    interpreter: Any,
    input_details: dict,
    output_details: dict,
    input_tensor: np.ndarray,
    times: dict[str, list[float]],
) -> tuple[int, float, np.ndarray]:
    t0 = now_ns()
    interpreter.set_tensor(input_details["index"], input_tensor)
    t1 = now_ns()
    interpreter.invoke()
    t2 = now_ns()
    output = interpreter.get_tensor(output_details["index"])
    t3 = now_ns()
    scores = dequantize_output(output, output_details)
    t4 = now_ns()
    label_id = int(np.argmax(scores))
    confidence = float(scores[label_id])
    t5 = now_ns()

    add_ms(times, "tflite_set_tensor", t0, t1)
    add_ms(times, "tflite_invoke", t1, t2)
    add_ms(times, "tflite_get_tensor", t2, t3)
    add_ms(times, "postprocess_dequantize", t3, t4)
    add_ms(times, "postprocess_argmax_confidence", t4, t5)
    add_ms(times, "predict_total", t0, t5)
    return label_id, confidence, scores


def maybe_draw_dashboard(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    labels: list[str],
    scores: np.ndarray,
    label: str,
    confidence: float,
    threshold: float,
    state: StableState,
    model_path: str,
):
    from pi_rps_window_dashboard import RuntimeStats, draw_dashboard

    stats = RuntimeStats()
    stats.loop_fps = 0.0
    stats.inference_ms = 0.0
    return draw_dashboard(
        frame=frame,
        roi=roi,
        labels=labels,
        scores=scores,
        label=label,
        display_label="uncertain" if confidence < threshold else label,
        confidence=confidence,
        threshold=threshold,
        state=state,
        stats=stats,
        camera_backend="benchmark",
        model_path=model_path,
        interpreter_name="benchmark",
    )


def run_model(args: argparse.Namespace, model_path: str) -> dict[str, Any]:
    labels = list(DEFAULT_LABELS)

    load_start = time.perf_counter()
    interpreter, input_details, output_details = load_interpreter(model_path)
    load_ms = (time.perf_counter() - load_start) * 1000.0

    frame_source, camera_backend = open_frame_source(args)
    times: dict[str, list[float]] = defaultdict(list)
    predictions: dict[str, int] = defaultdict(int)
    confidences: list[float] = []
    state = StableState()
    frames_seen = 0

    try:
        total_target = args.warmup + args.runs
        while frames_seen < total_target:
            measured = frames_seen >= args.warmup
            loop_start = now_ns()

            t0 = now_ns()
            ok, frame = frame_source.read()
            t1 = now_ns()
            if not ok or frame is None:
                continue
            if measured:
                add_ms(times, "camera_read", t0, t1)

            if args.mirror:
                t0 = now_ns()
                frame = cv2.flip(frame, 1)
                t1 = now_ns()
                if measured:
                    add_ms(times, "mirror_flip", t0, t1)

            input_tensor, roi = prepare_input(frame, input_details, args.roi_scale, times if measured else defaultdict(list))
            label_id, confidence, scores = run_predict(
                interpreter,
                input_details,
                output_details,
                input_tensor,
                times if measured else defaultdict(list),
            )
            label = labels[label_id]

            t0 = now_ns()
            update_stable_state(state, label, confidence, args.threshold)
            t1 = now_ns()
            if measured:
                add_ms(times, "stable_state_update", t0, t1)

            if args.dashboard_draw:
                t0 = now_ns()
                _ = maybe_draw_dashboard(
                    frame, roi, labels, scores, label, confidence, args.threshold, state, model_path
                )
                t1 = now_ns()
                if measured:
                    add_ms(times, "dashboard_draw_no_imshow", t0, t1)

            loop_end = now_ns()
            if measured:
                add_ms(times, "total_loop", loop_start, loop_end)
                predictions[label] += 1
                confidences.append(confidence)
            frames_seen += 1
    finally:
        frame_source.release()

    result = {
        "model": str(Path(model_path).resolve()),
        "model_size_mb": round(Path(model_path).stat().st_size / 1024 / 1024, 4),
        "camera_backend": camera_backend,
        "load_allocate_ms": round(load_ms, 4),
        "input_shape": input_details["shape"].tolist(),
        "input_dtype": str(input_details["dtype"]),
        "input_quantization": input_details.get("quantization"),
        "output_shape": output_details["shape"].tolist(),
        "output_dtype": str(output_details["dtype"]),
        "output_quantization": output_details.get("quantization"),
        "runs": args.runs,
        "warmup": args.warmup,
        "prediction_counts": dict(predictions),
        "mean_confidence": round(statistics.fmean(confidences), 4) if confidences else 0.0,
        "timing": summarize_all(times),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    args = parse_args()
    results = [run_model(args, model_path) for model_path in args.model]
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
