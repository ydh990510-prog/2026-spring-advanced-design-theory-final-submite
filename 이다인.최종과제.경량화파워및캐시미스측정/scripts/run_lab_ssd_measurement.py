#!/usr/bin/env python3
"""Sidecar measurement runner for the professor's original RPS SSD lab."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "models" / "RPS_PreTrained_SSD.tflite"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "lab_ssd_measurement"

CLASS_LIST = "_ Scissors Rock Paper".split()
LABEL_NORMALIZED = {"scissors": "Scissors", "rock": "Rock", "paper": "Paper"}
COLOR_LIST = [(), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
IMG_SIZE = 320
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240


def load_interpreter_class():
    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter, "tflite_runtime"
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter

            return Interpreter, "tensorflow.lite"
        except ImportError as exc:
            raise SystemExit(
                "No TFLite interpreter found. Install tflite-runtime on Raspberry Pi "
                "or tensorflow for inspection/dry-run environments."
            ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["idle", "camera_only", "ssd_detection", "ssd_no_display"],
        default="ssd_detection",
    )
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--expected-label",
        choices=["none", "scissors", "rock", "paper"],
        default="none",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--display", choices=["auto", "on", "off"], default="auto")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def run_dir_for(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [args.mode]
    if args.expected_label != "none":
        parts.append(args.expected_label)
    parts.append(timestamp)
    return resolve_path(args.output_dir) / "_".join(parts)


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return None


def read_first_float(paths: list[Path], divisor: float = 1.0) -> float | None:
    for path in paths:
        text = read_text(path)
        if text is None:
            continue
        try:
            return float(text) / divisor
        except ValueError:
            continue
    return None


def run_command(command: list[str], timeout: float = 2.0) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout or result.stderr).strip()
    return output or None


def parse_vcgencmd_temperature(output: str | None) -> float | None:
    if not output:
        return None
    match = re.search(r"temp=([0-9.]+)'?C", output)
    return float(match.group(1)) if match else None


def parse_vcgencmd_clock(output: str | None) -> float | None:
    if not output or "=" not in output:
        return None
    try:
        return float(output.split("=", 1)[1]) / 1_000_000.0
    except ValueError:
        return None


def read_soc_temp_c() -> float | None:
    paths = [
        Path("/sys/class/thermal/thermal_zone0/temp"),
        Path("/sys/devices/virtual/thermal/thermal_zone0/temp"),
    ]
    value = read_first_float(paths, divisor=1000.0)
    if value is not None:
        return value
    return parse_vcgencmd_temperature(run_command(["vcgencmd", "measure_temp"]))


def read_pmic_temp_c() -> float | None:
    candidates = sorted(Path("/sys/class/thermal").glob("thermal_zone*/type"))
    for type_path in candidates:
        label = read_text(type_path)
        if not label or "pmic" not in label.lower():
            continue
        temp = read_first_float([type_path.parent / "temp"], divisor=1000.0)
        if temp is not None:
            return temp
    return None


def read_arm_clock_mhz() -> float | None:
    paths = [
        Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"),
        Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq"),
    ]
    value = read_first_float(paths, divisor=1000.0)
    if value is not None:
        return value
    return parse_vcgencmd_clock(run_command(["vcgencmd", "measure_clock", "arm"]))


def read_throttled_raw() -> str | None:
    output = run_command(["vcgencmd", "get_throttled"])
    if not output:
        return None
    if "=" in output:
        return output.split("=", 1)[1].strip()
    return output


def parse_throttled(raw: str | None) -> tuple[bool | None, bool | None]:
    if raw is None:
        return None, None
    try:
        value = int(raw, 16)
    except ValueError:
        return None, None
    throttled_any = bool(value & ((1 << 2) | (1 << 18)))
    undervoltage_any = bool(value & ((1 << 0) | (1 << 16)))
    return throttled_any, undervoltage_any


def read_cpu_times() -> tuple[int, int] | None:
    text = read_text(Path("/proc/stat"))
    if not text:
        return None
    first = text.splitlines()[0].split()
    if not first or first[0] != "cpu":
        return None
    values = [int(part) for part in first[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values)
    return total, idle


def cpu_percent(prev: tuple[int, int] | None, cur: tuple[int, int] | None) -> float | None:
    if prev is None or cur is None:
        return None
    total_delta = cur[0] - prev[0]
    idle_delta = cur[1] - prev[1]
    if total_delta <= 0:
        return None
    return max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))


def read_process_cpu_time() -> float | None:
    text = read_text(Path("/proc/self/stat"))
    if not text:
        return None
    try:
        after_name = text.rsplit(")", 1)[1].strip().split()
        utime_ticks = int(after_name[11])
        stime_ticks = int(after_name[12])
        ticks_per_second = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        return (utime_ticks + stime_ticks) / float(ticks_per_second)
    except (IndexError, KeyError, OSError, ValueError):
        return None


def process_cpu_percent(prev_cpu: float | None, cur_cpu: float | None, elapsed_s: float) -> float | None:
    if prev_cpu is None or cur_cpu is None or elapsed_s <= 0:
        return None
    return max(0.0, 100.0 * (cur_cpu - prev_cpu) / elapsed_s)


def read_memory_mb() -> dict[str, float | None]:
    status = read_text(Path("/proc/self/status"))
    rss_mb = None
    vms_mb = None
    if status:
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                rss_mb = float(line.split()[1]) / 1024.0
            elif line.startswith("VmSize:"):
                vms_mb = float(line.split()[1]) / 1024.0
    if rss_mb is not None or vms_mb is not None:
        return {"rss_mb": rss_mb, "vms_mb": vms_mb}

    try:
        import resource

        raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1024.0 * 1024.0 if platform.system().lower() == "darwin" else 1024.0
        rss_mb = raw_rss / divisor
    except (ImportError, OSError):
        rss_mb = None
    return {"rss_mb": rss_mb, "vms_mb": None}


class SystemSampler:
    def __init__(self) -> None:
        self.prev_wall = time.perf_counter()
        self.prev_cpu_times = read_cpu_times()
        self.prev_process_cpu = read_process_cpu_time()
        self.rows: list[dict[str, Any]] = []

    def sample(self, elapsed_s: float, phase: str) -> dict[str, Any]:
        now = time.perf_counter()
        cur_cpu_times = read_cpu_times()
        cur_process_cpu = read_process_cpu_time()
        mem = read_memory_mb()
        raw_throttled = read_throttled_raw()
        throttled_any, undervoltage_any = parse_throttled(raw_throttled)
        elapsed_wall = now - self.prev_wall
        row = {
            "elapsed_s": elapsed_s,
            "phase": phase,
            "rss_mb": mem["rss_mb"],
            "vms_mb": mem["vms_mb"],
            "cpu_percent_total": cpu_percent(self.prev_cpu_times, cur_cpu_times),
            "process_cpu_percent": process_cpu_percent(
                self.prev_process_cpu,
                cur_process_cpu,
                elapsed_wall,
            ),
            "soc_temp_c": read_soc_temp_c(),
            "pmic_temp_c": read_pmic_temp_c(),
            "arm_clock_mhz": read_arm_clock_mhz(),
            "get_throttled_raw": raw_throttled,
            "throttled_any": throttled_any,
            "undervoltage_any": undervoltage_any,
        }
        self.prev_wall = now
        self.prev_cpu_times = cur_cpu_times
        self.prev_process_cpu = cur_process_cpu
        self.rows.append(row)
        return row


def mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def maximum(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return max(clean) if clean else None


def bool_any(values: list[Any]) -> bool | None:
    clean = [value for value in values if value is not None and value != ""]
    if not clean:
        return None
    return any(bool(value) for value in clean)


def preprocess(frame: np.ndarray, input_dtype: Any) -> np.ndarray:
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, 0)
    img = img * (2 / 255) - 1
    return img.astype(input_dtype)


def extract_detections(output_details: list[dict[str, Any]], interpreter: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = interpreter.get_tensor(output_details[0]["index"])[0]
    boxes = interpreter.get_tensor(output_details[1]["index"])[0]
    class_indexes = interpreter.get_tensor(output_details[3]["index"])[0].astype(int)
    return boxes, class_indexes, scores


def detection_label(class_index_raw: int) -> str | None:
    class_index = int(class_index_raw) + 1
    if class_index < 1 or class_index >= len(CLASS_LIST):
        return None
    return CLASS_LIST[class_index]


def top_detection(class_indexes: np.ndarray, scores: np.ndarray) -> tuple[str | None, float | None]:
    best_label = None
    best_score = None
    for class_index_raw, score_raw in zip(class_indexes, scores):
        label = detection_label(int(class_index_raw))
        if label is None:
            continue
        score = float(score_raw)
        if best_score is None or score > best_score:
            best_label = label
            best_score = score
    return best_label, best_score


def draw_detections(
    frame: np.ndarray,
    boxes: np.ndarray,
    class_indexes: np.ndarray,
    scores: np.ndarray,
    confidence_threshold: float,
) -> int:
    frame_h, frame_w, _ = frame.shape
    drawn = 0
    for bbox, class_index_raw, score in zip(boxes, class_indexes, scores):
        if float(score) <= confidence_threshold:
            continue
        class_index = int(class_index_raw) + 1
        if class_index < 1 or class_index >= len(CLASS_LIST):
            continue

        label = CLASS_LIST[class_index]
        color = COLOR_LIST[class_index]
        confidence = round(float(score) * 100)
        ymin, xmin, ymax, xmax = bbox
        xmin = int(xmin * frame_w)
        xmax = int(xmax * frame_w)
        ymin = int(ymin * frame_h)
        ymax = int(ymax * frame_h)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=color, thickness=2)
        cv2.putText(
            frame,
            f"{label}: {confidence}%".upper(),
            (xmin, max(20, ymin - 7)),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            color,
            2,
        )
        drawn += 1
    return drawn


def choose_display(args: argparse.Namespace) -> tuple[bool, str]:
    if args.mode == "ssd_no_display":
        return False, "disabled by mode ssd_no_display"
    if args.mode not in {"ssd_detection", "camera_only"}:
        return False, "mode does not display frames"
    if args.display == "off":
        return False, "disabled by --display off"

    system = platform.system().lower()
    has_display_env = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if args.display == "auto" and system == "linux" and not has_display_env:
        return False, "DISPLAY/WAYLAND_DISPLAY not set"
    if args.display == "on" and system == "linux" and not has_display_env:
        return False, "display requested but DISPLAY/WAYLAND_DISPLAY not set"
    return True, "enabled"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def summarize(
    args: argparse.Namespace,
    run_dir: Path,
    model_path: Path,
    interpreter_backend: str | None,
    display_enabled: bool,
    display_reason: str,
    frame_rows: list[dict[str, Any]],
    system_rows: list[dict[str, Any]],
    measured_start: float,
    measured_end: float,
) -> dict[str, Any]:
    measured_rows = [row for row in frame_rows if not row.get("is_warmup")]
    measured_duration_s = max(0.0, measured_end - measured_start)
    frame_count = len(measured_rows)
    expected_label = None if args.expected_label == "none" else LABEL_NORMALIZED[args.expected_label]
    threshold = args.confidence_threshold

    detected_rows = [
        row for row in measured_rows if safe_float(row.get("top_confidence")) is not None and safe_float(row.get("top_confidence")) >= threshold
    ]
    correct_rows = [
        row for row in detected_rows if expected_label is not None and row.get("top_label") == expected_label
    ]
    wrong_rows = [
        row for row in detected_rows if expected_label is not None and row.get("top_label") != expected_label
    ]
    top_counts = Counter(row.get("top_label") or "none" for row in measured_rows)

    rss_values = [safe_float(row.get("rss_mb")) for row in system_rows]
    rss_clean = [value for value in rss_values if value is not None]
    vms_values = [safe_float(row.get("vms_mb")) for row in system_rows]
    soc_values = [safe_float(row.get("soc_temp_c")) for row in system_rows]
    pmic_values = [safe_float(row.get("pmic_temp_c")) for row in system_rows]
    clock_values = [safe_float(row.get("arm_clock_mhz")) for row in system_rows]
    soc_clean = [value for value in soc_values if value is not None]
    raw_throttled = [row.get("get_throttled_raw") for row in system_rows if row.get("get_throttled_raw")]

    summary = {
        "run_dir": str(run_dir),
        "mode": args.mode,
        "model": str(model_path),
        "interpreter_backend": interpreter_backend,
        "duration_requested_s": args.duration,
        "warmup_requested_s": args.warmup,
        "started_at": datetime.fromtimestamp(measured_start).isoformat(timespec="seconds"),
        "ended_at": datetime.fromtimestamp(measured_end).isoformat(timespec="seconds"),
        "display_enabled": display_enabled,
        "display_reason": display_reason,
        "metrics_note": "CPU, temperature, clock, and throttling values are power-related proxy metrics, not direct watt measurements.",
        "accuracy_note": "Detection accuracy is a frame-level label accuracy proxy against the expected label, not formal object detection mAP.",
        "frame_count": frame_count,
        "measured_duration_s": measured_duration_s,
        "effective_fps_by_duration": frame_count / measured_duration_s if measured_duration_s > 0 else None,
        "mean_capture_ms": mean([safe_float(row.get("capture_ms")) for row in measured_rows]),
        "mean_preprocess_ms": mean([safe_float(row.get("preprocess_ms")) for row in measured_rows]),
        "mean_inference_ms": mean([safe_float(row.get("inference_ms")) for row in measured_rows]),
        "mean_postprocess_draw_ms": mean([safe_float(row.get("postprocess_draw_ms")) for row in measured_rows]),
        "mean_display_wait_ms": mean([safe_float(row.get("display_wait_ms")) for row in measured_rows]),
        "mean_total_frame_ms": mean([safe_float(row.get("total_frame_ms")) for row in measured_rows]),
        "expected_label": expected_label or "none",
        "confidence_threshold": threshold,
        "top_label": top_counts.most_common(1)[0][0] if top_counts else None,
        "top_confidence": mean([safe_float(row.get("top_confidence")) for row in measured_rows]),
        "detection_rate": len(detected_rows) / frame_count if frame_count else None,
        "frame_accuracy_proxy": len(correct_rows) / frame_count if frame_count and expected_label else None,
        "miss_rate": (frame_count - len(detected_rows)) / frame_count if frame_count else None,
        "wrong_label_rate": len(wrong_rows) / frame_count if frame_count and expected_label else None,
        "top_label_counts": dict(top_counts),
        "rss_mb_start": rss_clean[0] if rss_clean else None,
        "rss_mb_end": rss_clean[-1] if rss_clean else None,
        "rss_mb_mean": mean(rss_values),
        "rss_mb_peak": maximum(rss_values),
        "rss_mb_delta": (rss_clean[-1] - rss_clean[0]) if len(rss_clean) >= 2 else None,
        "vms_mb_mean": mean(vms_values),
        "vms_mb_peak": maximum(vms_values),
        "cpu_percent_total_mean": mean([safe_float(row.get("cpu_percent_total")) for row in system_rows]),
        "process_cpu_percent_mean": mean([safe_float(row.get("process_cpu_percent")) for row in system_rows]),
        "soc_temp_start_c": soc_clean[0] if soc_clean else None,
        "soc_temp_end_c": soc_clean[-1] if soc_clean else None,
        "soc_temp_delta_c": (soc_clean[-1] - soc_clean[0]) if len(soc_clean) >= 2 else None,
        "soc_temp_mean_c": mean(soc_values),
        "soc_temp_max_c": maximum(soc_values),
        "pmic_temp_mean_c": mean(pmic_values),
        "pmic_temp_max_c": maximum(pmic_values),
        "arm_clock_mhz_mean": mean(clock_values),
        "throttled_any": bool_any([row.get("throttled_any") for row in system_rows]),
        "undervoltage_any": bool_any([row.get("undervoltage_any") for row in system_rows]),
        "get_throttled_raw": raw_throttled[-1] if raw_throttled else None,
    }
    return summary


def run_idle(args: argparse.Namespace, sampler: SystemSampler) -> tuple[list[dict[str, Any]], float, float]:
    frame_rows: list[dict[str, Any]] = []
    start = time.time()
    measured_start = start + args.warmup
    end = measured_start + args.duration
    next_sample = 0.0
    while time.time() < end:
        now = time.time()
        elapsed = now - start
        phase = "warmup" if now < measured_start else "measure"
        if elapsed >= next_sample:
            sampler.sample(elapsed, phase)
            next_sample += 1.0
        time.sleep(0.05)
    measured_end = time.time()
    return frame_rows, measured_start, measured_end


def open_camera() -> Any:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit("Could not open camera index 0.")
    return cap


def run_camera_mode(
    args: argparse.Namespace,
    sampler: SystemSampler,
    display_enabled: bool,
    interpreter: Any | None,
    input_details: list[dict[str, Any]] | None,
    output_details: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], float, float]:
    cap = open_camera()
    input_dtype = input_details[0]["dtype"] if input_details else None
    frame_rows: list[dict[str, Any]] = []
    start = time.time()
    measured_start = start + args.warmup
    end = measured_start + args.duration
    next_system_sample = 0.0
    last_fps_time = time.time()
    frame_index = 0
    try:
        while time.time() < end:
            frame_start_perf = time.perf_counter()
            frame_start_wall = time.time()
            elapsed = frame_start_wall - start
            phase = "warmup" if frame_start_wall < measured_start else "measure"
            if elapsed >= next_system_sample:
                sampler.sample(elapsed, phase)
                next_system_sample += 1.0

            t0 = time.perf_counter()
            ret, frame = cap.read()
            t1 = time.perf_counter()
            if not ret:
                break

            preprocess_ms = None
            inference_ms = None
            postprocess_draw_ms = None
            top_label = None
            top_conf = None
            detection_pass = False
            drawn_count = 0
            if args.mode in {"ssd_detection", "ssd_no_display"} and interpreter is not None:
                t2 = time.perf_counter()
                img = preprocess(frame, input_dtype)
                t3 = time.perf_counter()
                interpreter.set_tensor(input_details[0]["index"], img)
                interpreter.invoke()
                t4 = time.perf_counter()
                boxes, class_indexes, scores = extract_detections(output_details, interpreter)
                top_label, top_conf = top_detection(class_indexes, scores)
                detection_pass = top_conf is not None and top_conf >= args.confidence_threshold
                t5 = time.perf_counter()
                if args.mode == "ssd_detection" and display_enabled:
                    drawn_count = draw_detections(frame, boxes, class_indexes, scores, args.confidence_threshold)
                t6 = time.perf_counter()
                preprocess_ms = (t3 - t2) * 1000.0
                inference_ms = (t4 - t3) * 1000.0
                postprocess_draw_ms = (t6 - t5) * 1000.0

            display_wait_ms = None
            if display_enabled:
                now = time.time()
                fps = 1.0 / (now - last_fps_time) if now > last_fps_time else 0.0
                last_fps_time = now
                cv2.putText(frame, f"FPS: {fps:.1f}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
                display_start = time.perf_counter()
                window_name = "RPS SSD Object Detection" if args.mode == "ssd_detection" else "cam"
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(10)
                display_end = time.perf_counter()
                display_wait_ms = (display_end - display_start) * 1000.0
                if key == ord("q"):
                    break

            frame_end_perf = time.perf_counter()
            expected_label = None if args.expected_label == "none" else LABEL_NORMALIZED[args.expected_label]
            frame_rows.append(
                {
                    "frame_index": frame_index,
                    "elapsed_s": elapsed,
                    "phase": phase,
                    "is_warmup": phase == "warmup",
                    "capture_ms": (t1 - t0) * 1000.0,
                    "preprocess_ms": preprocess_ms,
                    "inference_ms": inference_ms,
                    "postprocess_draw_ms": postprocess_draw_ms,
                    "display_wait_ms": display_wait_ms,
                    "total_frame_ms": (frame_end_perf - frame_start_perf) * 1000.0,
                    "top_label": top_label,
                    "top_confidence": top_conf,
                    "detection_pass": detection_pass,
                    "expected_label": expected_label or "none",
                    "expected_match": bool(expected_label and detection_pass and top_label == expected_label),
                    "drawn_count": drawn_count,
                }
            )
            frame_index += 1
    finally:
        cap.release()
        if display_enabled:
            cv2.destroyAllWindows()
    measured_end = time.time()
    return frame_rows, measured_start, measured_end


def main() -> int:
    args = parse_args()
    if args.duration < 0 or args.warmup < 0:
        raise SystemExit("--duration and --warmup must be non-negative.")

    model_path = resolve_path(args.model)
    if args.mode in {"ssd_detection", "ssd_no_display"} and not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    run_dir = run_dir_for(args)
    run_dir.mkdir(parents=True, exist_ok=False)
    display_enabled, display_reason = choose_display(args)

    interpreter = None
    input_details = None
    output_details = None
    interpreter_backend = None
    if args.mode in {"ssd_detection", "ssd_no_display"}:
        interpreter_class, interpreter_backend = load_interpreter_class()
        interpreter = interpreter_class(model_path=str(model_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("model:", model_path)
        print("input:", input_details)
        print("outputs:", output_details)

    sampler = SystemSampler()
    sampler.sample(0.0, "start")
    if args.mode == "idle":
        frame_rows, measured_start, measured_end = run_idle(args, sampler)
    else:
        frame_rows, measured_start, measured_end = run_camera_mode(
            args,
            sampler,
            display_enabled,
            interpreter,
            input_details,
            output_details,
        )
    sampler.sample(max(0.0, time.time() - (measured_start - args.warmup)), "end")

    frame_fields = [
        "frame_index",
        "elapsed_s",
        "phase",
        "is_warmup",
        "capture_ms",
        "preprocess_ms",
        "inference_ms",
        "postprocess_draw_ms",
        "display_wait_ms",
        "total_frame_ms",
        "top_label",
        "top_confidence",
        "detection_pass",
        "expected_label",
        "expected_match",
        "drawn_count",
    ]
    system_fields = [
        "elapsed_s",
        "phase",
        "rss_mb",
        "vms_mb",
        "cpu_percent_total",
        "process_cpu_percent",
        "soc_temp_c",
        "pmic_temp_c",
        "arm_clock_mhz",
        "get_throttled_raw",
        "throttled_any",
        "undervoltage_any",
    ]
    write_csv(run_dir / "frame_metrics.csv", frame_rows, frame_fields)
    write_csv(run_dir / "system_metrics.csv", sampler.rows, system_fields)
    summary = summarize(
        args,
        run_dir,
        model_path,
        interpreter_backend,
        display_enabled,
        display_reason,
        frame_rows,
        sampler.rows,
        measured_start,
        measured_end,
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved run: {run_dir}")
    print(f"Summary: {run_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
