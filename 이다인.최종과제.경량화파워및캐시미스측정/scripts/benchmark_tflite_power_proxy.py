#!/usr/bin/env python3
"""Headless TFLite benchmark with power-related proxy metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "headless_tflite_power_proxy"
DEFAULT_MODELS = [
    REPO_ROOT / "models" / "RPS_PreTrained_SSD.tflite",
    REPO_ROOT / "models" / "rps_ssd_lite_fp32.tflite",
    REPO_ROOT / "models" / "rps_ssd_lite_int8.tflite",
]


def load_interpreter_class():
    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter, "tflite_runtime"
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter

            return Interpreter, "tensorflow.lite"
        except ImportError as exc:
            raise SystemExit("Install tflite-runtime on Raspberry Pi or tensorflow on Mac.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--cooldown", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return None


def run_command(command: list[str], timeout: float = 2.0) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout or result.stderr).strip()
    return output or None


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
        return {"rss_mb": rss_mb, "vms_mb": vms_mb}

    try:
        import resource

        raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1024.0 * 1024.0 if platform.system().lower() == "darwin" else 1024.0
        rss_mb = raw_rss / divisor
    except (ImportError, OSError):
        rss_mb = None
    return {"rss_mb": rss_mb, "vms_mb": None}


def read_cpu_times() -> tuple[int, int] | None:
    text = read_text(Path("/proc/stat"))
    if not text:
        return None
    parts = text.splitlines()[0].split()
    if not parts or parts[0] != "cpu":
        return None
    values = [int(item) for item in parts[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


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
        ticks = int(after_name[11]) + int(after_name[12])
        return ticks / float(os.sysconf(os.sysconf_names["SC_CLK_TCK"]))
    except (IndexError, KeyError, OSError, ValueError):
        return None


def parse_temp(output: str | None) -> float | None:
    if not output or "temp=" not in output:
        return None
    try:
        return float(output.split("temp=", 1)[1].split("'")[0])
    except ValueError:
        return None


def read_soc_temp_c() -> float | None:
    text = read_text(Path("/sys/class/thermal/thermal_zone0/temp"))
    if text:
        try:
            return float(text) / 1000.0
        except ValueError:
            pass
    return parse_temp(run_command(["vcgencmd", "measure_temp"]))


def read_pmic_temp_c() -> float | None:
    for type_path in sorted(Path("/sys/class/thermal").glob("thermal_zone*/type")):
        label = read_text(type_path)
        if label and "pmic" in label.lower():
            temp = read_text(type_path.parent / "temp")
            if temp:
                try:
                    return float(temp) / 1000.0
                except ValueError:
                    return None
    return None


def read_arm_clock_mhz() -> float | None:
    for path in [
        Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"),
        Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq"),
    ]:
        text = read_text(path)
        if text:
            try:
                return float(text) / 1000.0
            except ValueError:
                pass
    output = run_command(["vcgencmd", "measure_clock", "arm"])
    if output and "=" in output:
        try:
            return float(output.split("=", 1)[1]) / 1_000_000.0
        except ValueError:
            return None
    return None


def read_throttled_raw() -> str | None:
    output = run_command(["vcgencmd", "get_throttled"])
    if not output:
        return None
    return output.split("=", 1)[1].strip() if "=" in output else output


def throttled_flags(raw: str | None) -> tuple[bool | None, bool | None]:
    if raw is None:
        return None, None
    try:
        value = int(raw, 16)
    except ValueError:
        return None, None
    return bool(value & ((1 << 2) | (1 << 18))), bool(value & ((1 << 0) | (1 << 16)))


def sample_system(elapsed_s: float) -> dict[str, Any]:
    mem = read_memory_mb()
    raw = read_throttled_raw()
    throttled_any, undervoltage_any = throttled_flags(raw)
    return {
        "elapsed_s": elapsed_s,
        "rss_mb": mem["rss_mb"],
        "vms_mb": mem["vms_mb"],
        "soc_temp_c": read_soc_temp_c(),
        "pmic_temp_c": read_pmic_temp_c(),
        "arm_clock_mhz": read_arm_clock_mhz(),
        "get_throttled_raw": raw,
        "throttled_any": throttled_any,
        "undervoltage_any": undervoltage_any,
    }


def make_input(input_detail: dict[str, Any]) -> np.ndarray:
    shape = input_detail["shape"]
    dtype = input_detail["dtype"]
    fp = np.random.random(shape).astype(np.float32)
    if dtype == np.float32:
        return fp

    scale, zero_point = input_detail["quantization"]
    if not scale:
        raise SystemExit(f"Quantized input has invalid scale: {scale}")
    q = np.round(fp / scale + zero_point)
    q = np.clip(q, np.iinfo(dtype).min, np.iinfo(dtype).max)
    return q.astype(dtype)


def mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def maximum(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return max(clean) if clean else None


def benchmark_model(
    model_path: Path,
    interpreter_class: type,
    backend: str,
    warmup: int,
    repeat: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    interpreter = interpreter_class(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    sample = make_input(input_detail)

    for _ in range(warmup):
        interpreter.set_tensor(input_detail["index"], sample)
        interpreter.invoke()

    iteration_rows = []
    system_rows = []
    start_wall = time.time()
    prev_wall = time.perf_counter()
    prev_cpu = read_cpu_times()
    prev_proc = read_process_cpu_time()
    system_rows.append({"model": model_path.name, **sample_system(0.0), "cpu_percent_total": None, "process_cpu_percent": None})

    times = []
    for index in range(repeat):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_detail["index"], sample)
        interpreter.invoke()
        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000.0
        times.append(inference_ms)
        iteration_rows.append({"model": model_path.name, "iteration": index, "inference_ms": inference_ms})

        if index == repeat - 1 or (index + 1) % 10 == 0:
            now = time.perf_counter()
            cur_cpu = read_cpu_times()
            cur_proc = read_process_cpu_time()
            elapsed_wall = now - prev_wall
            proc_pct = None
            if prev_proc is not None and cur_proc is not None and elapsed_wall > 0:
                proc_pct = max(0.0, 100.0 * (cur_proc - prev_proc) / elapsed_wall)
            system_rows.append(
                {
                    "model": model_path.name,
                    **sample_system(time.time() - start_wall),
                    "cpu_percent_total": cpu_percent(prev_cpu, cur_cpu),
                    "process_cpu_percent": proc_pct,
                }
            )
            prev_wall = now
            prev_cpu = cur_cpu
            prev_proc = cur_proc

    rss = [row.get("rss_mb") for row in system_rows]
    vms = [row.get("vms_mb") for row in system_rows]
    soc = [row.get("soc_temp_c") for row in system_rows]
    pmic = [row.get("pmic_temp_c") for row in system_rows]
    clock = [row.get("arm_clock_mhz") for row in system_rows]
    throttled_values = [row.get("throttled_any") for row in system_rows if row.get("throttled_any") is not None]
    undervoltage_values = [row.get("undervoltage_any") for row in system_rows if row.get("undervoltage_any") is not None]

    summary = {
        "model": model_path.name,
        "model_path": str(model_path),
        "backend": backend,
        "file_size_bytes": model_path.stat().st_size,
        "file_size_mb": model_path.stat().st_size / (1024 * 1024),
        "input_shape": input_detail["shape"].tolist(),
        "input_dtype": np.dtype(input_detail["dtype"]).name,
        "input_quantization": input_detail["quantization"],
        "output_shapes": [detail["shape"].tolist() for detail in output_details],
        "output_dtypes": [np.dtype(detail["dtype"]).name for detail in output_details],
        "output_quantization": [detail["quantization"] for detail in output_details],
        "warmup": warmup,
        "repeat": repeat,
        "mean_inference_ms": statistics.fmean(times),
        "median_inference_ms": statistics.median(times),
        "min_inference_ms": min(times),
        "max_inference_ms": max(times),
        "rss_mb_start": next((value for value in rss if value is not None), None),
        "rss_mb_end": next((value for value in reversed(rss) if value is not None), None),
        "rss_mb_mean": mean(rss),
        "rss_mb_peak": maximum(rss),
        "vms_mb_mean": mean(vms),
        "vms_mb_peak": maximum(vms),
        "cpu_percent_total_mean": mean([row.get("cpu_percent_total") for row in system_rows]),
        "process_cpu_percent_mean": mean([row.get("process_cpu_percent") for row in system_rows]),
        "soc_temp_start_c": next((value for value in soc if value is not None), None),
        "soc_temp_end_c": next((value for value in reversed(soc) if value is not None), None),
        "soc_temp_mean_c": mean(soc),
        "soc_temp_max_c": maximum(soc),
        "pmic_temp_mean_c": mean(pmic),
        "pmic_temp_max_c": maximum(pmic),
        "arm_clock_mhz_mean": mean(clock),
        "throttled_any": any(throttled_values) if throttled_values else None,
        "undervoltage_any": any(undervoltage_values) if undervoltage_values else None,
        "get_throttled_raw": next((row.get("get_throttled_raw") for row in reversed(system_rows) if row.get("get_throttled_raw")), None),
        "metrics_note": "CPU, temperature, clock, and throttling values are power-related proxy metrics, not direct watt measurements.",
        "benchmark_note": "This is a headless random-input inference benchmark. It does not measure camera display behavior or formal detection accuracy.",
    }
    return summary, iteration_rows, system_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def json_safe(value: Any) -> Any:
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> int:
    args = parse_args()
    model_paths = [resolve(path) for path in args.models]
    missing = [path for path in model_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing model(s): {', '.join(str(path) for path in missing)}")

    interpreter_class, backend = load_interpreter_class()
    run_dir = resolve(args.output_dir) / datetime.now().strftime("headless_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    summaries = []
    all_iterations = []
    all_system = []
    for index, model_path in enumerate(model_paths):
        print(f"Benchmarking {model_path.name}...")
        summary, iteration_rows, system_rows = benchmark_model(
            model_path,
            interpreter_class,
            backend,
            args.warmup,
            args.repeat,
        )
        summaries.append(summary)
        all_iterations.extend(iteration_rows)
        all_system.extend(system_rows)
        if index < len(model_paths) - 1 and args.cooldown > 0:
            time.sleep(args.cooldown)

    write_csv(run_dir / "iteration_metrics.csv", all_iterations)
    write_csv(run_dir / "system_metrics.csv", all_system)
    write_csv(run_dir / "summary.csv", summaries)
    (run_dir / "summary.json").write_text(json.dumps(json_safe(summaries), indent=2) + "\n", encoding="utf-8")

    print(f"Saved benchmark: {run_dir}")
    for summary in summaries:
        print(
            f"{summary['model']}: mean={summary['mean_inference_ms']:.3f} ms, "
            f"size={summary['file_size_mb']:.3f} MB, rss_mean={summary['rss_mb_mean']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
