#!/usr/bin/env python3
"""Measure Raspberry Pi CSI camera FPS through the existing rpicam MJPEG path."""

from __future__ import annotations

import argparse
import json
import os
import select
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class RpicamMjpegFrameSourceQuiet:
    def __init__(self, camera: str, width: int, height: int, fps: int) -> None:
        command = [
            "rpicam-vid",
            "--camera",
            camera if camera.isdigit() else "0",
            "--timeout",
            "0",
            "--codec",
            "mjpeg",
            "--width",
            str(width),
            "--height",
            str(height),
            "--framerate",
            str(fps),
            "--nopreview",
            "--flush",
            "--output",
            "-",
        ]
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self.buffer = bytearray()

    def read(self, timeout_sec: float = 2.0) -> tuple[bool, np.ndarray | None]:
        if self.process.stdout is None:
            return False, None
        fd = self.process.stdout.fileno()

        deadline = time.perf_counter() + timeout_sec
        while time.perf_counter() < deadline:
            start = self.buffer.find(b"\xff\xd8")
            end = self.buffer.find(b"\xff\xd9", start + 2 if start >= 0 else 0)
            if start >= 0 and end >= 0:
                jpeg = bytes(self.buffer[start : end + 2])
                del self.buffer[: end + 2]
                frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                return frame is not None, frame

            wait = max(0.0, min(0.25, deadline - time.perf_counter()))
            ready, _, _ = select.select([fd], [], [], wait)
            if not ready:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                return False, None
            self.buffer.extend(chunk)
        return False, None

    def release(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()


DEFAULT_CASES = [
    ("imx219_640x480_103", 640, 480, 103),
    ("imx219_1920x1080_47", 1920, 1080, 47),
    ("imx219_1640x1232_41", 1640, 1232, 41),
    ("imx219_3280x2464_21", 3280, 2464, 21),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="0")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--json-out")
    parser.add_argument(
        "--case",
        action="append",
        help="Case as name,width,height,fps. May be repeated. Defaults to imx219 modes.",
    )
    return parser.parse_args()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100.0
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    weight = index - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "mean_ms": round(statistics.fmean(values), 4),
        "median_ms": round(statistics.median(values), 4),
        "p90_ms": round(percentile(values, 90), 4),
        "p95_ms": round(percentile(values, 95), 4),
        "p99_ms": round(percentile(values, 99), 4),
        "min_ms": round(min(values), 4),
        "max_ms": round(max(values), 4),
        "std_ms": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
    }


def parse_cases(items: list[str] | None) -> list[tuple[str, int, int, int]]:
    if not items:
        return DEFAULT_CASES
    cases = []
    for item in items:
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Bad --case {item!r}; expected name,width,height,fps")
        name, width, height, fps = parts
        cases.append((name, int(width), int(height), int(fps)))
    return cases


def camera_modes() -> str:
    try:
        proc = subprocess.run(
            ["rpicam-hello", "--list-cameras"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return (proc.stdout + proc.stderr).strip()
    except Exception as exc:
        return f"rpicam-hello failed: {exc!r}"


def measure_case(
    camera: str,
    name: str,
    width: int,
    height: int,
    fps: int,
    seconds: float,
    warmup: int,
) -> dict[str, Any]:
    source = RpicamMjpegFrameSourceQuiet(camera, width, height, fps)
    read_ms: list[float] = []
    frame_intervals_ms: list[float] = []
    shapes: dict[str, int] = {}
    frames = 0
    failed_reads = 0
    first_frame_ms = None
    start = time.perf_counter()
    last_frame_time = None

    try:
        for _ in range(warmup):
            ok, _frame = source.read()
            if not ok:
                failed_reads += 1

        begin = time.perf_counter()
        end = begin + seconds
        while time.perf_counter() < end:
            t0 = time.perf_counter()
            ok, frame = source.read()
            t1 = time.perf_counter()
            if not ok or frame is None:
                failed_reads += 1
                continue
            if first_frame_ms is None:
                first_frame_ms = (t1 - start) * 1000.0
            frames += 1
            read_ms.append((t1 - t0) * 1000.0)
            now = t1
            if last_frame_time is not None:
                frame_intervals_ms.append((now - last_frame_time) * 1000.0)
            last_frame_time = now
            shapes[str(tuple(frame.shape))] = shapes.get(str(tuple(frame.shape)), 0) + 1
        elapsed = time.perf_counter() - begin
    finally:
        source.release()

    return {
        "name": name,
        "camera": camera,
        "requested": {"width": width, "height": height, "fps": fps},
        "frames": frames,
        "failed_reads": failed_reads,
        "elapsed_sec": round(elapsed, 4),
        "measured_fps": round(frames / elapsed if elapsed else 0.0, 3),
        "first_frame_ms": round(first_frame_ms, 3) if first_frame_ms is not None else None,
        "frame_shapes": shapes,
        "read_ms": summarize(read_ms),
        "frame_interval_ms": summarize(frame_intervals_ms),
    }


def main() -> int:
    args = parse_args()
    cases = parse_cases(args.case)
    result = {
        "camera_modes": camera_modes(),
        "seconds": args.seconds,
        "warmup": args.warmup,
        "cases": [],
    }
    for case in cases:
        item = measure_case(args.camera, *case, seconds=args.seconds, warmup=args.warmup)
        result["cases"].append(item)
        print(json.dumps(item, ensure_ascii=False))
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
