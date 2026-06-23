#!/usr/bin/env python3
"""Compact 800x480 dashboard for small Raspberry Pi touch displays."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pi_realtime_rps import (
    DEFAULT_LABELS,
    StableState,
    load_interpreter,
    make_event,
    open_frame_source,
    predict,
    preprocess_frame,
    update_stable_state,
)


WINDOW_NAME = "RPS Compact Dashboard"


@dataclass
class RuntimeStats:
    frame_count: int = 0
    loop_fps: float = 0.0
    inference_ms: float = 0.0
    last_event: dict[str, Any] | None = None
    fps_history: deque[float] = field(default_factory=lambda: deque(maxlen=80))
    confidence_history: deque[float] = field(default_factory=lambda: deque(maxlen=80))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact RPS dashboard for 800x480 displays.")
    parser.add_argument("--model", required=True, help="Path to .tflite model.")
    parser.add_argument("--camera", default="0")
    parser.add_argument("--camera-backend", choices=("auto", "opencv", "rpicam"), default="auto")
    parser.add_argument("--capture-width", type=int, default=640)
    parser.add_argument("--capture-height", type=int, default=480)
    parser.add_argument("--capture-fps", type=int, default=58)
    parser.add_argument("--camera-buffer-size", type=int, default=1)
    parser.add_argument(
        "--interpreter-backend",
        choices=("auto", "tflite_runtime", "tensorflow"),
        default="tensorflow",
    )
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--stable-frames", type=int, default=8)
    parser.add_argument("--cooldown", type=float, default=1.5)
    parser.add_argument("--roi-scale", type=float, default=1.0)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--display-width", type=int, default=800)
    parser.add_argument("--display-height", type=int, default=480)
    parser.add_argument("--fullscreen", action="store_true")
    return parser.parse_args()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def put_text(
    image: np.ndarray,
    text: str,
    pos: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = (238, 242, 248),
    thickness: int = 2,
) -> None:
    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_score_bar(
    image: np.ndarray,
    label: str,
    value: float,
    x: int,
    y: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    cv2.rectangle(image, (x, y), (x + width, y + height), (50, 56, 66), -1)
    fill = int(width * clamp(value, 0.0, 1.0))
    cv2.rectangle(image, (x, y), (x + fill, y + height), color, -1)
    cv2.rectangle(image, (x, y), (x + width, y + height), (130, 140, 154), 2)
    put_text(image, label.upper(), (x, y - 8), 0.54, (238, 242, 248), 2)
    put_text(image, f"{value:.2f}", (x + width - 64, y + height - 8), 0.54, (255, 255, 255), 2)


def draw_history(image: np.ndarray, values: deque[float], rect: tuple[int, int, int, int], max_value: float, color: tuple[int, int, int]) -> None:
    x, y, w, h = rect
    cv2.rectangle(image, (x, y), (x + w, y + h), (35, 41, 50), -1)
    cv2.rectangle(image, (x, y), (x + w, y + h), (105, 116, 130), 1)
    vals = list(values)
    if len(vals) < 2:
        return
    pts = []
    for idx, value in enumerate(vals):
        px = x + int(idx * (w - 1) / max(1, len(vals) - 1))
        py = y + h - 5 - int(clamp(value / max_value, 0.0, 1.0) * (h - 10))
        pts.append((px, py))
    for p1, p2 in zip(pts, pts[1:]):
        cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)


def scaled_roi(roi: tuple[int, int, int, int], sx: float, sy: float, ox: int, oy: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    return (
        ox + int(x1 * sx),
        oy + int(y1 * sy),
        ox + int(x2 * sx),
        oy + int(y2 * sy),
    )


def draw_dashboard(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    labels: list[str],
    scores: np.ndarray,
    label: str,
    display_label: str,
    confidence: float,
    threshold: float,
    state: StableState,
    stats: RuntimeStats,
    camera_backend: str,
    model_path: str,
    interpreter_name: str,
    display_width: int,
    display_height: int,
) -> np.ndarray:
    canvas = np.full((display_height, display_width, 3), (20, 24, 30), dtype=np.uint8)
    left_w = int(display_width * 0.54)
    right_x = left_w
    right_w = display_width - right_x
    video_h = int(left_w * 0.75)
    if video_h > display_height - 92:
        video_h = display_height - 92
        left_w = int(video_h * 4 / 3)
        right_x = left_w
        right_w = display_width - right_x

    frame_view = cv2.resize(frame, (left_w, video_h), interpolation=cv2.INTER_AREA)
    canvas[0:video_h, 0:left_w] = frame_view
    sx = left_w / max(1, frame.shape[1])
    sy = video_h / max(1, frame.shape[0])
    x1, y1, x2, y2 = scaled_roi(roi, sx, sy, 0, 0)
    ok_color = (70, 220, 110)
    warn_color = (50, 170, 255)
    box_color = ok_color if confidence >= threshold else warn_color
    cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 3)

    cv2.rectangle(canvas, (0, video_h), (left_w, display_height), (28, 34, 42), -1)
    put_text(canvas, f"FPS {stats.loop_fps:4.1f}", (18, video_h + 38), 0.82, (80, 235, 120), 2)
    put_text(canvas, f"infer {stats.inference_ms:4.2f} ms", (18, video_h + 76), 0.72, (235, 245, 255), 2)
    put_text(canvas, "q/ESC quit  m mirror", (18, display_height - 16), 0.48, (210, 218, 228), 1)

    cv2.rectangle(canvas, (right_x, 0), (display_width, display_height), (30, 36, 45), -1)
    cv2.line(canvas, (right_x, 0), (right_x, display_height), (95, 105, 120), 2)

    x = right_x + 22
    bar_w = max(220, right_w - 44)
    result = display_label.upper()
    put_text(canvas, "RPS", (x, 36), 0.72, (255, 255, 255), 2)
    put_text(canvas, result, (x, 86), 1.12, box_color, 3)
    put_text(canvas, f"conf {confidence:.3f}", (x, 124), 0.62, (235, 245, 255), 2)
    put_text(canvas, f"stable {state.count} / {threshold:.2f}", (x, 154), 0.54, (210, 218, 228), 2)

    bar_colors = [(80, 190, 255), (90, 225, 130), (230, 170, 80)]
    y = 212
    for idx, name in enumerate(labels):
        value = float(scores[idx]) if idx < len(scores) else 0.0
        draw_score_bar(canvas, name, value, x, y, bar_w, 30, bar_colors[idx % len(bar_colors)])
        y += 66

    put_text(canvas, "FPS", (x, y + 10), 0.48, (200, 210, 222), 1)
    draw_history(canvas, stats.fps_history, (x + 52, y - 10, bar_w - 52, 36), 70.0, (90, 225, 130))
    y += 52
    put_text(canvas, "CONF", (x, y + 10), 0.48, (200, 210, 222), 1)
    draw_history(canvas, stats.confidence_history, (x + 52, y - 10, bar_w - 52, 36), 1.0, (80, 190, 255))

    model_name = Path(model_path).name
    footer = f"{camera_backend} | {interpreter_name}"
    put_text(canvas, footer[:32], (x, display_height - 38), 0.43, (180, 190, 204), 1)
    put_text(canvas, model_name[:34], (x, display_height - 15), 0.38, (160, 170, 184), 1)
    return canvas


def main() -> int:
    args = parse_args()
    labels = [part.strip() for part in args.labels.split(",") if part.strip()]
    if len(labels) != 3:
        raise ValueError("--labels must contain exactly 3 comma-separated labels.")

    interpreter, input_details, output_details = load_interpreter(
        args.model,
        args.interpreter_backend,
        args.num_threads,
    )
    interpreter_name = "tensorflow" if args.interpreter_backend == "tensorflow" else interpreter.__class__.__module__.split(".")[0]

    frame_source, camera_backend = open_frame_source(args)
    state = StableState()
    stats = RuntimeStats()
    threshold = float(args.threshold)
    roi_scale = float(args.roi_scale)
    mirror = bool(args.mirror)
    previous_time = time.perf_counter()

    print("[compact-dashboard]")
    print("model:", args.model)
    print("camera:", camera_backend)
    print("display:", f"{args.display_width}x{args.display_height}")
    print("threshold:", threshold)
    print("press q or ESC to quit")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.display_width, args.display_height)
    if args.fullscreen:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            ok, frame = frame_source.read()
            if not ok or frame is None:
                print("camera read failed")
                time.sleep(0.1)
                continue

            if mirror:
                frame = cv2.flip(frame, 1)

            input_tensor, roi = preprocess_frame(frame, input_details, roi_scale)
            infer_start = time.perf_counter()
            label_id, confidence, scores = predict(interpreter, input_details, output_details, input_tensor)
            stats.inference_ms = (time.perf_counter() - infer_start) * 1000.0

            label = labels[label_id]
            display_label = "uncertain" if confidence < threshold else label
            update_stable_state(state, label, confidence, threshold)

            now = time.perf_counter()
            frame_time = now - previous_time
            previous_time = now
            if frame_time > 0:
                instant_fps = 1.0 / frame_time
                stats.loop_fps = 0.9 * stats.loop_fps + 0.1 * instant_fps if stats.loop_fps else instant_fps

            stats.frame_count += 1
            stats.fps_history.append(stats.loop_fps)
            stats.confidence_history.append(confidence)

            wall_time = time.time()
            if state.count >= args.stable_frames and wall_time - state.last_event_time >= args.cooldown:
                event = make_event(label, confidence, labels)
                stats.last_event = event
                print(json.dumps(event, ensure_ascii=False))
                state.last_event_time = wall_time
                state.count = 0

            dashboard = draw_dashboard(
                frame=frame,
                roi=roi,
                labels=labels,
                scores=scores,
                label=label,
                display_label=display_label,
                confidence=confidence,
                threshold=threshold,
                state=state,
                stats=stats,
                camera_backend=camera_backend,
                model_path=args.model,
                interpreter_name=interpreter_name,
                display_width=args.display_width,
                display_height=args.display_height,
            )

            cv2.imshow(WINDOW_NAME, dashboard)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("m"):
                mirror = not mirror
            elif key in (ord("+"), ord("=")):
                threshold = clamp(threshold + 0.02, 0.0, 0.99)
            elif key in (ord("-"), ord("_")):
                threshold = clamp(threshold - 0.02, 0.0, 0.99)
            elif key == ord("["):
                roi_scale = clamp(roi_scale - 0.05, 0.2, 1.0)
            elif key == ord("]"):
                roi_scale = clamp(roi_scale + 0.05, 0.2, 1.0)
            elif key == ord("r"):
                state = StableState()
                stats.last_event = None
    finally:
        frame_source.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
