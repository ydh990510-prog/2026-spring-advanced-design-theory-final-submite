#!/usr/bin/env python3
"""Run real-time rock-paper-scissors inference with a TFLite model."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import time
import warnings
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any

import cv2
import numpy as np

warnings.filterwarnings("ignore", message=".*tf\\.lite\\.Interpreter is deprecated.*")
warnings.filterwarnings("ignore", message=".*smallest subnormal.*")


DEFAULT_LABELS = ["scissors", "rock", "paper"]
WIN_CASES = {
    ("rock", "scissors"),
    ("scissors", "paper"),
    ("paper", "rock"),
}


@dataclass
class StableState:
    last_label: str | None = None
    count: int = 0
    last_event_time: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an RPS TFLite model on Raspberry Pi camera frames."
    )
    parser.add_argument("--model", required=True, help="Path to .tflite model.")
    parser.add_argument(
        "--camera",
        default="0",
        help="Camera index/path. Examples: 0, 1, /dev/video0. Default: 0",
    )
    parser.add_argument(
        "--camera-backend",
        choices=("auto", "opencv", "rpicam"),
        default="auto",
        help="Camera backend. Use rpicam for Raspberry Pi libcamera cameras.",
    )
    parser.add_argument("--capture-width", type=int, default=640, help="Camera capture width.")
    parser.add_argument("--capture-height", type=int, default=480, help="Camera capture height.")
    parser.add_argument("--capture-fps", type=int, default=30, help="Camera capture FPS.")
    parser.add_argument(
        "--camera-buffer-size",
        type=int,
        default=1,
        help="OpenCV camera buffer size. Use 4 for stable 30 FPS on the tested USB camera.",
    )
    parser.add_argument(
        "--interpreter-backend",
        choices=("auto", "tflite_runtime", "tensorflow"),
        default="auto",
        help="TFLite interpreter backend. Use tensorflow to enable the tested full-int8 XNNPACK path.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Interpreter thread count. Default lets the runtime decide.",
    )
    parser.add_argument(
        "--status-every",
        type=int,
        default=1,
        help="In headless mode, print status every N frames. Use 0 to disable status lines.",
    )
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated labels in model output order. Default: scissors,rock,paper",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Minimum confidence for a frame to count as reliable.",
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=8,
        help="Consecutive matching frames required for final decision.",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=1.5,
        help="Seconds to wait after emitting a game event.",
    )
    parser.add_argument(
        "--roi-scale",
        type=float,
        default=1.0,
        help="Center crop scale from 0.2 to 1.0. Use 1.0 for full frame.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not open cv2.imshow window; print JSON/status only.",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror camera frame horizontally for easier interaction.",
    )
    return parser.parse_args()


class OpenCvFrameSource:
    def __init__(
        self,
        camera: str,
        width: int,
        height: int,
        fps: int,
        buffer_size: int = 1,
    ) -> None:
        self.camera = camera
        camera_id: int | str = int(camera) if camera.isdigit() else camera
        self.cap = cv2.VideoCapture(camera_id)
        if width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        if buffer_size > 0:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

    def is_opened(self) -> bool:
        return bool(self.cap.isOpened())

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self.cap.read()
        return ok, frame if ok else None

    def release(self) -> None:
        self.cap.release()


class RpicamMjpegFrameSource:
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
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self.buffer = bytearray()

    def is_opened(self) -> bool:
        return self.process.poll() is None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.process.stdout is None:
            return False, None

        while True:
            start = self.buffer.find(b"\xff\xd8")
            end = self.buffer.find(b"\xff\xd9", start + 2 if start >= 0 else 0)
            if start >= 0 and end >= 0:
                jpeg = bytes(self.buffer[start : end + 2])
                del self.buffer[: end + 2]
                frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                return frame is not None, frame

            chunk = self.process.stdout.read(4096)
            if not chunk:
                return False, None
            self.buffer.extend(chunk)

    def release(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()


def camera_diagnostics() -> str:
    lines = ["Camera diagnostics:"]
    video_devices = sorted(glob("/dev/video*"))
    lines.append(f"- /dev/video*: {video_devices if video_devices else 'none'}")
    lines.append(f"- rpicam-vid: {shutil.which('rpicam-vid') or 'not found'}")
    lines.append(f"- rpicam-hello: {shutil.which('rpicam-hello') or 'not found'}")
    if shutil.which("rpicam-hello"):
        try:
            result = subprocess.run(
                ["rpicam-hello", "--list-cameras"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = (result.stdout + result.stderr).strip()
            lines.append(f"- rpicam-hello --list-cameras: {output or 'no output'}")
        except Exception as exc:
            lines.append(f"- rpicam-hello --list-cameras failed: {exc}")
    return "\n".join(lines)


def open_frame_source(args: argparse.Namespace) -> tuple[Any, str]:
    errors = []

    if args.camera_backend in ("auto", "opencv"):
        source = OpenCvFrameSource(
            args.camera,
            args.capture_width,
            args.capture_height,
            args.capture_fps,
            getattr(args, "camera_buffer_size", 1),
        )
        if source.is_opened():
            return source, f"opencv:{args.camera}"
        source.release()
        errors.append(f"opencv:{args.camera} not available")

    if args.camera_backend in ("auto", "rpicam"):
        if shutil.which("rpicam-vid"):
            source = RpicamMjpegFrameSource(
                args.camera,
                args.capture_width,
                args.capture_height,
                args.capture_fps,
            )
            ok, frame = source.read()
            if ok and frame is not None:
                source.buffer[:0] = cv2.imencode(".jpg", frame)[1].tobytes()
                return source, f"rpicam:{args.camera}"
            source.release()
            errors.append("rpicam-vid did not provide a frame")
        else:
            errors.append("rpicam-vid not found")

    raise RuntimeError(
        "Cannot open camera.\n"
        + "\n".join(f"- {error}" for error in errors)
        + "\n"
        + camera_diagnostics()
    )


def judge(user: str, computer: str) -> str:
    if user == computer:
        return "draw"
    if (user, computer) in WIN_CASES:
        return "win"
    return "lose"


def interpreter_classes(backend: str = "auto") -> list[tuple[str, Any]]:
    classes = []
    if backend in ("auto", "tflite_runtime"):
        try:
            from tflite_runtime.interpreter import Interpreter

            classes.append(("tflite_runtime", Interpreter))
        except ImportError:
            pass

    if backend in ("auto", "tensorflow"):
        try:
            from tensorflow.lite.python.interpreter import Interpreter

            classes.append(("tensorflow", Interpreter))
        except ImportError:
            pass

    if not classes:
        raise SystemExit(
            f"TFLite interpreter backend '{backend}' not found. "
            "Install tflite-runtime or tensorflow."
        )
    return classes


def create_interpreter(
    model_path: Path,
    backend: str = "auto",
    num_threads: int | None = None,
) -> tuple[Any, str]:
    errors = []
    for name, Interpreter in interpreter_classes(backend):
        try:
            kwargs = {"model_path": str(model_path)}
            if num_threads is not None:
                kwargs["num_threads"] = num_threads
            return Interpreter(**kwargs), name
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("Cannot load TFLite model:\n" + "\n".join(errors))


def load_interpreter(
    model_path: str,
    backend: str = "auto",
    num_threads: int | None = None,
) -> tuple[Any, dict, dict]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path.resolve()}")

    interpreter, interpreter_name = create_interpreter(path, backend, num_threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print("[model]")
    print("path:", path.resolve())
    print("interpreter:", interpreter_name)
    print("requested backend:", backend)
    print("num_threads:", num_threads if num_threads is not None else "runtime default")
    print("input shape:", input_details["shape"])
    print("input dtype:", input_details["dtype"])
    print("input quantization:", input_details.get("quantization"))
    print("output shape:", output_details["shape"])
    print("output dtype:", output_details["dtype"])
    print("output quantization:", output_details.get("quantization"))
    return interpreter, input_details, output_details


def center_crop(frame: np.ndarray, roi_scale: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    roi_scale = min(1.0, max(0.2, roi_scale))
    height, width = frame.shape[:2]
    crop_size = int(min(width, height) * roi_scale)
    x1 = (width - crop_size) // 2
    y1 = (height - crop_size) // 2
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def quantize_input(image: np.ndarray, input_details: dict) -> np.ndarray:
    dtype = input_details["dtype"]
    if dtype == np.float32:
        return image.astype(np.float32)
    if dtype == np.uint8:
        return image.astype(np.uint8)
    if dtype == np.int8:
        scale, zero_point = input_details.get("quantization", (0.0, 0))
        if scale == 0:
            return image.astype(np.int8)
        quantized = np.round(image.astype(np.float32) / scale + zero_point)
        return np.clip(quantized, -128, 127).astype(np.int8)
    return image.astype(dtype)


def preprocess_frame(
    frame: np.ndarray,
    input_details: dict,
    roi_scale: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    input_shape = input_details["shape"]
    input_height = int(input_shape[1])
    input_width = int(input_shape[2])

    crop, roi = center_crop(frame, roi_scale)
    image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_width, input_height), interpolation=cv2.INTER_AREA)
    image = quantize_input(image, input_details)
    image = np.expand_dims(image, axis=0)
    return image, roi


def dequantize_output(output: np.ndarray, output_details: dict) -> np.ndarray:
    dtype = output_details["dtype"]
    if dtype in (np.uint8, np.int8):
        scale, zero_point = output_details.get("quantization", (0.0, 0))
        if scale != 0:
            output = (output.astype(np.float32) - zero_point) * scale
    return np.squeeze(output).astype(np.float32)


def predict(
    interpreter: Any,
    input_details: dict,
    output_details: dict,
    input_tensor: np.ndarray,
) -> tuple[int, float, np.ndarray]:
    interpreter.set_tensor(input_details["index"], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])
    scores = dequantize_output(output, output_details)
    label_id = int(np.argmax(scores))
    confidence = float(scores[label_id])
    return label_id, confidence, scores


def update_stable_state(
    state: StableState,
    label: str,
    confidence: float,
    threshold: float,
) -> None:
    if confidence < threshold:
        state.last_label = None
        state.count = 0
        return

    if label == state.last_label:
        state.count += 1
    else:
        state.last_label = label
        state.count = 1


def make_event(user_label: str, confidence: float, choices: list[str]) -> dict:
    computer = random.choice(choices)
    return {
        "timestamp": time.time(),
        "user": user_label,
        "computer": computer,
        "result": judge(user_label, computer),
        "confidence": round(confidence, 3),
    }


def draw_overlay(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    label: str,
    confidence: float,
    stable_count: int,
    fps: float,
    threshold: float,
) -> None:
    x1, y1, x2, y2 = roi
    color = (0, 180, 0) if confidence >= threshold else (0, 165, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    lines = [
        f"label: {label}",
        f"confidence: {confidence:.3f}",
        f"stable: {stable_count}",
        f"fps: {fps:.1f}",
    ]
    for idx, text in enumerate(lines):
        y = 28 + idx * 28
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (20, 20, 20),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


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

    frame_source, frame_source_name = open_frame_source(args)

    state = StableState()
    fps = 0.0
    frame_count = 0
    previous_time = time.perf_counter()

    print("[run]")
    print("labels:", labels)
    print("threshold:", args.threshold)
    print("stable_frames:", args.stable_frames)
    print("cooldown:", args.cooldown)
    print("camera_backend:", frame_source_name)
    print("press q to quit when using display mode")

    try:
        while True:
            ok, frame = frame_source.read()
            if not ok or frame is None:
                print("camera read failed")
                time.sleep(0.1)
                continue
            frame_count += 1

            if args.mirror:
                frame = cv2.flip(frame, 1)

            input_tensor, roi = preprocess_frame(frame, input_details, args.roi_scale)
            label_id, confidence, _scores = predict(
                interpreter,
                input_details,
                output_details,
                input_tensor,
            )
            label = labels[label_id]
            display_label = "uncertain" if confidence < args.threshold else label
            update_stable_state(state, label, confidence, args.threshold)

            now = time.perf_counter()
            frame_time = now - previous_time
            previous_time = now
            if frame_time > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / frame_time) if fps else 1.0 / frame_time

            wall_time = time.time()
            if (
                state.count >= args.stable_frames
                and wall_time - state.last_event_time >= args.cooldown
            ):
                event = make_event(label, confidence, labels)
                print(json.dumps(event, ensure_ascii=False))
                state.last_event_time = wall_time
                state.count = 0

            if args.headless:
                should_print_status = (
                    args.status_every > 0 and frame_count % args.status_every == 0
                )
                if not should_print_status:
                    continue
                print(
                    f"label={display_label} raw={label} confidence={confidence:.3f} "
                    f"threshold={args.threshold:.2f} stable={state.count} fps={fps:.1f}"
                )
                continue

            draw_overlay(
                frame,
                roi,
                display_label,
                confidence,
                state.count,
                fps,
                args.threshold,
            )
            cv2.imshow("RPS TFLite", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        frame_source.release()
        if not args.headless:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
