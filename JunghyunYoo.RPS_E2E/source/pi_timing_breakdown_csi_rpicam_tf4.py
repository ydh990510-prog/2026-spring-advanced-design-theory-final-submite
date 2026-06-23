#!/usr/bin/env python3
"""Run timing breakdown with CSI rpicam capture and TensorFlow TFLite 4 threads."""

from __future__ import annotations

import pi_realtime_rps as realtime
import pi_timing_breakdown as timing
from csi_rpicam_fps_probe import RpicamMjpegFrameSourceQuiet


def open_csi_rpicam(args):
    source = RpicamMjpegFrameSourceQuiet(
        args.camera,
        args.capture_width,
        args.capture_height,
        args.capture_fps,
    )
    ok, frame = source.read()
    if not ok or frame is None:
        source.release()
        raise RuntimeError("rpicam-vid did not provide a frame")
    return source, f"rpicam:{args.camera}:mjpeg"


def load_tensorflow_threads4(model_path: str):
    return realtime.load_interpreter(model_path, backend="tensorflow", num_threads=4)


timing.open_frame_source = open_csi_rpicam
timing.load_interpreter = load_tensorflow_threads4


if __name__ == "__main__":
    raise SystemExit(timing.main())
