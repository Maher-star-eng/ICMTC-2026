# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   24/06/2026

"""webcam_infer.py - live MacBook-webcam demo for Mission 2 flag detection.

ICMTC 2026, Mission 2. A simple live demo only: open the default webcam,
run the trained YOLO11n model frame-by-frame, draw boxes/labels/FPS, and
show it in a window until 'q' is pressed. It does NOT train, export, or
modify weights or any other project file.

    python -m vision.webcam_infer \\
        --weights training/runs/mission2/yolo11n_flags/weights/best.pt

================================================================================
ASSUMPTIONS (collected here; also marked inline with # ASSUMPTION: <tag>)
================================================================================

W1  REUSES infer.py
    Detection extraction (`extract_detections`) and drawing
    (`draw_detections`, `DetectionResult`) are imported from infer.py rather
    than re-implemented, so both tools draw boxes/labels identically and a
    fix in one never silently diverges from the other.  `class_color` is not
    imported directly - it is used internally by `draw_detections`, so the
    per-class colour consistency still holds without needing its own import.
    infer.py is a local project module, not a third-party dependency - the
    "Ultralytics/OpenCV/argparse/pathlib only" constraint is about not
    adding new external packages, not about duplicating existing code.

W2  DEVICE SELECTION
    --device auto (default) probes `torch` directly (MPS -> CUDA -> CPU) so
    the exact chosen device can be printed at startup, per the requirement.
    infer.py's own `resolve_device` only maps "auto" to None (letting
    Ultralytics choose silently) which is enough for a one-shot CLI report
    but not for a live demo banner naming MPS/CUDA/CPU. If torch is not
    importable, device selection is left to Ultralytics (value=None) and
    reported as such - never a crash.

W3  CAMERA / GUI SEAM
    `cv2.VideoCapture` and the `cv2.imshow`/`cv2.waitKey` calls are behind
    injectable parameters (`capture_factory`, `show_fn`, `wait_key_fn`) so
    the frame loop itself is unit-testable without a real camera or
    display - tests fake a short frame stream and a scripted key press.

W4  QUIT / END OF STREAM
    The loop exits cleanly on 'q' (case-sensitive lowercase, matching the
    requirement) OR when the camera stops returning frames (ret=False),
    whichever comes first - never on any other keypress.

W5  NEVER CRASH ON A FAILED FRAME
    A single failed `cap.read()` (ret=False mid-stream, e.g. camera
    unplugged) stops the loop gracefully with a printed message, not an
    unhandled exception; the camera is always released and windows always
    destroyed via `finally`, even on error or Ctrl-C.

W6  WEBCAM INDEX
    Always device index 0 (the requirement explicitly says "the default
    webcam"); not exposed as a CLI flag since the brief lists only
    --weights/--conf/--imgsz/--device.

USAGE
-----
    python -m vision.webcam_infer --weights best.pt --conf 0.25 --imgsz 640 --device auto
    Press 'q' in the window to quit.
================================================================================
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

import cv2
import numpy as np

from vision.infer import (
    DetectionResult,
    InferError,
    YoloFactory,
    YoloModel,
    _default_yolo_factory,
    draw_detections,
    extract_detections,
    validate_weights,
)

# ASSUMPTION W6: module-wide defaults, all CLI-overridable except camera index.
_DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parent
    / "training" / "runs" / "mission2" / "yolo11n_flags" / "weights" / "best.pt"
)
_DEFAULT_CONF = 0.25
_DEFAULT_IMGSZ = 640
_DEFAULT_DEVICE = "auto"
_CAMERA_INDEX = 0  # ASSUMPTION W6
_WINDOW_NAME = "Mission 2 Flag Detector"
_QUIT_KEY = "q"  # ASSUMPTION W4
_OVERLAY_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX


class CameraError(Exception):
    """Raised when the webcam cannot be opened or read from."""


# ---------------------------------------------------------------------------
# Device selection (requirement: print selected device MPS/CUDA/CPU). # ASSUMPTION W2
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DeviceSelection:
    """The device value passed to Ultralytics, plus a human-readable label."""

    value: str | None
    label: str


def select_device(requested: str) -> DeviceSelection:
    """Resolve --device into an Ultralytics device value + printable label."""
    if requested.lower() != "auto":
        return DeviceSelection(value=requested, label=requested)

    try:
        import torch
    except ImportError:  # pragma: no cover - depends on local install
        return DeviceSelection(value=None, label="auto (torch unavailable; Ultralytics will choose)")

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and bool(mps.is_available()):
        return DeviceSelection(value="mps", label="Apple MPS")
    if bool(torch.cuda.is_available()):
        return DeviceSelection(value="cuda", label="NVIDIA CUDA")
    return DeviceSelection(value="cpu", label="CPU")


# ---------------------------------------------------------------------------
# Camera seam (dependency-injectable so the loop is testable without hardware).
# ---------------------------------------------------------------------------
class VideoCapture(Protocol):
    """Small Protocol for the `cv2.VideoCapture` surface this module uses."""

    def isOpened(self) -> bool: ...  # noqa: N802 - matches cv2's camelCase API

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def get(self, propId: int) -> float: ...  # noqa: N803 - matches cv2's camelCase API

    def release(self) -> None: ...


CaptureFactory = Callable[[int], VideoCapture]


def open_webcam(index: int, capture_factory: CaptureFactory) -> VideoCapture:
    """Open the webcam at `index`.  Raises CameraError if it cannot be opened."""
    capture = capture_factory(index)
    if capture is None or not capture.isOpened():
        raise CameraError(
            f"could not open webcam (index={index}); check that a camera is "
            "connected and this app has camera permission (macOS: System "
            "Settings -> Privacy & Security -> Camera)."
        )
    return capture


# ---------------------------------------------------------------------------
# Overlay (requirement: FPS + inference time, top-left corner).
# ---------------------------------------------------------------------------
def draw_overlay(frame: np.ndarray, fps: float, infer_ms: float) -> None:
    """Draw FPS and inference-time text in the top-left corner, in place."""
    lines = [f"FPS: {fps:.1f}", f"Inference: {infer_ms:.1f} ms"]
    y = 24
    for line in lines:
        (text_w, text_h), baseline = cv2.getTextSize(line, _OVERLAY_FONT, 0.6, 2)
        cv2.rectangle(frame, (4, y - text_h - 4), (12 + text_w, y + baseline), (0, 0, 0), -1)
        cv2.putText(frame, line, (8, y), _OVERLAY_FONT, 0.6, (0, 255, 0), 2)
        y += text_h + baseline + 12


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class WebcamConfig:
    """Full configuration for one `python webcam_infer.py` run."""

    weights: Path = field(default_factory=lambda: _DEFAULT_WEIGHTS)
    conf: float = _DEFAULT_CONF
    imgsz: int = _DEFAULT_IMGSZ
    device: str = _DEFAULT_DEVICE


def process_frame(
    model: YoloModel, frame: np.ndarray, config: WebcamConfig, device: DeviceSelection,
) -> tuple[np.ndarray, list[DetectionResult], float]:
    """Run one frame through the model; return (annotated_frame, detections, infer_ms)."""
    names = getattr(model, "names", {}) or {}
    start = time.perf_counter()
    results = model.predict(
        source=frame, conf=config.conf, imgsz=config.imgsz, device=device.value, verbose=False,
    )
    infer_ms = (time.perf_counter() - start) * 1000.0

    detections = extract_detections(results[0], names)
    annotated = draw_detections(frame, detections)
    fps = 1000.0 / infer_ms if infer_ms > 0 else 0.0
    draw_overlay(annotated, fps, infer_ms)
    return annotated, detections, infer_ms


def run_webcam_demo(
    config: WebcamConfig,
    *,
    yolo_factory: YoloFactory | None = None,
    capture_factory: CaptureFactory | None = None,
    show_fn: Callable[[str, np.ndarray], None] | None = None,
    wait_key_fn: Callable[[], int] | None = None,
    output_fn: Callable[[str], None] = print,
    camera_index: int = _CAMERA_INDEX,
    max_frames: int | None = None,
) -> int:
    """Run the live webcam demo loop; returns the number of frames processed.

    Raises:
        InferError: if the weights file is missing or the model fails to load.
        CameraError: if the webcam cannot be opened.
    """
    weights = validate_weights(config.weights)
    factory = yolo_factory or _default_yolo_factory
    model = factory(str(weights))
    output_fn(f"Model: {weights}")

    device = select_device(config.device)
    output_fn(f"Device: {device.label}")

    cap_factory = capture_factory or cv2.VideoCapture
    capture = open_webcam(camera_index, cap_factory)

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fn(f"Webcam resolution: {width}x{height}")
    output_fn(f"Press '{_QUIT_KEY}' in the window to quit.")

    display = show_fn or (lambda name, img: cv2.imshow(name, img))
    key_fn = wait_key_fn or (lambda: cv2.waitKey(1) & 0xFF)

    frames_processed = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                output_fn("Webcam stopped returning frames; exiting.")  # ASSUMPTION W5
                break

            annotated, _detections, _infer_ms = process_frame(model, frame, config, device)
            frames_processed += 1
            display(_WINDOW_NAME, annotated)

            if key_fn() == ord(_QUIT_KEY):  # ASSUMPTION W4
                output_fn("Quit key pressed; exiting.")
                break
            if max_frames is not None and frames_processed >= max_frames:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()  # ASSUMPTION W5: always clean up, even on error

    return frames_processed


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> WebcamConfig:
    """Parse CLI arguments into a typed `WebcamConfig`."""
    parser = argparse.ArgumentParser(
        description="Live webcam demo for the Mission 2 flag detector. "
        "Inference only - never trains, exports, or modifies weights."
    )
    parser.add_argument("--weights", default=str(_DEFAULT_WEIGHTS), help="Path to trained .pt weights.")
    parser.add_argument("--conf", type=float, default=_DEFAULT_CONF, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=_DEFAULT_IMGSZ, help="Inference image size.")
    parser.add_argument("--device", default=_DEFAULT_DEVICE, help="'auto', 'cpu', 'cuda', 'mps', or a device id.")
    args = parser.parse_args(argv)

    return WebcamConfig(
        weights=Path(args.weights).expanduser(),
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `python webcam_infer.py`."""
    config = parse_args(argv)
    try:
        run_webcam_demo(config)
    except (InferError, CameraError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI use
    raise SystemExit(main(sys.argv[1:]))
