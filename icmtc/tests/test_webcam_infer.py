# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   24/06/2026

"""Tests for webcam_infer.py.

All tests are hardware-free: no real camera, no display, and no real
Ultralytics weights. The camera is faked via `capture_factory` (a scripted
frame sequence), the GUI via `show_fn`/`wait_key_fn`, and the model via
`yolo_factory` - mirroring tests/test_infer.py's approach for infer.py.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vision.infer import DetectionResult, InferError
from vision.webcam_infer import (
    CameraError,
    DeviceSelection,
    WebcamConfig,
    draw_overlay,
    open_webcam,
    parse_args,
    process_frame,
    run_webcam_demo,
    select_device,
)

FRAME_SHAPE = (48, 64, 3)


def make_frame() -> np.ndarray:
    return np.full(FRAME_SHAPE, 100, dtype=np.uint8)


def fake_results(detections: list[DetectionResult]):
    if not detections:
        boxes = SimpleNamespace(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            cls=np.zeros((0,), dtype=np.float32),
            conf=np.zeros((0,), dtype=np.float32),
        )
    else:
        boxes = SimpleNamespace(
            xyxy=np.array([d.box for d in detections], dtype=np.float32),
            cls=np.array([d.class_id for d in detections], dtype=np.float32),
            conf=np.array([d.score for d in detections], dtype=np.float32),
        )
    return SimpleNamespace(boxes=boxes)


class FakeYoloModel:
    """Dependency-injected stand-in for `ultralytics.YOLO`."""

    def __init__(self, names: dict[int, str], scripted: list[DetectionResult]):
        self.names = names
        self._scripted = scripted
        self.predict_calls: list[dict] = []

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return [fake_results(self._scripted)]


class FakeCapture:
    """Scripted stand-in for `cv2.VideoCapture`."""

    def __init__(self, frames: list[np.ndarray], *, opened: bool = True, width=640, height=480):
        self._frames = list(frames)
        self._opened = opened
        self._width = width
        self._height = height
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def get(self, propId: int) -> float:  # noqa: N803 - matches cv2's camelCase API
        # Order-independent: just return width for the first call, height for the second.
        return float(self._width if propId == 3 else self._height)

    def release(self) -> None:
        self.released = True


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def test_select_device_explicit_passthrough():
    selection = select_device("cpu")
    assert selection == DeviceSelection(value="cpu", label="cpu")


def test_select_device_auto_falls_back_gracefully_without_torch(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch in this test env")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    selection = select_device("auto")
    assert selection.value is None
    assert "auto" in selection.label


# ---------------------------------------------------------------------------
# Camera opening (requirement: clear error, never a crash)
# ---------------------------------------------------------------------------
def test_open_webcam_raises_camera_error_when_not_opened():
    def factory(index):
        return FakeCapture([], opened=False)

    with pytest.raises(CameraError, match="could not open webcam"):
        open_webcam(0, factory)


def test_open_webcam_returns_capture_when_opened():
    capture = FakeCapture([make_frame()])
    result = open_webcam(0, lambda index: capture)
    assert result is capture


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------
def test_draw_overlay_modifies_frame_in_place():
    frame = make_frame()
    before = frame.copy()
    draw_overlay(frame, fps=30.0, infer_ms=33.3)
    assert not np.array_equal(frame, before)


# ---------------------------------------------------------------------------
# Per-frame processing
# ---------------------------------------------------------------------------
def test_process_frame_returns_detections_and_annotated_copy():
    scripted = [DetectionResult(box=(1, 1, 20, 20), class_id=0, class_name="Egypt", score=0.9)]
    model = FakeYoloModel(names={0: "Egypt"}, scripted=scripted)
    frame = make_frame()
    config = WebcamConfig()
    device = DeviceSelection(value=None, label="auto")

    annotated, detections, infer_ms = process_frame(model, frame, config, device)

    assert len(detections) == 1
    assert detections[0].class_name == "Egypt"
    assert infer_ms >= 0.0
    assert annotated.shape == frame.shape
    assert not np.array_equal(annotated, frame)  # overlay + box were drawn
    assert model.predict_calls[0]["conf"] == config.conf
    assert model.predict_calls[0]["imgsz"] == config.imgsz


# ---------------------------------------------------------------------------
# Full loop orchestration
# ---------------------------------------------------------------------------
def test_run_webcam_demo_processes_frames_until_stream_ends(tmp_path: Path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"not-a-real-model")
    model = FakeYoloModel(names={0: "Egypt"}, scripted=[])
    capture = FakeCapture([make_frame(), make_frame(), make_frame()])

    shown: list[str] = []
    config = WebcamConfig(weights=weights)

    frames = run_webcam_demo(
        config,
        yolo_factory=lambda w: model,
        capture_factory=lambda index: capture,
        show_fn=lambda name, img: shown.append(name),
        wait_key_fn=lambda: -1,  # never 'q'
    )

    assert frames == 3
    assert shown == ["Mission 2 Flag Detector"] * 3
    assert capture.released is True


def test_run_webcam_demo_quits_on_q_key(tmp_path: Path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"not-a-real-model")
    model = FakeYoloModel(names={0: "Egypt"}, scripted=[])
    # Enough frames to run forever if quitting didn't work.
    capture = FakeCapture([make_frame() for _ in range(100)])

    key_presses = iter([-1, -1, ord("q")])
    config = WebcamConfig(weights=weights)

    frames = run_webcam_demo(
        config,
        yolo_factory=lambda w: model,
        capture_factory=lambda index: capture,
        show_fn=lambda name, img: None,
        wait_key_fn=lambda: next(key_presses),
    )

    assert frames == 3
    assert capture.released is True


def test_run_webcam_demo_releases_capture_even_on_predict_error(tmp_path: Path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"not-a-real-model")

    class BoomModel(FakeYoloModel):
        def predict(self, **kwargs):
            raise RuntimeError("boom")

    model = BoomModel(names={0: "Egypt"}, scripted=[])
    capture = FakeCapture([make_frame()])
    config = WebcamConfig(weights=weights)

    with pytest.raises(RuntimeError, match="boom"):
        run_webcam_demo(
            config,
            yolo_factory=lambda w: model,
            capture_factory=lambda index: capture,
            show_fn=lambda name, img: None,
            wait_key_fn=lambda: -1,
        )
    assert capture.released is True  # ASSUMPTION W5: finally always releases


def test_run_webcam_demo_raises_for_missing_weights(tmp_path: Path):
    config = WebcamConfig(weights=tmp_path / "missing.pt")
    with pytest.raises(InferError, match="weights file not found"):
        run_webcam_demo(
            config,
            yolo_factory=lambda w: FakeYoloModel({}, []),
            capture_factory=lambda index: FakeCapture([make_frame()]),
        )


def test_run_webcam_demo_raises_camera_error_when_camera_unavailable(tmp_path: Path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")
    config = WebcamConfig(weights=weights)

    with pytest.raises(CameraError, match="could not open webcam"):
        run_webcam_demo(
            config,
            yolo_factory=lambda w: FakeYoloModel({0: "Egypt"}, []),
            capture_factory=lambda index: FakeCapture([], opened=False),
        )


def test_run_webcam_demo_prints_model_device_and_resolution(tmp_path: Path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")
    config = WebcamConfig(weights=weights, device="cpu")
    printed: list[str] = []

    run_webcam_demo(
        config,
        yolo_factory=lambda w: FakeYoloModel({0: "Egypt"}, []),
        capture_factory=lambda index: FakeCapture([make_frame()], width=1280, height=720),
        show_fn=lambda name, img: None,
        wait_key_fn=lambda: -1,
        output_fn=printed.append,
    )

    joined = "\n".join(printed)
    assert str(weights) in joined
    assert "cpu" in joined
    assert "1280x720" in joined


def test_run_webcam_demo_respects_max_frames(tmp_path: Path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")
    config = WebcamConfig(weights=weights)
    capture = FakeCapture([make_frame() for _ in range(50)])

    frames = run_webcam_demo(
        config,
        yolo_factory=lambda w: FakeYoloModel({0: "Egypt"}, []),
        capture_factory=lambda index: capture,
        show_fn=lambda name, img: None,
        wait_key_fn=lambda: -1,
        max_frames=5,
    )
    assert frames == 5


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
def test_parse_args_defaults():
    config = parse_args([])
    assert config.conf == 0.25
    assert config.imgsz == 640
    assert config.device == "auto"


def test_parse_args_overrides():
    config = parse_args(
        ["--weights", "best.pt", "--conf", "0.4", "--imgsz", "1280", "--device", "mps"]
    )
    assert config.weights == Path("best.pt")
    assert config.conf == 0.4
    assert config.imgsz == 1280
    assert config.device == "mps"
