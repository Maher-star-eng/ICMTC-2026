# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   23/06/2026

"""Tests for infer.py.

All tests are hardware-free: no camera, no GPU, and no real Ultralytics
weights. A fake YOLO model (returning synthetic `Results`-like objects) is
dependency-injected via `yolo_factory`, and images are real small PNGs
written to `tmp_path` with OpenCV so the actual decode/draw/save path is
exercised, not a mock of it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vision.infer import (
    DetectionResult,
    InferConfig,
    InferError,
    class_color,
    collect_images,
    extract_detections,
    next_run_dir,
    parse_args,
    render_detection_summary,
    render_run_summary,
    resolve_device,
    run_infer,
    validate_source,
    validate_weights,
    write_label_txt,
)

IMAGE_SHAPE = (100, 120, 3)  # H, W, 3


def make_image(path: Path, shape: tuple[int, int, int] = IMAGE_SHAPE) -> Path:
    frame = np.full(shape, 127, dtype=np.uint8)
    cv2.imwrite(str(path), frame)
    return path


def fake_results(detections: list[DetectionResult]):
    """Build one Ultralytics-Results-shaped object from DetectionResults."""
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
        self.task = "detect"
        self._scripted = scripted
        self.predict_calls: list[dict] = []

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return [fake_results(self._scripted)]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_validate_weights_missing_raises(tmp_path: Path):
    with pytest.raises(InferError, match="weights file not found"):
        validate_weights(tmp_path / "nope.pt")


def test_validate_weights_directory_raises(tmp_path: Path):
    with pytest.raises(InferError, match="not a file"):
        validate_weights(tmp_path)


def test_validate_source_missing_raises(tmp_path: Path):
    with pytest.raises(InferError, match="source not found"):
        validate_source(tmp_path / "nope.jpg")


def test_validate_source_accepts_existing_file_or_dir(tmp_path: Path):
    image = make_image(tmp_path / "a.jpg")
    assert validate_source(image) == image
    assert validate_source(tmp_path) == tmp_path


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------
def test_collect_images_single_file(tmp_path: Path):
    image = make_image(tmp_path / "a.jpg")
    assert collect_images(image) == [image]


def test_collect_images_single_file_rejects_non_image(tmp_path: Path):
    bogus = tmp_path / "notes.txt"
    bogus.write_text("hello")
    with pytest.raises(InferError, match="not a recognized image type"):
        collect_images(bogus)


def test_collect_images_folder_is_sorted_and_filters_non_images(tmp_path: Path):
    make_image(tmp_path / "b.png")
    make_image(tmp_path / "a.jpg")
    (tmp_path / "readme.txt").write_text("skip me")
    images = collect_images(tmp_path)
    assert [p.name for p in images] == ["a.jpg", "b.png"]


def test_collect_images_empty_folder_raises(tmp_path: Path):
    with pytest.raises(InferError, match="no images found"):
        collect_images(tmp_path)


# ---------------------------------------------------------------------------
# Device resolution / class color
# ---------------------------------------------------------------------------
def test_resolve_device_auto_is_none():
    assert resolve_device("auto") is None
    assert resolve_device("AUTO") is None


def test_resolve_device_passthrough():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:0") == "cuda:0"


def test_class_color_is_deterministic_and_distinct():
    assert class_color(3) == class_color(3)
    assert class_color(1) != class_color(2)
    for color in (class_color(0), class_color(5), class_color(42)):
        assert all(0 <= channel <= 255 for channel in color)


# ---------------------------------------------------------------------------
# Detection extraction
# ---------------------------------------------------------------------------
def test_extract_detections_empty_boxes_returns_empty_list():
    result = fake_results([])
    assert extract_detections(result, {0: "Egypt"}) == []


def test_extract_detections_maps_class_names():
    scripted = [DetectionResult(box=(1, 2, 30, 40), class_id=0, class_name="Egypt", score=0.9)]
    result = fake_results(scripted)
    out = extract_detections(result, {0: "Egypt"})
    assert len(out) == 1
    assert out[0].class_name == "Egypt"
    assert out[0].class_id == 0
    assert out[0].score == pytest.approx(0.9)


def test_extract_detections_unknown_class_id_falls_back_to_str():
    scripted = [DetectionResult(box=(0, 0, 10, 10), class_id=7, class_name="", score=0.5)]
    result = fake_results(scripted)
    out = extract_detections(result, {})
    assert out[0].class_name == "7"


# ---------------------------------------------------------------------------
# Summary rendering (requirement 6 + 9)
# ---------------------------------------------------------------------------
def test_render_detection_summary_no_detections():
    assert render_detection_summary([]) == "No flags detected."


def test_render_detection_summary_sorted_by_confidence():
    detections = [
        DetectionResult(box=(0, 0, 1, 1), class_id=0, class_name="Brazil", score=0.94),
        DetectionResult(box=(0, 0, 1, 1), class_id=1, class_name="Egypt", score=0.98),
        DetectionResult(box=(0, 0, 1, 1), class_id=2, class_name="France", score=0.97),
    ]
    summary = render_detection_summary(detections)
    lines = summary.splitlines()
    assert lines[0] == "Detected 3 objects"
    assert lines[1] == ""
    names_in_order = [line.split()[0] for line in lines[2:]]
    assert names_in_order == ["Egypt", "France", "Brazil"]


# ---------------------------------------------------------------------------
# Output directory (requirement 5: never overwrite)
# ---------------------------------------------------------------------------
def test_next_run_dir_increments(tmp_path: Path):
    first = next_run_dir(tmp_path, stem="exp")
    second = next_run_dir(tmp_path, stem="exp")
    third = next_run_dir(tmp_path, stem="exp")
    assert first.name == "exp"
    assert second.name == "exp2"
    assert third.name == "exp3"
    assert first.exists() and second.exists() and third.exists()


# ---------------------------------------------------------------------------
# Label file writing (requirement: --save-txt / --save-conf)
# ---------------------------------------------------------------------------
def test_write_label_txt_normalizes_and_includes_conf(tmp_path: Path):
    detections = [DetectionResult(box=(10, 10, 30, 50), class_id=2, class_name="X", score=0.75)]
    out = tmp_path / "a.txt"
    write_label_txt(out, detections, image_shape=(100, 100), save_conf=True)
    line = out.read_text().strip()
    parts = line.split()
    assert parts[0] == "2"
    assert len(parts) == 6
    assert float(parts[5]) == pytest.approx(0.75)


def test_write_label_txt_empty_detections_writes_empty_file(tmp_path: Path):
    out = tmp_path / "empty.txt"
    write_label_txt(out, [], image_shape=(100, 100), save_conf=False)
    assert out.read_text() == ""


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------
def test_render_run_summary_computes_average_time_and_fps(tmp_path: Path):
    from vision.infer import ImageOutcome

    outcomes = [
        ImageOutcome(image_path=tmp_path / "a.jpg", detections=(), infer_ms=10.0, image_shape=(1, 1)),
        ImageOutcome(image_path=tmp_path / "b.jpg", detections=(), infer_ms=30.0, image_shape=(1, 1)),
    ]
    summary = render_run_summary(outcomes, tmp_path)
    assert "Images processed : 2" in summary
    assert "Average inference time : 20.0 ms" in summary
    assert "Average FPS : 50.0" in summary


# ---------------------------------------------------------------------------
# End-to-end orchestration (fake model, real image files, real cv2/np I/O)
# ---------------------------------------------------------------------------
def test_run_infer_single_image_with_detections(tmp_path: Path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"not-a-real-model")
    image = make_image(tmp_path / "a.jpg")
    project = tmp_path / "runs"

    scripted = [DetectionResult(box=(5, 5, 50, 50), class_id=0, class_name="Egypt", score=0.98)]
    model = FakeYoloModel(names={0: "Egypt"}, scripted=scripted)

    config = InferConfig(weights=weights, source=image, project=project)
    outcomes = run_infer(config, yolo_factory=lambda w: model)

    assert len(outcomes) == 1
    assert len(outcomes[0].detections) == 1
    run_dir = project / "exp"
    assert (run_dir / "a.jpg").exists()
    assert model.predict_calls[0]["conf"] == config.conf
    assert model.predict_calls[0]["device"] is None  # ASSUMPTION I5: "auto" -> None


def test_run_infer_folder_processes_every_image_and_never_overwrites(tmp_path: Path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"not-a-real-model")
    source_dir = tmp_path / "images"
    source_dir.mkdir()
    make_image(source_dir / "a.jpg")
    make_image(source_dir / "b.jpg")
    project = tmp_path / "runs"

    model = FakeYoloModel(names={0: "Egypt"}, scripted=[])
    config = InferConfig(weights=weights, source=source_dir, project=project)

    first_outcomes = run_infer(config, yolo_factory=lambda w: model)
    second_outcomes = run_infer(config, yolo_factory=lambda w: model)

    assert len(first_outcomes) == 2
    assert len(second_outcomes) == 2
    assert (project / "exp").exists()
    assert (project / "exp2").exists()  # second run does not overwrite the first


def test_run_infer_no_detections_prints_no_flags_message(tmp_path: Path, capsys):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"not-a-real-model")
    image = make_image(tmp_path / "a.jpg")
    model = FakeYoloModel(names={0: "Egypt"}, scripted=[])

    config = InferConfig(weights=weights, source=image, project=tmp_path / "runs")
    run_infer(config, yolo_factory=lambda w: model)

    assert "No flags detected." in capsys.readouterr().out


def test_run_infer_save_txt_writes_label_files(tmp_path: Path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"not-a-real-model")
    image = make_image(tmp_path / "a.jpg")
    scripted = [DetectionResult(box=(1, 1, 20, 20), class_id=0, class_name="Egypt", score=0.9)]
    model = FakeYoloModel(names={0: "Egypt"}, scripted=scripted)

    config = InferConfig(
        weights=weights, source=image, project=tmp_path / "runs", save_txt=True, save_conf=True,
    )
    run_infer(config, yolo_factory=lambda w: model)

    label_path = tmp_path / "runs" / "exp" / "a.txt"
    assert label_path.exists()
    assert len(label_path.read_text().split()) == 6


def test_run_infer_raises_for_missing_weights(tmp_path: Path):
    image = make_image(tmp_path / "a.jpg")
    config = InferConfig(weights=tmp_path / "missing.pt", source=image)
    with pytest.raises(InferError, match="weights file not found"):
        run_infer(config, yolo_factory=lambda w: FakeYoloModel({}, []))


def test_run_infer_raises_for_missing_source(tmp_path: Path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")
    config = InferConfig(weights=weights, source=tmp_path / "missing.jpg")
    with pytest.raises(InferError, match="source not found"):
        run_infer(config, yolo_factory=lambda w: FakeYoloModel({}, []))


def test_run_infer_skips_unreadable_image_in_folder(tmp_path: Path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")
    source_dir = tmp_path / "images"
    source_dir.mkdir()
    make_image(source_dir / "good.jpg")
    # A file with an image suffix but corrupt/non-image bytes: cv2.imread -> None.
    (source_dir / "bad.jpg").write_bytes(b"not an image")

    model = FakeYoloModel(names={0: "Egypt"}, scripted=[])
    config = InferConfig(weights=weights, source=source_dir, project=tmp_path / "runs")
    outcomes = run_infer(config, yolo_factory=lambda w: model)

    assert len(outcomes) == 1
    assert outcomes[0].image_path.name == "good.jpg"


def test_run_infer_raises_when_every_image_is_unreadable(tmp_path: Path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")

    model = FakeYoloModel(names={0: "Egypt"}, scripted=[])
    config = InferConfig(weights=weights, source=bad, project=tmp_path / "runs")
    with pytest.raises(InferError, match="no image could be processed"):
        run_infer(config, yolo_factory=lambda w: model)


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
def test_parse_args_defaults():
    config = parse_args(["--source", "test_images/"])
    assert config.source == Path("test_images/")
    assert config.conf == 0.25
    assert config.imgsz == 640
    assert config.device == "auto"
    assert config.save_txt is False
    assert config.save_conf is False


def test_parse_args_overrides():
    config = parse_args(
        [
            "--weights", "best.pt",
            "--source", "img.jpg",
            "--conf", "0.5",
            "--imgsz", "1280",
            "--device", "cpu",
            "--save-txt",
            "--save-conf",
        ]
    )
    assert config.weights == Path("best.pt")
    assert config.conf == 0.5
    assert config.imgsz == 1280
    assert config.device == "cpu"
    assert config.save_txt is True
    assert config.save_conf is True
