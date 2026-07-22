# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   26/06/2026

"""Tests for export.py.

All tests are hardware-free: no camera, no Pi, no real ncnn runtime, and no
real Ultralytics weights. Real ncnn/pt inference is replaced by small stub
tensors of the correct shape (matching the exported YOLOv8/YOLO11 head:
(1, 4+nc, N) with nc=195) and dependency-injected fake YOLO models, so the
verification LOGIC itself is exercised directly rather than assuming a real
runtime is present in CI.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from training import export as export_module
from training.export import (
    ExportConfig,
    ExportError,
    LayoutError,
    compare_detections,
    export_to_ncnn,
    masked_argmax_predictions,
    masked_predictions_to_detections,
    parse_args,
    render_report,
    run_export_and_verify,
    verify_masking,
    verify_output_layout,
)
from vision.detector import letterbox
from vision.tracker import Detection, Tracker

NUM_CLASSES = 195
# _orient_attrs picks the smaller axis as attrs (4+nc); num_anchors must
# comfortably exceed 4+195=199 in every synthetic fixture, or the heuristic
# picks the wrong axis -- same constraint the real exported tensor satisfies
# (a 640 input yields 8400 anchors).
NUM_ANCHORS = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def det(box, class_id=1, score=0.9) -> Detection:
    return Detection(box=box, class_id=class_id, score=score, coord=None)


def make_raw_output(
    num_classes: int = NUM_CLASSES,
    num_anchors: int = 8400,
    *,
    transposed: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """Build a synthetic (1, 4+nc, N) raw model output tensor.

    Boxes are plausible cx,cy,w,h in [0, 640) model-input pixels; class
    scores are small random values in [0, 0.1) so no anchor accidentally
    clears a realistic confidence threshold unless deliberately boosted.
    """
    rng = np.random.default_rng(seed)
    cx = rng.uniform(50, 590, size=num_anchors)
    cy = rng.uniform(50, 590, size=num_anchors)
    w = rng.uniform(10, 60, size=num_anchors)
    h = rng.uniform(10, 60, size=num_anchors)
    boxes = np.stack([cx, cy, w, h], axis=0)  # (4, N)
    scores = rng.uniform(0.0, 0.1, size=(num_classes, num_anchors))
    attrs = np.concatenate([boxes, scores], axis=0)  # (4+nc, N)
    if transposed:
        attrs = attrs.T  # (N, 4+nc)
    return attrs[None, ...].astype(np.float32)


# ---------------------------------------------------------------------------
# 1. Layout check
# ---------------------------------------------------------------------------
def test_layout_accepts_correct_attrs_first_shape():
    raw = make_raw_output(num_classes=195, num_anchors=8400)
    report = verify_output_layout(raw, expected_num_classes=195, input_size=640)

    assert report.raw_shape == (1, 199, 8400)
    assert report.orientation == "attrs_first"
    assert report.num_classes_detected == 195
    assert report.num_anchors == 8400


def test_layout_detects_and_accepts_transposed_shape():
    raw = make_raw_output(num_classes=195, num_anchors=8400, transposed=True)
    report = verify_output_layout(raw, expected_num_classes=195, input_size=640)

    assert report.raw_shape == (1, 8400, 199)
    assert report.orientation == "anchors_first"
    assert report.num_classes_detected == 195
    assert report.num_anchors == 8400


def test_layout_rejects_wrong_num_classes():
    raw = make_raw_output(num_classes=80, num_anchors=8400)  # COCO nc, not 195

    with pytest.raises(LayoutError, match="num_classes"):
        verify_output_layout(raw, expected_num_classes=195, input_size=640)


def test_layout_rejects_shape_matching_neither_axis():
    # Neither axis equals 4+195=199 (an extra/garbled head shape).
    raw = np.zeros((1, 84, 100), dtype=np.float32)

    with pytest.raises(LayoutError, match="does not match expected"):
        verify_output_layout(raw, expected_num_classes=195, input_size=640)


def test_layout_rejects_bad_ndim():
    raw = np.zeros((1, 2, 3, 4), dtype=np.float32)

    with pytest.raises(LayoutError, match="ndim"):
        verify_output_layout(raw, expected_num_classes=195, input_size=640)


# ---------------------------------------------------------------------------
# 2. Masking on a synthetic tensor
# ---------------------------------------------------------------------------
def test_masking_never_emits_a_non_target_class():
    """The global-argmax winner is a non-target class; masking must exclude it."""
    raw = make_raw_output(num_classes=195, num_anchors=NUM_ANCHORS, seed=1)
    target_ids = (3, 17, 42, 88)

    # Force anchor 0's GLOBAL argmax to a non-target class (class 100), well
    # above every target id's score, so an unmasked decode would pick it.
    raw[0, 4 + 100, 0] = 0.99
    for tid in target_ids:
        raw[0, 4 + tid, 0] = 0.2  # still comfortably above conf_threshold

    report = verify_masking(raw, target_ids, conf_threshold=0.1)
    predictions = masked_argmax_predictions(raw, target_ids, conf_threshold=0.1)

    assert report.passed
    assert all(p.class_id in target_ids for p in predictions)
    assert all(p.class_id != 100 for p in predictions)
    # Anchor 0 specifically must be a target id, never class 100.
    assert predictions[0].class_id in target_ids


def test_masking_yields_none_when_no_target_clears_threshold():
    raw = make_raw_output(num_classes=195, num_anchors=NUM_ANCHORS, seed=2)
    target_ids = (3, 17, 42, 88)
    # All scores are in [0, 0.1); nothing clears a high threshold.
    predictions = masked_argmax_predictions(raw, target_ids, conf_threshold=0.9)

    assert predictions == []


def test_masking_rejects_empty_or_out_of_range_target_ids():
    raw = make_raw_output(num_classes=195, num_anchors=NUM_ANCHORS, seed=3)

    with pytest.raises(ValueError, match="non-empty"):
        masked_argmax_predictions(raw, (), conf_threshold=0.25)

    with pytest.raises(ValueError, match="out of range"):
        masked_argmax_predictions(raw, (9999,), conf_threshold=0.25)


def test_masked_argmax_predictions_delegates_to_detectors_canonical_function(monkeypatch):
    """masked_argmax_predictions must call vision.detector's canonical
    "restrict columns, then argmax" function, not a re-derived copy of it -
    this is what prevents export.py's verification and Detector's live
    decode from silently diverging again (see export.py ASSUMPTION E10 /
    detector.py ASSUMPTION D13).
    """
    raw = make_raw_output(num_classes=195, num_anchors=NUM_ANCHORS, seed=4)
    target_ids = (3, 17, 42, 88)

    real_masked_class_argmax = export_module._masked_class_argmax
    calls = []

    def spy(class_scores, target_class_ids):
        calls.append((class_scores.shape, tuple(target_class_ids)))
        return real_masked_class_argmax(class_scores, target_class_ids)

    monkeypatch.setattr(export_module, "_masked_class_argmax", spy)

    masked_argmax_predictions(raw, target_ids, conf_threshold=0.05)

    assert len(calls) == 1
    scores_shape, called_target_ids = calls[0]
    assert scores_shape == (NUM_ANCHORS, 195)
    assert called_target_ids == target_ids


# ---------------------------------------------------------------------------
# 3. Parity comparator
# ---------------------------------------------------------------------------
def test_parity_passes_within_tolerance():
    pt_dets = [det((10.0, 10.0, 50.0, 50.0), class_id=3, score=0.90)]
    ncnn_dets = [det((11.0, 9.0, 51.0, 51.0), class_id=3, score=0.88)]  # <=3px, <=0.05

    result = compare_detections(pt_dets, ncnn_dets, box_pixel_tol=3.0, score_tol=0.05)

    assert result.passed
    assert result.matched == 1
    assert result.unmatched_pt == 0
    assert result.unmatched_ncnn == 0


def test_parity_fails_beyond_box_tolerance():
    pt_dets = [det((10.0, 10.0, 50.0, 50.0), class_id=3, score=0.90)]
    ncnn_dets = [det((30.0, 10.0, 70.0, 50.0), class_id=3, score=0.90)]  # 20px shift

    result = compare_detections(pt_dets, ncnn_dets, box_pixel_tol=3.0, score_tol=0.05)

    assert not result.passed
    assert result.unmatched_pt == 1
    assert result.mismatches


def test_parity_fails_beyond_score_tolerance():
    pt_dets = [det((10.0, 10.0, 50.0, 50.0), class_id=3, score=0.90)]
    ncnn_dets = [det((10.0, 10.0, 50.0, 50.0), class_id=3, score=0.40)]

    result = compare_detections(pt_dets, ncnn_dets, box_pixel_tol=3.0, score_tol=0.05)

    assert not result.passed


def test_parity_fails_on_class_mismatch_and_extra_detections():
    pt_dets = [det((10.0, 10.0, 50.0, 50.0), class_id=3, score=0.90)]
    ncnn_dets = [
        det((10.0, 10.0, 50.0, 50.0), class_id=4, score=0.90),  # wrong class
        det((200.0, 200.0, 240.0, 240.0), class_id=7, score=0.5),  # extra
    ]

    result = compare_detections(pt_dets, ncnn_dets, box_pixel_tol=3.0, score_tol=0.05)

    assert not result.passed
    assert result.unmatched_pt == 1
    assert result.unmatched_ncnn == 2


# ---------------------------------------------------------------------------
# 4. Detection output feeds a real Tracker
# ---------------------------------------------------------------------------
def test_masked_predictions_become_valid_detections_and_feed_a_tracker():
    raw = make_raw_output(num_classes=195, num_anchors=NUM_ANCHORS, seed=4)
    target_ids = (3, 17, 42, 88)
    raw[0, 4 + 3, 0] = 0.9  # ensure at least one confident target prediction

    _, transform = letterbox(np.zeros((480, 640, 3), dtype=np.uint8), 640)
    predictions = masked_argmax_predictions(raw, target_ids, conf_threshold=0.5)
    detections = masked_predictions_to_detections(predictions, transform)

    assert detections
    assert all(isinstance(d, Detection) for d in detections)
    assert all(d.class_id in target_ids for d in detections)

    tracker = Tracker()
    confirmed = tracker.update(detections, frame_idx=0)
    assert isinstance(confirmed, list)  # never raises; accepted cleanly


# ---------------------------------------------------------------------------
# 5. Graceful skip - never fabricate a pass
# ---------------------------------------------------------------------------
class _FakeExportModel:
    """Fake Ultralytics YOLO model: .export() stages files like the real one."""

    def __init__(self, export_dir: Path, num_classes: int = NUM_CLASSES, num_anchors: int = 100):
        self._export_dir = export_dir
        self._num_classes = num_classes
        self._num_anchors = num_anchors

    def export(self, **kwargs) -> str:
        self._export_dir.mkdir(parents=True, exist_ok=True)
        (self._export_dir / "model.ncnn.param").write_text("7767517\n", encoding="utf-8")
        (self._export_dir / "model.ncnn.bin").write_bytes(b"\x00" * 16)
        return str(self._export_dir)

    def predict(self, **kwargs):
        boxes = SimpleNamespace(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            cls=np.zeros((0,), dtype=np.float32),
            conf=np.zeros((0,), dtype=np.float32),
        )
        return [SimpleNamespace(boxes=boxes)]


def test_export_to_ncnn_stages_stable_param_and_bin_paths(tmp_path: Path):
    export_dir = tmp_path / "raw_export" / "best_ncnn_model"
    fake_model = _FakeExportModel(export_dir)

    result = export_to_ncnn(
        "best.pt",
        tmp_path / "models",
        yolo_factory=lambda weights: fake_model,
        stem="yolo11n_flags",
    )

    assert result.param_path == tmp_path / "models" / "yolo11n_flags.ncnn.param"
    assert result.bin_path == tmp_path / "models" / "yolo11n_flags.ncnn.bin"
    assert result.param_path.exists()
    assert result.bin_path.exists()
    # No leftover temp files from the atomic-stage step.
    assert list((tmp_path / "models").glob("*.tmp")) == []


def test_export_to_ncnn_raises_clear_error_and_leaves_no_half_written_files(tmp_path: Path):
    class _BoomModel:
        def export(self, **kwargs):
            raise RuntimeError("pnnx crashed")

    with pytest.raises(ExportError, match="pnnx crashed"):
        export_to_ncnn(
            "best.pt",
            tmp_path / "models",
            yolo_factory=lambda weights: _BoomModel(),
        )

    out_dir = tmp_path / "models"
    assert not out_dir.exists() or list(out_dir.iterdir()) == []


def test_run_export_and_verify_skips_when_no_test_image(tmp_path: Path):
    export_dir = tmp_path / "raw_export" / "best_ncnn_model"
    fake_model = _FakeExportModel(export_dir)
    config = ExportConfig(
        weights=Path("best.pt"),
        out_dir=tmp_path / "models",
        test_image=None,
    )
    outputs: list[str] = []

    stages = run_export_and_verify(
        config, yolo_factory=lambda weights: fake_model, output_fn=outputs.append
    )

    by_name = {s.name: s for s in stages}
    assert by_name["export"].status == "passed"
    assert by_name["layout"].status == "skipped"
    assert by_name["parity"].status == "skipped"
    assert by_name["masking"].status == "skipped"
    assert by_name["tracker_integration"].status == "skipped"
    assert not any(s.status == "failed" for s in stages)
    assert "no --test-image" in by_name["layout"].detail
    report = "\n".join(outputs)
    assert "RESULT: PASSED with" in report


def test_run_export_and_verify_skips_when_test_image_missing(tmp_path: Path):
    export_dir = tmp_path / "raw_export" / "best_ncnn_model"
    fake_model = _FakeExportModel(export_dir)
    config = ExportConfig(
        weights=Path("best.pt"),
        out_dir=tmp_path / "models",
        test_image=tmp_path / "does_not_exist.jpg",
    )

    stages = run_export_and_verify(config, yolo_factory=lambda weights: fake_model)

    by_name = {s.name: s for s in stages}
    assert by_name["layout"].status == "skipped"
    assert "not found" in by_name["layout"].detail
    assert not any(s.status == "failed" for s in stages)


def test_run_export_and_verify_skips_when_ncnn_unavailable(tmp_path: Path, monkeypatch):
    export_dir = tmp_path / "raw_export" / "best_ncnn_model"
    fake_model = _FakeExportModel(export_dir)
    image_path = tmp_path / "sample.jpg"

    import cv2

    cv2.imwrite(str(image_path), np.zeros((64, 64, 3), dtype=np.uint8))

    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "ncnn":
            raise ImportError("simulated: ncnn not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    config = ExportConfig(
        weights=Path("best.pt"), out_dir=tmp_path / "models", test_image=image_path
    )
    stages = run_export_and_verify(config, yolo_factory=lambda weights: fake_model)

    by_name = {s.name: s for s in stages}
    assert by_name["export"].status == "passed"
    assert by_name["layout"].status == "skipped"
    assert "ncnn-python is not installed" in by_name["layout"].detail
    assert not any(s.status == "failed" for s in stages)


def test_export_failure_reports_a_failed_stage_not_a_skip(tmp_path: Path):
    class _BoomModel:
        def export(self, **kwargs):
            raise RuntimeError("simulated export crash")

    config = ExportConfig(
        weights=Path("best.pt"), out_dir=tmp_path / "models", test_image=None
    )
    stages = run_export_and_verify(config, yolo_factory=lambda weights: _BoomModel())

    assert stages[0].name == "export"
    assert stages[0].status == "failed"
    assert len(stages) == 1  # nothing downstream runs after a failed export


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
def test_parse_args_defaults_and_target_ids(tmp_path: Path):
    config = parse_args(
        [
            "--weights",
            str(tmp_path / "best.pt"),
            "--out",
            str(tmp_path / "models"),
            "--target-ids",
            "3,17,42,88",
        ]
    )

    assert config.target_class_ids == (3, 17, 42, 88)
    assert config.num_classes == 195
    assert config.imgsz == 640
    assert config.quantize is None  # default --quantize 32 -> FP32 -> None


def test_parse_args_warns_and_falls_back_without_target_ids(tmp_path: Path):
    config = parse_args(
        ["--weights", str(tmp_path / "best.pt"), "--out", str(tmp_path / "models")]
    )

    assert config.target_class_ids == (0, 1, 2, 3)


def test_parse_args_fp16_quantize_flag(tmp_path: Path):
    config = parse_args(
        [
            "--weights",
            str(tmp_path / "best.pt"),
            "--out",
            str(tmp_path / "models"),
            "--quantize",
            "16",
        ]
    )
    assert config.quantize == 16


def test_render_report_marks_failed_result():
    from training.export import StageResult

    stages = [
        StageResult("export", "passed", "ok"),
        StageResult("layout", "failed", "shape mismatch"),
    ]
    report = render_report(stages)
    assert "RESULT: FAILED" in report
    assert "[FAIL] layout" in report
