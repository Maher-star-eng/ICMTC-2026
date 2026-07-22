# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   17/06/2026

"""Tests for detector.py.

All tests are hardware-free and model-free: they use StubBackend and synthetic
numpy frames.  No camera, no network, and none of onnxruntime / torch / ncnn is
imported.  The letterbox round-trip is the critical one - it guards the inverse
transform that maps model-space boxes back to original-frame pixels.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import DEFAULT_CONFIDENCE_THRESHOLD, SPECIAL_CONFIDENCE_THRESHOLDS
from vision.detector import (
    Detector,
    DetectorBackend,
    DetectorConfigError,
    LetterboxTransform,
    RawDetections,
    StubBackend,
    _decode_yolov8,
    _masked_class_argmax,
    build_detector,
    letterbox,
    resolve_special_thresholds,
)
from vision.tracker import Detection, Tracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_frame(width: int, height: int) -> np.ndarray:
    """A synthetic BGR uint8 frame; pixel content is irrelevant to the stub."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def det(box, class_id=1, score=0.9) -> Detection:
    """Build a Detection in original-frame pixels (coord defaults to None)."""
    return Detection(box=box, class_id=class_id, score=score, coord=None)


# ---------------------------------------------------------------------------
# 1. Letterbox round-trip (the critical one)
# ---------------------------------------------------------------------------
def test_letterbox_round_trip_maps_model_box_to_exact_original_pixels():
    """A known model-space box inverts to the exact expected original pixels.

    1280x720 -> 640: scale = min(640/1280, 640/720) = 0.5, so the content is
    640x360 with 140px of top/bottom pad and no horizontal pad.
    """
    frame = make_frame(1280, 720)
    _, transform = letterbox(frame, input_size=640)

    assert transform.scale == pytest.approx(0.5)
    assert transform.pad_x == pytest.approx(0.0)
    assert transform.pad_y == pytest.approx(140.0)

    model_box = np.array([[100.0, 200.0, 300.0, 400.0]])
    original = transform.inverse_boxes(model_box)[0]

    # x_orig = (x_model - pad_x) / scale ; y_orig = (y_model - pad_y) / scale
    expected = (200.0, 120.0, 600.0, 520.0)
    assert tuple(original) == pytest.approx(expected)

    # Forward then inverse is identity for an in-bounds original box.
    orig_box = (10.0, 20.0, 640.0, 500.0)
    back = transform.inverse_boxes(np.array([transform.forward_box(orig_box)]))[0]
    assert tuple(back) == pytest.approx(orig_box)


# ---------------------------------------------------------------------------
# 2. Threshold filtering
# ---------------------------------------------------------------------------
def test_threshold_drops_low_and_keeps_high_confidence():
    """Detections below conf_threshold are dropped; those at/above are kept."""
    frame = make_frame(640, 480)
    scripted = [
        det((10, 10, 50, 50), class_id=1, score=0.90),   # keep
        det((100, 100, 140, 140), class_id=2, score=0.20),  # drop
        det((200, 200, 240, 240), class_id=3, score=0.30),  # keep (== threshold)
    ]
    detector = Detector(StubBackend(scripted), conf_threshold=0.30, nms_iou=0.45)

    out = detector.detect(frame)

    scores = sorted(d.score for d in out)
    assert len(out) == 2
    assert scores == pytest.approx([0.30, 0.90])


# ---------------------------------------------------------------------------
# 3. NMS
# ---------------------------------------------------------------------------
def test_nms_collapses_overlap_keeps_distant_and_respects_class():
    """Overlapping same-class boxes collapse; distant survive; classes independent."""
    frame = make_frame(640, 640)
    scripted = [
        # Two heavily-overlapping class-1 boxes -> collapse to the higher score.
        det((10, 10, 110, 110), class_id=1, score=0.95),
        det((14, 14, 114, 114), class_id=1, score=0.80),
        # A distant class-1 box -> survives.
        det((400, 400, 500, 500), class_id=1, score=0.90),
        # A class-2 box overlapping the first cluster -> different class, kept.
        det((12, 12, 112, 112), class_id=2, score=0.85),
    ]
    detector = Detector(StubBackend(scripted), conf_threshold=0.25, nms_iou=0.45)

    out = detector.detect(frame)

    class1 = [d for d in out if d.class_id == 1]
    class2 = [d for d in out if d.class_id == 2]
    assert len(class1) == 2  # collapsed cluster + distant box
    assert len(class2) == 1  # overlapping but different class, not suppressed
    # The surviving box of the overlapping class-1 cluster is the higher score.
    survivor = min(class1, key=lambda d: d.box[0])  # the one near the origin
    assert survivor.score == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# 4. Stub passthrough
# ---------------------------------------------------------------------------
def test_stub_passthrough_yields_valid_detections_within_bounds():
    """N scripted detections -> N Detections, coord is None, boxes in-frame."""
    frame = make_frame(800, 600)
    scripted = [
        det((10, 20, 60, 80), class_id=0, score=0.7),
        det((100, 120, 180, 200), class_id=1, score=0.6),
        det((300, 300, 360, 360), class_id=2, score=0.99),
    ]
    detector = Detector(StubBackend(scripted), conf_threshold=0.25, nms_iou=0.45)

    out = detector.detect(frame)

    assert len(out) == 3
    for d in out:
        assert d.coord is None
        x1, y1, x2, y2 = d.box
        assert 0.0 <= x1 < x2 <= 800.0
        assert 0.0 <= y1 < y2 <= 600.0

    # Boxes round-trip back to their scripted originals (forward then inverse).
    got = sorted(tuple(d.box) for d in out)
    want = sorted(d.box for d in scripted)
    for g, w in zip(got, want):
        assert g == pytest.approx(w)


# ---------------------------------------------------------------------------
# 5. Type compatibility - the tracker accepts the output
# ---------------------------------------------------------------------------
def test_output_is_tracker_detection_and_tracker_consumes_it():
    """detect() output is exactly tracker.Detection and drives a Tracker."""
    frame = make_frame(640, 480)
    scripted = [det((50, 50, 100, 100), class_id=2, score=0.9)]
    detector = Detector(StubBackend(scripted), conf_threshold=0.25)

    out = detector.detect(frame)
    assert out and all(isinstance(d, Detection) for d in out)

    tracker = Tracker(iou_threshold=0.3, min_hits=3, max_age=5)
    for frame_idx in range(3):
        confirmed = tracker.update(detector.detect(frame), frame_idx)
    # Three consistent frames confirm one flag of class 2.
    assert len(confirmed) == 1
    assert confirmed[0].committed_class() == 2


# ---------------------------------------------------------------------------
# 6. Bad input never raises
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad",
    [
        None,
        np.zeros((0, 0, 3), dtype=np.uint8),   # empty
        np.zeros((640, 480), dtype=np.uint8),  # missing channel axis
        np.zeros((10, 10, 4), dtype=np.uint8), # wrong channel count
        "not an array",
    ],
)
def test_bad_input_returns_empty_and_does_not_raise(bad):
    """Empty / wrong-shape / None / non-array input -> [] without raising."""
    detector = Detector(StubBackend([det((1, 1, 5, 5))]))
    assert detector.detect(bad) == []


def test_backend_error_is_caught_and_returns_empty():
    """A backend that raises during infer is contained; detect returns []."""

    def boom(_frame):
        raise RuntimeError("simulated inference failure")

    detector = Detector(StubBackend(boom))
    assert detector.detect(make_frame(320, 240)) == []


# ---------------------------------------------------------------------------
# 7. Determinism
# ---------------------------------------------------------------------------
def test_same_frame_same_stub_is_deterministic():
    """Identical frame + identical stub yields identical output twice."""
    frame = make_frame(1280, 720)
    scripted = [
        det((10, 10, 60, 60), class_id=1, score=0.9),
        det((12, 12, 62, 62), class_id=1, score=0.9),   # tie score -> stable order
        det((500, 400, 560, 460), class_id=2, score=0.8),
    ]
    detector = Detector(StubBackend(scripted), conf_threshold=0.25, nms_iou=0.45)

    first = detector.detect(frame)
    second = detector.detect(frame)

    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert a.box == pytest.approx(b.box)
        assert a.class_id == b.class_id
        assert a.score == pytest.approx(b.score)


# ---------------------------------------------------------------------------
# Factory smoke test - config-driven threshold, no model required
# ---------------------------------------------------------------------------
def test_build_detector_stub_uses_config_default_threshold():
    """build_detector(cfg, 'stub') pulls conf from core.config, not cfg.mission.

    cfg.mission.confidence_threshold is superseded by
    core.config.DEFAULT_CONFIDENCE_THRESHOLD for detector filtering
    (config.py ASSUMPTION T3) - it is deliberately NOT read here any more,
    which is why the mock below still carries it (schema shape) but its
    value is irrelevant to the assertions.
    """

    class _Mission:
        confidence_threshold = 0.5  # present for schema shape only; not read
        iou_threshold = 0.3  # tracker's IoU - must NOT be used as NMS IoU

    class _Cfg:
        mission = _Mission()

    scripted = [
        det((10, 10, 50, 50), score=0.9),   # keep (>= 0.60 default)
        det((60, 60, 100, 100), score=0.4),  # drop (< 0.60 default)
    ]
    detector = build_detector(_Cfg(), backend="stub", scripted=scripted)

    assert detector.conf_threshold == pytest.approx(DEFAULT_CONFIDENCE_THRESHOLD)
    assert detector.special_thresholds == {}  # no class_names given -> no overrides
    assert detector.nms_iou == pytest.approx(0.45)  # module default, not tracker IoU
    out = detector.detect(make_frame(640, 480))
    assert len(out) == 1
    assert out[0].score == pytest.approx(0.9)


def test_build_detector_unknown_backend_raises():
    """An unknown backend name is a hard error at construction."""

    class _Cfg:
        class mission:
            confidence_threshold = 0.25
            iou_threshold = 0.3

    with pytest.raises(ValueError):
        build_detector(_Cfg(), backend="nope")


# ---------------------------------------------------------------------------
# 8. Class-aware confidence thresholds (SPECIAL_CONFIDENCE_THRESHOLDS)
#
# A small synthetic id<->name mapping stands in for the real 196-class model
# mapping - the resolution logic being tested doesn't care what the ids are,
# only that names resolve correctly and missing names fail loudly. class_id 5
# ("France") stands in for any of the ~192 "healthy" classes not in
# SPECIAL_CONFIDENCE_THRESHOLDS.
# ---------------------------------------------------------------------------
_TEST_CLASS_NAMES = {
    1: "Egypt",
    2: "Iraq",
    3: "Syria",
    4: "Yemen",
    5: "France",
}


def _resolved_special_thresholds() -> dict:
    """SPECIAL_CONFIDENCE_THRESHOLDS resolved against _TEST_CLASS_NAMES."""
    return resolve_special_thresholds(_TEST_CLASS_NAMES, SPECIAL_CONFIDENCE_THRESHOLDS)


def _threshold_aware_detector(scripted) -> Detector:
    return Detector(
        StubBackend(scripted),
        conf_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        special_thresholds=_resolved_special_thresholds(),
    )


def test_resolve_special_thresholds_maps_names_to_ids():
    """Every SPECIAL_CONFIDENCE_THRESHOLDS name resolves to its model id."""
    resolved = _resolved_special_thresholds()
    assert resolved == {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}


def test_resolve_special_thresholds_missing_name_raises_descriptive_exception():
    """A model missing one of the 4 briefed names fails loudly, naming it."""
    incomplete_names = {1: "Egypt", 2: "Iraq", 5: "France"}  # Syria, Yemen missing

    with pytest.raises(DetectorConfigError) as excinfo:
        resolve_special_thresholds(incomplete_names, SPECIAL_CONFIDENCE_THRESHOLDS)

    message = str(excinfo.value)
    assert "Syria" in message
    assert "Yemen" in message


@pytest.mark.parametrize("class_id", [1, 2, 3, 4])  # Egypt, Iraq, Syria, Yemen
def test_special_class_at_030_is_accepted(class_id):
    """Egypt/Iraq/Syria/Yemen at 0.30 >= their 0.25 threshold -> accepted."""
    frame = make_frame(640, 480)
    detector = _threshold_aware_detector(
        [det((10, 10, 50, 50), class_id=class_id, score=0.30)]
    )

    out = detector.detect(frame)

    assert len(out) == 1
    assert out[0].class_id == class_id
    assert out[0].score == pytest.approx(0.30)


@pytest.mark.parametrize("class_id", [1, 2, 3, 4])  # Egypt, Iraq, Syria, Yemen
def test_special_class_at_024_is_rejected(class_id):
    """Egypt/Iraq/Syria/Yemen at 0.24 < their 0.25 threshold -> rejected."""
    frame = make_frame(640, 480)
    detector = _threshold_aware_detector(
        [det((10, 10, 50, 50), class_id=class_id, score=0.24)]
    )

    out = detector.detect(frame)

    assert out == []


def test_france_at_030_is_rejected():
    """France (a healthy class) at 0.30 < its 0.60 default -> rejected."""
    frame = make_frame(640, 480)
    detector = _threshold_aware_detector(
        [det((10, 10, 50, 50), class_id=5, score=0.30)]
    )

    out = detector.detect(frame)

    assert out == []


def test_france_at_061_is_accepted():
    """France (a healthy class) at 0.61 >= its 0.60 default -> accepted."""
    frame = make_frame(640, 480)
    detector = _threshold_aware_detector(
        [det((10, 10, 50, 50), class_id=5, score=0.61)]
    )

    out = detector.detect(frame)

    assert len(out) == 1
    assert out[0].class_id == 5


def test_unknown_class_falls_back_to_default_threshold():
    """A class id absent from special_thresholds uses conf_threshold (0.60)."""
    frame = make_frame(640, 480)
    scripted = [
        det((10, 10, 50, 50), class_id=999, score=0.59),   # < 0.60 -> drop
        det((60, 60, 100, 100), class_id=999, score=0.61),  # >= 0.60 -> keep
    ]
    detector = _threshold_aware_detector(scripted)

    out = detector.detect(frame)

    assert len(out) == 1
    assert out[0].score == pytest.approx(0.61)


def test_threshold_for_class_matches_pseudocode_directly():
    """threshold_for_class mirrors SPECIAL_THRESHOLDS.get(class_id, DEFAULT)."""
    detector = Detector(
        StubBackend([]),
        conf_threshold=0.60,
        special_thresholds={1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25},
    )

    assert detector.threshold_for_class(1) == pytest.approx(0.25)  # Egypt id
    assert detector.threshold_for_class(4) == pytest.approx(0.25)  # Yemen id
    assert detector.threshold_for_class(999) == pytest.approx(0.60)  # unknown -> default


def test_healthy_classes_unchanged_when_no_special_thresholds_configured():
    """No special_thresholds (or empty) -> identical to the old global-threshold behavior."""
    frame = make_frame(640, 480)
    scripted = [
        det((10, 10, 50, 50), class_id=1, score=0.30),   # would be Egypt, but no mapping given
        det((60, 60, 100, 100), class_id=5, score=0.61),  # would be France
    ]
    detector = Detector(StubBackend(scripted), conf_threshold=0.60, nms_iou=0.45)

    out = detector.detect(frame)

    # Global 0.60 threshold applies uniformly: 0.30 dropped, 0.61 kept.
    assert len(out) == 1
    assert out[0].class_id == 5
    assert out[0].score == pytest.approx(0.61)


def test_build_detector_resolves_class_names_and_applies_special_thresholds():
    """build_detector(cfg, class_names=...) resolves names -> ids at startup."""

    class _Mission:
        confidence_threshold = 0.5  # not read; schema shape only
        iou_threshold = 0.3

    class _Cfg:
        mission = _Mission()

    scripted = [
        det((10, 10, 50, 50), class_id=1, score=0.30),    # Egypt @ 0.30 -> keep
        det((60, 60, 100, 100), class_id=5, score=0.30),  # France @ 0.30 -> drop
    ]
    detector = build_detector(
        _Cfg(), backend="stub", scripted=scripted, class_names=_TEST_CLASS_NAMES,
    )

    assert detector.conf_threshold == pytest.approx(DEFAULT_CONFIDENCE_THRESHOLD)
    assert detector.special_thresholds == {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}

    out = detector.detect(make_frame(640, 480))
    assert len(out) == 1
    assert out[0].class_id == 1


def test_build_detector_missing_special_class_name_raises_at_startup():
    """A model missing one of the 4 briefed names fails loudly at build time,
    before any frame is ever processed."""

    class _Mission:
        confidence_threshold = 0.5
        iou_threshold = 0.3

    class _Cfg:
        mission = _Mission()

    incomplete_names = {1: "Egypt", 2: "Iraq", 5: "France"}  # Syria, Yemen missing

    with pytest.raises(DetectorConfigError):
        build_detector(_Cfg(), backend="stub", class_names=incomplete_names)


def test_build_detector_without_class_names_skips_resolution():
    """class_names=None (the default) -> every class uses the default threshold.

    This is the stub/no-model-yet path (detector.py's own documented scope:
    NO TRAINED MODEL EXISTS YET) and must not raise even though
    SPECIAL_CONFIDENCE_THRESHOLDS is non-empty.
    """

    class _Mission:
        confidence_threshold = 0.5
        iou_threshold = 0.3

    class _Cfg:
        mission = _Mission()

    detector = build_detector(_Cfg(), backend="stub")

    assert detector.special_thresholds == {}


# ---------------------------------------------------------------------------
# 9. Class-masking BEFORE argmax (GAP-2, ASSUMPTION D13)
#
# Unlike the stub-based tests above, masking lives in the decode step
# (`_decode_yolov8` / `_masked_class_argmax`), which StubBackend bypasses
# entirely (it fabricates RawDetections straight from scripted Detection
# objects - there is no raw class-score matrix to mask).  These tests
# exercise the real decode path: either `_masked_class_argmax` /
# `_decode_yolov8` directly on synthetic raw tensors, or the full
# `Detector.detect()` pipeline via `_RawTensorBackend`, a minimal test-only
# backend that decodes a fixed raw tensor exactly like Onnx/Torch/Ncnn do.
# ---------------------------------------------------------------------------
def _make_synthetic_raw_tensor(num_classes: int, num_anchors: int, seed: int) -> np.ndarray:
    """Build a synthetic (1, 4+nc, N) raw model output tensor.

    Same construction as training/tests/test_export.py's make_raw_output,
    reproduced locally so test_detector.py stays a self-contained,
    zero-heavy-dependency file (no cross-import from training/).
    """
    rng = np.random.default_rng(seed)
    cx = rng.uniform(50, 590, size=num_anchors)
    cy = rng.uniform(50, 590, size=num_anchors)
    w = rng.uniform(10, 60, size=num_anchors)
    h = rng.uniform(10, 60, size=num_anchors)
    boxes = np.stack([cx, cy, w, h], axis=0)  # (4, N)
    scores = rng.uniform(0.0, 1.0, size=(num_classes, num_anchors))
    attrs = np.concatenate([boxes, scores], axis=0)  # (4+nc, N)
    return attrs[None, ...].astype(np.float32)


class _RawTensorBackend(DetectorBackend):
    """Test-only backend: decodes a fixed raw tensor via the real
    `_decode_yolov8`, exactly like Onnx/Torch/Ncnn backends do, without
    needing any heavy runtime installed.  Exercises `Detector.detect()`'s
    threading of `target_class_ids` into `backend.infer` end to end.
    """

    def __init__(self, raw_output: np.ndarray, input_size: int = 640) -> None:
        self._raw_output = raw_output
        self._input_size = input_size

    def infer(self, model_input, transform, target_class_ids=None) -> RawDetections:
        return _decode_yolov8(self._raw_output, self._input_size, target_class_ids)


def _egypt_shaped_tensor() -> np.ndarray:
    """A hand-built (1, 14, 30) tensor reproducing the proven bug scenario.

    10 classes, 30 anchors.  At anchor 0: target class 2 ("Egypt") scores
    0.30 (clears its own 0.25 special threshold); a non-target class 5
    scores 0.35, higher than the target - enough to win an UNMASKED global
    argmax, but not enough to clear class 5's own 0.60 default threshold.
    Every other anchor/class sits at 0.01, well below any threshold, so
    only anchor 0 can ever survive - isolating the one row under test.
    """
    num_classes, num_anchors = 10, 30
    raw = np.zeros((1, 4 + num_classes, num_anchors), dtype=np.float64)
    raw[0, 0, :] = np.linspace(60, 580, num_anchors)  # cx
    raw[0, 1, :] = np.linspace(60, 580, num_anchors)  # cy
    raw[0, 2, :] = 40.0  # w
    raw[0, 3, :] = 40.0  # h
    raw[0, 4:, :] = 0.01  # baseline: below every threshold everywhere
    raw[0, 4 + 5, 0] = 0.35   # non-target class 5: wins an unmasked argmax
    raw[0, 4 + 2, 0] = 0.30   # target class 2 ("Egypt"): clears ITS threshold
    raw[0, 4 + 7, 0] = 0.05   # other briefed classes ("Iraq"/"Syria"/"Yemen")
    raw[0, 4 + 8, 0] = 0.05
    raw[0, 4 + 9, 0] = 0.05
    return raw


_EGYPT_LIKE_TARGET_IDS = frozenset({2, 7, 8, 9})
_EGYPT_LIKE_SPECIAL_THRESHOLDS = {2: 0.25, 7: 0.25, 8: 0.25, 9: 0.25}


def test_masked_class_wins_locally_and_survives_its_own_threshold():
    """THE regression test: proves the exact bug is fixed.

    Unmasked, anchor 0's row is dropped entirely: the global argmax picks
    the non-target class (0.35) over the target (0.30), and 0.35 fails
    that non-target's own 0.60 default threshold - the target never gets a
    chance to compete on its own 0.25 threshold.  Masked to the briefed
    set, the target wins locally among just the 4 briefed classes and
    survives at 0.30 >= 0.25.
    """
    raw = _egypt_shaped_tensor()
    detector = Detector(
        _RawTensorBackend(raw),
        input_size=640,
        conf_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        special_thresholds=_EGYPT_LIKE_SPECIAL_THRESHOLDS,
        target_class_ids=_EGYPT_LIKE_TARGET_IDS,
    )

    out = detector.detect(make_frame(640, 640))

    assert len(out) == 1
    assert out[0].class_id == 2
    assert out[0].score == pytest.approx(0.30)


def test_unmasked_drops_the_same_row_the_old_post_hoc_filter_would_have_kept_by_luck():
    """Same tensor, target_class_ids=None: reproduces the old, wrong behavior.

    This is the control case proving the regression test above is actually
    exercising the fix, not an artifact of the tensor - without masking,
    the target detection is lost.
    """
    raw = _egypt_shaped_tensor()
    detector = Detector(
        _RawTensorBackend(raw),
        input_size=640,
        conf_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        special_thresholds=_EGYPT_LIKE_SPECIAL_THRESHOLDS,
        target_class_ids=None,
    )

    out = detector.detect(make_frame(640, 640))

    assert out == []


def test_special_thresholds_compose_correctly_with_masking():
    """special_thresholds + masking together: the exact diagnostic distinction.

    Masked: the target's OWN special threshold (0.25) applies and it
    survives at 0.30.  Unmasked: a higher-scoring non-target class wins the
    row instead, that non-target's default threshold (0.60) applies to a
    0.35 score, and the row is dropped - the target is never even
    considered.  Both outcomes are asserted together so the composition
    (not just masking alone, not just thresholds alone) is what's verified.
    """
    raw = _egypt_shaped_tensor()

    masked = Detector(
        _RawTensorBackend(raw), input_size=640,
        conf_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        special_thresholds=_EGYPT_LIKE_SPECIAL_THRESHOLDS,
        target_class_ids=_EGYPT_LIKE_TARGET_IDS,
    ).detect(make_frame(640, 640))

    unmasked = Detector(
        _RawTensorBackend(raw), input_size=640,
        conf_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        special_thresholds=_EGYPT_LIKE_SPECIAL_THRESHOLDS,
        target_class_ids=None,
    ).detect(make_frame(640, 640))

    assert [d.class_id for d in masked] == [2]
    assert unmasked == []


def test_set_target_classes_updates_live_without_reconstructing_detector():
    """construct once -> decode -> change target set -> decode again."""
    num_classes, num_anchors = 6, 20
    raw = np.zeros((1, 4 + num_classes, num_anchors), dtype=np.float64)
    raw[0, 0, :] = np.linspace(60, 580, num_anchors)
    raw[0, 1, :] = np.linspace(60, 580, num_anchors)
    raw[0, 2, :] = 40.0
    raw[0, 3, :] = 40.0
    raw[0, 4:, :] = 0.01
    raw[0, 4 + 2, 0] = 0.30  # class 2: special threshold 0.25
    raw[0, 4 + 5, 0] = 0.65  # class 5: default threshold 0.60

    detector = Detector(
        _RawTensorBackend(raw), input_size=640,
        conf_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        special_thresholds={2: 0.25},
        target_class_ids=frozenset({2}),
    )

    first = detector.detect(make_frame(640, 640))
    assert [d.class_id for d in first] == [2]

    detector.set_target_classes(frozenset({5}))  # no reconstruction
    second = detector.detect(make_frame(640, 640))
    assert [d.class_id for d in second] == [5]

    detector.set_target_classes(None)  # back to fully unmasked
    third = detector.detect(make_frame(640, 640))
    # Unmasked global argmax at anchor 0 picks class 5 (0.65 > 0.30), which
    # clears its own 0.60 default threshold.
    assert [d.class_id for d in third] == [5]


@pytest.mark.parametrize("target_size", [1, 3, 4, 10])
def test_masked_class_argmax_works_for_any_target_set_size(target_size):
    """General N - nothing hardcodes a specific count of briefed classes."""
    num_classes, num_anchors = 20, 200
    rng = np.random.default_rng(1000 + target_size)
    class_scores = rng.uniform(0.0, 1.0, size=(num_anchors, num_classes))
    target_ids = rng.choice(num_classes, size=target_size, replace=False)

    scores, class_ids = _masked_class_argmax(class_scores, target_ids)

    assert scores.shape == (num_anchors,)
    assert class_ids.shape == (num_anchors,)
    assert set(np.unique(class_ids).tolist()) <= set(target_ids.tolist())


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("target_size", [1, 2, 5, 20])
def test_masked_class_argmax_never_leaks_a_non_target_class(seed, target_size):
    """Property test: across many random tensors/target sets, never a leak."""
    num_classes, num_anchors = 50, 300
    rng = np.random.default_rng(seed * 100 + target_size)
    class_scores = rng.uniform(0.0, 1.0, size=(num_anchors, num_classes))
    target_ids = rng.choice(num_classes, size=target_size, replace=False)
    target_set = set(target_ids.tolist())

    _scores, class_ids = _masked_class_argmax(class_scores, target_ids)

    assert set(np.unique(class_ids).tolist()) <= target_set


def test_masked_class_argmax_rejects_empty_target_ids():
    class_scores = np.random.default_rng(0).uniform(size=(10, 5))
    with pytest.raises(ValueError, match="non-empty"):
        _masked_class_argmax(class_scores, [])


def test_masked_class_argmax_rejects_out_of_range_target_id():
    class_scores = np.random.default_rng(0).uniform(size=(10, 5))
    with pytest.raises(ValueError, match="out of range"):
        _masked_class_argmax(class_scores, [99])


def test_decode_yolov8_masked_never_emits_a_non_target_class():
    """Same leak-proof property, through the real decode entrypoint."""
    raw = _make_synthetic_raw_tensor(num_classes=195, num_anchors=8400, seed=7)
    target_ids = (3, 17, 42, 88)

    decoded = _decode_yolov8(raw, input_size=640, target_class_ids=target_ids)

    assert set(np.unique(decoded.class_ids).tolist()) <= set(target_ids)


def test_decode_yolov8_unmasked_is_byte_identical_to_pre_masking_snapshot():
    """Pinned snapshot, captured from the real pre-change `_decode_yolov8`
    on this exact tensor BEFORE ASSUMPTION D13 was implemented - this test
    checks against real historical output, not a re-derivation of the same
    max/argmax formula the implementation itself uses.

    target_class_ids=None (the default) must remain byte-identical: every
    training/validation script depends on unmasked decode never changing.
    """
    raw = _make_synthetic_raw_tensor(num_classes=6, num_anchors=20, seed=42)

    decoded = _decode_yolov8(raw, input_size=640)  # target_class_ids omitted -> None

    expected_boxes = [
        [452.007477, 437.657288, 483.865082, 481.077454],
        [261.177399, 224.666611, 312.81131, 258.221436],
        [491.136261, 555.046021, 536.149536, 593.307861],
        [413.769562, 508.160431, 439.387909, 556.4104],
        [75.049271, 449.459137, 126.662262, 491.195038],
        [551.71698, 136.265411, 601.955139, 173.944397],
        [446.328491, 283.049133, 475.702393, 321.009521],
        [462.266541, 61.055279, 486.682922, 86.252785],
        [97.118973, 127.545883, 141.243744, 139.086777],
        [284.7146, 402.928497, 301.702209, 434.764374],
        [240.233215, 441.806946, 260.228638, 462.536194],
        [545.268982, 557.242065, 555.637146, 587.668457],
        [373.014069, 199.610611, 422.36026, 252.280777],
        [472.669983, 239.199768, 515.912537, 260.896729],
        [266.814514, 297.10257, 312.072815, 310.017731],
        [148.190674, 140.279938, 197.227142, 164.349121],
        [333.002899, 107.817772, 365.9487, 132.497467],
        [65.24279, 285.332733, 103.679848, 328.428558],
        [488.425903, 153.605255, 505.415771, 191.456848],
        [383.235535, 387.102112, 398.962036, 436.296997],
    ]
    expected_scores = [
        0.908581, 0.916012, 0.937826, 0.969176, 0.778751, 0.71689, 0.82979,
        0.808252, 0.74679, 0.952899, 0.936813, 0.92512, 0.581061, 0.936044,
        0.590916, 0.856614, 0.958559, 0.992376, 0.891677, 0.961898,
    ]
    expected_class_ids = [1, 4, 3, 1, 1, 1, 4, 4, 5, 4, 5, 2, 2, 4, 2, 1, 2, 4, 4, 0]

    np.testing.assert_allclose(decoded.boxes, expected_boxes, atol=1e-5)
    assert decoded.scores.tolist() == pytest.approx(expected_scores, abs=1e-5)
    assert decoded.class_ids.tolist() == expected_class_ids
    assert decoded.boxes.dtype == np.float64
    assert decoded.scores.dtype == np.float64
    assert decoded.class_ids.dtype == np.int64
