# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   26/06/2026

"""export.py - export best.pt -> ncnn AND verify it is decode-compatible.

ICMTC 2026, Mission 2.  Runs entirely on the training Mac: no Pi, no camera,
no MAVLink.  Exporting a .pt to ncnn is one Ultralytics call; that is NOT
the point of this file.  The point is to catch the failure mode where the
export "succeeds" but silently changes the output tensor's shape/orientation
(wrong nc, transposed axes, an extra NMS/objectness head) so that
detector.py's `_decode_yolov8` misreads it - which produces zero errors and
zero detections on the Pi, discovered only in the field.

    export.py best.pt -> models/*.ncnn.{param,bin}   [export_to_ncnn]
                       -> raw ncnn output shape/dtype  [verify_output_layout]
                       -> .pt vs ncnn detections agree  [compare_detections]
                       -> class-mask-before-argmax works on the real tensor
                          [verify_masking / masked_argmax_predictions]
                       -> masked predictions are valid tracker.Detection
                          objects a real Tracker accepts

Every check reuses detector.py's real decode path (`_decode_yolov8`,
`_orient_attrs`, `letterbox`, `NcnnBackend`, `Detector`) and tracker.py's
real `Detection`/`Tracker` - not a reimplementation of either.

================================================================================
ASSUMPTIONS (collected here; also marked inline with # ASSUMPTION: <tag>)
================================================================================

E1  EXPORT IMGSZ
    Defaults to 640, matching detector.py's ASSUMPTION D1 (`_DEFAULT_INPUT_SIZE`,
    imported from detector.py so the two can never drift independently).
    Override with --imgsz if the trained model uses a different input size.

E2  PARITY TOLERANCE
    Detections are matched by class_id equality + nearest box (max per-
    coordinate pixel difference), not IoU: export-induced numeric error is a
    small coordinate shift, not a shape change, so a direct pixel tolerance
    is the more debuggable signal.  Defaults: box_pixel_tol=3.0 px (in
    ORIGINAL-frame pixels, after the letterbox is inverted),
    score_tol=0.05.  Both are CLI-overridable (--box-pixel-tol/--score-tol).
    These are starting points, not calibrated against a real export - tighten
    once real parity numbers are observed.

E3  EXPECTED OUTPUT LAYOUT
    A YOLOv8/YOLO11-style detect head: (1, 4+nc, N) or its transpose
    (1, N, 4+nc), nc=195 by default (--num-classes to override).  Both
    orientations are detected and accepted; `_orient_attrs` (shared with
    detector.py) decides which axis is which, and this file additionally
    checks that decision is self-consistent with what `_decode_yolov8`
    itself would pick, so a "the decode branch is wrong for this export"
    bug cannot pass silently.

E4  QUANTIZATION
    Default export is FP32 (quantize=None), not FP16, so the first
    verification pass measures export-format error alone rather than
    conflating it with quantization error.  --quantize 16 opts into FP16
    once FP32 parity is confirmed acceptable.

E5  TARGET-ID SOURCE
    The 4 briefed target flag ids come from the CLI (--target-ids
    3,17,42,88), never from config.py: `MissionConfig.target_flag_class` is a
    single int (ASSUMPTION M3 in config.py), not the 4-id briefed set this
    mission uses, so there is nothing to read there.  If --target-ids is
    omitted, this file WARNS and falls back to a (0, 1, 2, 3) placeholder so
    the masking mechanism can still be smoke-tested - never a silent no-op.

E6  NUM_CLASSES DEFAULT
    195, per the current training run (~195 flag classes).  --num-classes
    overrides it for other runs/tests.

E7  DETECTOR.PY REFACTOR
    `detector._decode_yolov8`'s orientation-detection was extracted into
    `detector._orient_attrs` (behavior-preserving; test_detector.py still
    passes unchanged) so this file's layout/masking checks reuse the exact
    same orientation logic instead of forking it.  No other decode logic
    was duplicated: box math (cx,cy,w,h -> x1,y1,x2,y2) is the only bit
    re-derived here, and only for the masked-argmax path, which
    `_decode_yolov8` has no parameter for.

E8  TEST IMAGE I/O
    Loaded via OpenCV (`cv2.imread`), which already returns BGR uint8 -
    exactly detector.py's frame convention, no channel-order fix-up needed.
    cv2 is a Mac-only dev dependency of this file; detector.py/tracker.py
    never import it.

E9  GRACEFUL SKIP, NEVER A FAKED PASS
    If ncnn-python is not importable, or no --test-image is given, or the
    image fails to decode, every downstream check is reported SKIPPED with
    the reason - never silently reported as passed.  Only export (stage 0)
    is unconditional; stages 1-4 all require both a real ncnn runtime and a
    test image.

E10 MASKING DELEGATES TO DETECTOR.PY - GAP-2, RESOLVED
    `masked_argmax_predictions`'s "restrict class-score columns to
    `target_class_ids`, then argmax within that set" step is
    `vision.detector._masked_class_argmax` - imported and called directly,
    not re-derived here.  `Detector`'s live decode (`_decode_yolov8`, when
    `Detector.target_class_ids` is set) calls the exact same function, so
    this stage verifies the real production code path, not a parallel
    implementation that could silently drift from it.  core/mission.py's
    former GAP-2 stopgap (`_ClassMaskedDetector`, a post-hoc filter) is
    gone; masking now happens before argmax in the live path too.

USAGE
-----
    python export.py --weights best.pt --out models/ --test-image sample.jpg \\
        --target-ids 3,17,42,88

Run this on the Mac after every retrain, before any weights ship to the Pi.
================================================================================
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from ioutils.logging_cfg import configure_logging, get_logger
from vision.detector import (
    Detector,
    LetterboxTransform,
    NcnnBackend,
    _DEFAULT_CONF_THRESHOLD,
    _DEFAULT_INPUT_SIZE,
    _DEFAULT_NMS_IOU,
    _decode_yolov8,
    _masked_class_argmax,
    _orient_attrs,
    letterbox,
)
from vision.tracker import Detection, Tracker

log = get_logger(__name__)

Box = tuple[float, float, float, float]

# ASSUMPTION E1/E4/E5/E6: module-wide defaults, all CLI-overridable.
_EXPORT_IMGSZ = _DEFAULT_INPUT_SIZE  # ASSUMPTION E1
_EXPECTED_NUM_CLASSES = 195  # ASSUMPTION E6
_DEFAULT_BOX_PIXEL_TOL = 3.0  # ASSUMPTION E2
_DEFAULT_SCORE_TOL = 0.05  # ASSUMPTION E2
_DEFAULT_TARGET_IDS: tuple[int, ...] = (0, 1, 2, 3)  # ASSUMPTION E5
_STABLE_MODEL_STEM = "yolo11n_flags"
_NCNN_INPUT_NAME = "in0"
_NCNN_OUTPUT_NAME = "out0"


class ExportError(Exception):
    """Raised when the .pt -> ncnn export itself cannot be completed."""


class LayoutError(Exception):
    """Raised when the exported model's raw output tensor is not decode-compatible."""


# ---------------------------------------------------------------------------
# Ultralytics seam (dependency-injectable so tests never need real weights).
# ---------------------------------------------------------------------------
class YoloModel(Protocol):
    """Small Protocol for the Ultralytics YOLO object used by this module."""

    def export(self, **kwargs: Any) -> str:
        """Export the model and return the output path."""
        ...

    def predict(self, **kwargs: Any) -> Any:
        """Run inference and return Ultralytics-style Results."""


class YoloFactory(Protocol):
    """Factory signature for dependency-injected YOLO model construction."""

    def __call__(self, weights: str) -> YoloModel:
        """Return a YOLO-like model loaded from weights."""
        ...


def _default_yolo_factory(weights: str) -> YoloModel:
    """Load a YOLO model from Ultralytics, with a clear missing-package error."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on local install
        raise ExportError(
            "Ultralytics is required for export.py. Install it, then rerun export.py."
        ) from exc
    return YOLO(weights)  # type: ignore[return-value]  # ultralytics 8.4.90's own stub for .predict() has an inconsistent default (source: ... = None where None isn't in the declared union) — not a project bug (see diagnostics_report.md C-4)


# ---------------------------------------------------------------------------
# Stage 0: export .pt -> ncnn.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ExportedNcnnModel:
    """The staged, stable-path ncnn export this file verifies.

    Attributes:
        param_path: Stable .ncnn.param path (e.g. models/yolo11n_flags.ncnn.param).
        bin_path: Stable .ncnn.bin path.
        source_dir: The raw Ultralytics export directory (kept for reference).
        imgsz: Model input side length used at export time.
    """

    param_path: Path
    bin_path: Path
    source_dir: Path
    imgsz: int


def export_to_ncnn(
    pt_path: str | Path,
    out_dir: str | Path,
    *,
    imgsz: int = _EXPORT_IMGSZ,
    quantize: int | None = None,
    stem: str = _STABLE_MODEL_STEM,
    yolo_factory: YoloFactory | None = None,
) -> ExportedNcnnModel:
    """Export `pt_path` to ncnn and stage it at a stable `out_dir/<stem>.ncnn.*` path.

    Ultralytics writes to `<weights-stem>_ncnn_model/model.ncnn.{param,bin}`;
    this copies those two files to a stable, predictable pair of paths so
    downstream tooling (detector.py's NcnnBackend, the Pi deployment) never
    has to know Ultralytics' export directory naming.  Staging is via a
    temp-file-then-replace so a failure partway through never leaves a
    half-written destination pair.

    Args:
        pt_path: Path to the trained .pt weights.
        out_dir: Directory to stage the stable .ncnn.param/.bin into.
        imgsz: Model input side length.  # ASSUMPTION E1
        quantize: None for FP32, 16 for FP16.  # ASSUMPTION E4
        stem: Base filename for the staged pair (default "yolo11n_flags").
        yolo_factory: Overrides YOLO model construction (tests only).

    Returns:
        The staged `ExportedNcnnModel`.

    Raises:
        ExportError: on any export or file-staging failure.
    """
    factory = yolo_factory or _default_yolo_factory
    out_path = Path(out_dir).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)

    log.info(
        "exporting %s -> ncnn (imgsz=%d, quantize=%s)", pt_path, imgsz, quantize
    )
    try:
        model = factory(str(pt_path))
        exported = model.export(format="ncnn", imgsz=imgsz, quantize=quantize)
    except Exception as exc:  # noqa: BLE001 - export failure must be reported, not typed
        raise ExportError(f"ultralytics ncnn export failed for {pt_path}: {exc}") from exc

    exported_dir = Path(exported)
    source_param = exported_dir / "model.ncnn.param"
    source_bin = exported_dir / "model.ncnn.bin"
    if not source_param.exists() or not source_bin.exists():
        raise ExportError(
            "export reported success but expected files are missing in "
            f"{exported_dir}: param_exists={source_param.exists()} "
            f"bin_exists={source_bin.exists()}"
        )

    dest_param = out_path / f"{stem}.ncnn.param"
    dest_bin = out_path / f"{stem}.ncnn.bin"
    tmp_param = dest_param.with_name(dest_param.name + ".tmp")
    tmp_bin = dest_bin.with_name(dest_bin.name + ".tmp")
    try:
        shutil.copy2(source_param, tmp_param)
        shutil.copy2(source_bin, tmp_bin)
        tmp_param.replace(dest_param)
        tmp_bin.replace(dest_bin)
    except OSError as exc:
        tmp_param.unlink(missing_ok=True)
        tmp_bin.unlink(missing_ok=True)
        raise ExportError(
            f"failed to stage exported ncnn files into {out_path}: {exc}"
        ) from exc

    log.info("exported ncnn model -> param=%s bin=%s", dest_param, dest_bin)
    return ExportedNcnnModel(
        param_path=dest_param, bin_path=dest_bin, source_dir=exported_dir, imgsz=imgsz
    )


# ---------------------------------------------------------------------------
# Test-image loading.
# ---------------------------------------------------------------------------
def load_test_image(path: str | Path) -> np.ndarray:
    """Load a test image as a (H, W, 3) BGR uint8 array.  # ASSUMPTION E8"""
    image_path = Path(path).expanduser()
    if not image_path.exists():
        raise ExportError(f"test image not found: {image_path}")

    import cv2  # ASSUMPTION E8: Mac-only dependency of this file only

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ExportError(
            f"failed to decode test image (unsupported or corrupt file): {image_path}"
        )
    return frame


# ---------------------------------------------------------------------------
# Raw ncnn forward pass - same preprocessing detector.NcnnBackend uses, but
# returns the tensor BEFORE decode so its shape/orientation can be inspected.
# ---------------------------------------------------------------------------
def ncnn_forward_raw(
    param_path: str | Path,
    bin_path: str | Path,
    input_size: int,
    frame: np.ndarray,
    *,
    input_name: str = _NCNN_INPUT_NAME,
    output_name: str = _NCNN_OUTPUT_NAME,
) -> tuple[np.ndarray, LetterboxTransform]:
    """Run one raw ncnn forward pass; return (raw_output, letterbox_transform).

    Mirrors detector.NcnnBackend.infer's preprocessing exactly (letterbox ->
    BGR->RGB -> normalize -> extract) but stops short of `_decode_yolov8`, so
    the pre-decode tensor is available for direct shape/dtype inspection -
    that seam is the whole point of this file.

    Raises:
        LayoutError: if the ncnn extractor itself reports an error.
    """
    import ncnn  # lazy: only this file's verification path needs it

    net = ncnn.Net()  # type: ignore[attr-defined]  # ncnn is a compiled C-extension with no type stubs — confirmed to work correctly at runtime (see diagnostics_report.md C-5)
    net.load_param(str(param_path))
    net.load_model(str(bin_path))

    model_input, transform = letterbox(frame, input_size)
    mat_in = ncnn.Mat.from_pixels(  # type: ignore[attr-defined]  # ncnn is a compiled C-extension with no type stubs — confirmed to work correctly at runtime (see diagnostics_report.md C-5)
        np.ascontiguousarray(model_input[:, :, ::-1]),  # BGR->RGB
        ncnn.Mat.PixelType.PIXEL_RGB,  # type: ignore[attr-defined]  # ncnn is a compiled C-extension with no type stubs — confirmed to work correctly at runtime (see diagnostics_report.md C-5)
        input_size,
        input_size,
    )
    mat_in.substract_mean_normalize([], [1 / 255.0, 1 / 255.0, 1 / 255.0])
    extractor = net.create_extractor()
    extractor.input(input_name, mat_in)
    ret, mat_out = extractor.extract(output_name)
    if ret != 0:
        raise LayoutError(f"ncnn extractor.extract({output_name!r}) returned error code {ret}")

    return np.array(mat_out)[None, ...], transform


# ---------------------------------------------------------------------------
# Stage 1: output-layout inspection.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LayoutReport:
    """Result of inspecting a raw ncnn output tensor's shape/orientation."""

    raw_shape: tuple[int, ...]
    raw_dtype: str
    orientation: str  # "attrs_first" ([1,4+nc,N]) or "anchors_first" ([1,N,4+nc])
    num_classes_detected: int
    num_anchors: int


def inspect_output_layout(
    raw_output: np.ndarray, expected_num_classes: int
) -> LayoutReport:
    """Validate a raw output tensor's shape against the YOLOv8/YOLO11 head layout.

    Accepts (1, 4+nc, N) or (1, N, 4+nc); rejects wrong nc, wrong ndim, or a
    shape where neither axis equals 4+nc (extra head, wrong task export,
    etc).  Fails loud with the actual vs expected shape - that is the point.

    Raises:
        LayoutError: on any shape mismatch.
    """
    arr = np.asarray(raw_output)
    log.info("raw ncnn output shape=%s dtype=%s", arr.shape, arr.dtype)

    squeezed = arr[0] if (arr.ndim == 3 and arr.shape[0] == 1) else arr
    if squeezed.ndim != 2:
        raise LayoutError(
            f"unexpected output ndim: shape={arr.shape}; expected (1, 4+nc, N) "
            "or (1, N, 4+nc)"
        )

    dim_a, dim_b = squeezed.shape
    expected_attrs = 4 + expected_num_classes
    if dim_a == expected_attrs:
        orientation = "attrs_first"
        num_anchors = dim_b
        detected_nc = dim_a - 4
    elif dim_b == expected_attrs:
        orientation = "anchors_first"
        num_anchors = dim_a
        detected_nc = dim_b - 4
    else:
        raise LayoutError(
            f"output shape {arr.shape} does not match expected (1, {expected_attrs}, N) "
            f"or (1, N, {expected_attrs}) for num_classes={expected_num_classes}; got "
            f"dims ({dim_a}, {dim_b}) - neither axis equals 4+nc={expected_attrs}. "
            "Likely wrong nc at export, an extra head (NMS/objectness), or a "
            "non-detect task export."
        )

    if detected_nc != expected_num_classes:
        raise LayoutError(
            f"exported model has num_classes={detected_nc}, expected "
            f"{expected_num_classes}; weights may not match this training run."
        )

    return LayoutReport(
        raw_shape=tuple(int(d) for d in arr.shape),
        raw_dtype=str(arr.dtype),
        orientation=orientation,
        num_classes_detected=detected_nc,
        num_anchors=int(num_anchors),
    )


def verify_output_layout(
    raw_output: np.ndarray, *, expected_num_classes: int, input_size: int
) -> LayoutReport:
    """Full layout check: shape/orientation, PLUS self-consistency with the
    real decode path (`_decode_yolov8` / `_orient_attrs`).

    Confirms the decode's own orientation heuristic (`shape[0] < shape[1]`)
    agrees with the orientation this file detected, and that decoding
    produces exactly one row per anchor with all class ids in range - i.e.
    that `_decode_yolov8` actually reads this export correctly, not just
    that the shape looks plausible.

    Raises:
        LayoutError: on any shape mismatch or decode/layout disagreement.
    """
    report = inspect_output_layout(raw_output, expected_num_classes)

    arr = np.asarray(raw_output)
    squeezed = arr[0] if (arr.ndim == 3 and arr.shape[0] == 1) else arr
    decode_picks_attrs_first = squeezed.shape[0] < squeezed.shape[1]
    expected_attrs_first = report.orientation == "attrs_first"
    if decode_picks_attrs_first != expected_attrs_first:
        raise LayoutError(
            "_decode_yolov8's orientation heuristic (shape[0]<shape[1]) disagrees "
            f"with the detected layout: raw shape={arr.shape}, detected "
            f"orientation={report.orientation}. The decode branch would misread "
            "this export."
        )

    decoded = _decode_yolov8(raw_output, input_size)
    if decoded.boxes.shape[0] != report.num_anchors:
        raise LayoutError(
            f"_decode_yolov8 returned {decoded.boxes.shape[0]} boxes but the raw "
            f"tensor has {report.num_anchors} anchors; decode/layout mismatch."
        )
    if decoded.class_ids.size and int(decoded.class_ids.max()) >= expected_num_classes:
        raise LayoutError(
            f"_decode_yolov8 produced class_id {int(decoded.class_ids.max())} "
            f">= expected_num_classes {expected_num_classes}."
        )
    return report


# ---------------------------------------------------------------------------
# Stage 2: .pt vs ncnn parity.
# ---------------------------------------------------------------------------
def _to_numpy(value: Any) -> np.ndarray:
    """Convert a numpy array, torch tensor, or test double to a numpy array."""
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    to_numpy = getattr(value, "numpy", None)
    if callable(to_numpy):
        value = to_numpy()
    return np.asarray(value)


def run_pt_predictions(
    pt_path: str | Path,
    frame: np.ndarray,
    *,
    imgsz: int,
    conf_threshold: float,
    nms_iou: float,
    yolo_factory: YoloFactory | None = None,
) -> list[Detection]:
    """Run the original .pt model via Ultralytics; return original-pixel Detections.

    Uses Ultralytics' own pre/postprocess (not detector.py's) - it is the
    ground truth this file verifies the ncnn export's decode path against.
    """
    factory = yolo_factory or _default_yolo_factory
    try:
        model = factory(str(pt_path))
        results = model.predict(
            source=frame, imgsz=imgsz, conf=conf_threshold, iou=nms_iou, verbose=False
        )
    except Exception as exc:  # noqa: BLE001 - must be reported, not swallowed
        raise ExportError(f"ultralytics .pt inference failed for {pt_path}: {exc}") from exc

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
        return []

    xyxy = _to_numpy(boxes.xyxy)
    cls = _to_numpy(boxes.cls)
    conf = _to_numpy(boxes.conf)

    detections: list[Detection] = []
    for i in range(xyxy.shape[0]):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            Detection(
                box=(x1, y1, x2, y2), class_id=int(cls[i]), score=float(conf[i]), coord=None
            )
        )
    return detections


@dataclass(frozen=True, slots=True)
class ParityResult:
    """Result of comparing .pt vs ncnn detections on the same image."""

    passed: bool
    num_pt: int
    num_ncnn: int
    matched: int
    unmatched_pt: int
    unmatched_ncnn: int
    max_box_pixel_diff: float
    max_score_diff: float
    mismatches: tuple[str, ...]


def _box_max_abs_diff(a: Box, b: Box) -> float:
    """Max per-coordinate absolute pixel difference between two boxes."""
    return max(abs(x - y) for x, y in zip(a, b))


def compare_detections(
    pt_detections: Sequence[Detection],
    ncnn_detections: Sequence[Detection],
    *,
    box_pixel_tol: float = _DEFAULT_BOX_PIXEL_TOL,
    score_tol: float = _DEFAULT_SCORE_TOL,
) -> ParityResult:
    """Greedy same-class nearest-box match between two Detection lists.  # ASSUMPTION E2

    Each .pt detection (highest score first) is matched to the nearest
    same-class, not-yet-claimed ncnn detection; the match passes only if the
    per-coordinate pixel diff and score diff are both within tolerance.  Any
    .pt detection with no acceptable match, and any leftover unmatched ncnn
    detection, counts against `passed`.
    """
    remaining = list(enumerate(ncnn_detections))
    matched = 0
    unmatched_pt = 0
    max_box_diff = 0.0
    max_score_diff = 0.0
    mismatches: list[str] = []

    for pt_det in sorted(pt_detections, key=lambda d: -d.score):
        best_pos = None
        best_diff = math.inf
        for pos, (_, cand) in enumerate(remaining):
            if cand.class_id != pt_det.class_id:
                continue
            diff = _box_max_abs_diff(pt_det.box, cand.box)
            if diff < best_diff:
                best_diff = diff
                best_pos = pos

        if best_pos is None:
            unmatched_pt += 1
            mismatches.append(
                f"pt class={pt_det.class_id} score={pt_det.score:.3f} "
                f"box={pt_det.box}: no ncnn detection of the same class"
            )
            continue

        _, cand = remaining.pop(best_pos)
        score_diff = abs(pt_det.score - cand.score)
        max_box_diff = max(max_box_diff, best_diff)
        max_score_diff = max(max_score_diff, score_diff)
        if best_diff > box_pixel_tol or score_diff > score_tol:
            unmatched_pt += 1
            mismatches.append(
                f"class={pt_det.class_id}: nearest ncnn box differs by "
                f"{best_diff:.2f}px (tol={box_pixel_tol}px), score differs by "
                f"{score_diff:.4f} (tol={score_tol})"
            )
            continue
        matched += 1

    unmatched_ncnn = len(remaining)
    return ParityResult(
        passed=(unmatched_pt == 0 and unmatched_ncnn == 0),
        num_pt=len(pt_detections),
        num_ncnn=len(ncnn_detections),
        matched=matched,
        unmatched_pt=unmatched_pt,
        unmatched_ncnn=unmatched_ncnn,
        max_box_pixel_diff=max_box_diff,
        max_score_diff=max_score_diff,
        mismatches=tuple(mismatches),
    )


# ---------------------------------------------------------------------------
# Stage 3: class-mask-before-argmax sanity on the real exported tensor.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MaskedPrediction:
    """One masked-argmax prediction from a raw model output tensor.

    Attributes:
        box: (x1, y1, x2, y2) in MODEL-INPUT pixels (mirrors RawDetections).
        class_id: Always one of the target_class_ids passed to the mask.
        score: The masked-argmax confidence (max over target ids only).
    """

    box: Box
    class_id: int
    score: float


def masked_argmax_predictions(
    raw_output: np.ndarray,
    target_class_ids: Sequence[int],
    *,
    conf_threshold: float = _DEFAULT_CONF_THRESHOLD,
) -> list[MaskedPrediction]:
    """Class-mask a raw output tensor to `target_class_ids`, then argmax.

    The mask-before-argmax step itself - restrict each anchor's class-score
    row to only the briefed target ids, then take score/argmax *within that
    restricted set*, never the global argmax - is delegated to
    `vision.detector._masked_class_argmax`: the exact same function
    `Detector`'s live decode (`_decode_yolov8`) calls when
    `Detector.target_class_ids` is set.  This file does not keep its own
    copy of that mechanism, so this verification and live production
    behavior can never silently diverge; it operates on the real exported
    tensor's orientation via the shared `_orient_attrs`.

    Raises:
        ValueError: if target_class_ids is empty or any id is out of range
            (raised by `_masked_class_argmax`).
    """
    attrs = _orient_attrs(raw_output)
    boxes_cxcywh = attrs[:4, :].T  # (N, 4)
    class_scores = attrs[4:, :].T  # (N, nc)

    scores, class_ids = _masked_class_argmax(class_scores, target_class_ids)

    keep = scores >= conf_threshold
    boxes_cxcywh = boxes_cxcywh[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    cx, cy, w, h = (
        boxes_cxcywh[:, 0],
        boxes_cxcywh[:, 1],
        boxes_cxcywh[:, 2],
        boxes_cxcywh[:, 3],
    )
    x1, y1, x2, y2 = cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0

    return [
        MaskedPrediction(
            box=(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
            class_id=int(class_ids[i]),
            score=float(scores[i]),
        )
        for i in range(scores.shape[0])
    ]


@dataclass(frozen=True, slots=True)
class MaskingReport:
    """Result of verifying the class-mask-before-argmax mechanism."""

    passed: bool
    num_predictions: int
    target_class_ids: tuple[int, ...]
    class_ids_seen: tuple[int, ...]
    detail: str


def verify_masking(
    raw_output: np.ndarray,
    target_class_ids: Sequence[int],
    *,
    conf_threshold: float = _DEFAULT_CONF_THRESHOLD,
) -> MaskingReport:
    """Run masked_argmax_predictions and confirm no non-target class leaked out.

    The construction of `masked_argmax_predictions` makes leakage
    structurally impossible; this check exists anyway as a fail-loud
    belt-and-suspenders assertion directly against the real tensor, not a
    trust-the-construction assumption.
    """
    predictions = masked_argmax_predictions(
        raw_output, target_class_ids, conf_threshold=conf_threshold
    )
    target_set = set(target_class_ids)
    seen = tuple(sorted({p.class_id for p in predictions}))
    offenders = sorted({p.class_id for p in predictions if p.class_id not in target_set})

    passed = not offenders
    detail = (
        f"{len(predictions)} masked predictions, classes seen={seen}"
        if passed
        else f"{len(offenders)} prediction(s) leaked non-target class(es): {offenders}"
    )
    return MaskingReport(
        passed=passed,
        num_predictions=len(predictions),
        target_class_ids=tuple(sorted(target_set)),
        class_ids_seen=seen,
        detail=detail,
    )


def masked_predictions_to_detections(
    predictions: Sequence[MaskedPrediction], transform: LetterboxTransform
) -> list[Detection]:
    """Convert MaskedPrediction (model-input px) to tracker.Detection (original px).

    Mirrors detector.Detector._postprocess's final step: invert the
    letterbox, then drop degenerate (zero/negative-area) boxes.
    """
    if not predictions:
        return []

    boxes = np.array([p.box for p in predictions], dtype=np.float64)
    orig_boxes = transform.inverse_boxes(boxes)

    detections: list[Detection] = []
    for i, pred in enumerate(predictions):
        x1, y1, x2, y2 = (float(v) for v in orig_boxes[i])
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            Detection(box=(x1, y1, x2, y2), class_id=pred.class_id, score=pred.score, coord=None)
        )
    return detections


# ---------------------------------------------------------------------------
# Orchestration - runs every stage in order, reports pass/fail/skip per stage.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ExportConfig:
    """Full configuration for one `python export.py` run."""

    weights: Path
    out_dir: Path
    test_image: Path | None
    target_class_ids: tuple[int, ...] = _DEFAULT_TARGET_IDS
    num_classes: int = _EXPECTED_NUM_CLASSES
    imgsz: int = _EXPORT_IMGSZ
    conf_threshold: float = _DEFAULT_CONF_THRESHOLD
    nms_iou: float = _DEFAULT_NMS_IOU
    box_pixel_tol: float = _DEFAULT_BOX_PIXEL_TOL
    score_tol: float = _DEFAULT_SCORE_TOL
    quantize: int | None = None
    stem: str = _STABLE_MODEL_STEM
    log_level: str = "INFO"


@dataclass(frozen=True, slots=True)
class StageResult:
    """One row of the verification report."""

    name: str
    status: str  # "passed" | "failed" | "skipped"
    detail: str


def render_report(stages: Sequence[StageResult]) -> str:
    """Render the final pass/fail/skip report for a `python export.py` run."""
    icons = {"passed": "[PASS]", "failed": "[FAIL]", "skipped": "[SKIP]"}
    lines = [
        "==================================================",
        "export.py verification report",
        "==================================================",
    ]
    for stage in stages:
        lines.append(f"{icons.get(stage.status, '[????]')} {stage.name}: {stage.detail}")
    lines.append("==================================================")

    failed = [s for s in stages if s.status == "failed"]
    skipped = [s for s in stages if s.status == "skipped"]
    if failed:
        lines.append(
            f"RESULT: FAILED ({len(failed)} stage(s) failed) - "
            "do NOT ship this export to the Pi."
        )
    elif skipped:
        lines.append(
            f"RESULT: PASSED with {len(skipped)} stage(s) skipped - re-run with "
            "--test-image and ncnn-python installed for full verification."
        )
    else:
        lines.append("RESULT: PASSED - export is decode-compatible.")
    lines.append("==================================================")
    return "\n".join(lines)


def run_export_and_verify(
    config: ExportConfig,
    *,
    yolo_factory: YoloFactory | None = None,
    output_fn: Callable[[str], None] = print,
) -> list[StageResult]:
    """Run export + all four verification stages; return the stage-by-stage report.

    Never raises out of a stage failure - every stage result is captured and
    the full report is printed once at the end (or as soon as a gate blocks
    further stages).  Only genuinely unexpected errors during argument
    handling would propagate.
    """
    stages: list[StageResult] = []

    try:
        exported = export_to_ncnn(
            config.weights,
            config.out_dir,
            imgsz=config.imgsz,
            quantize=config.quantize,
            stem=config.stem,
            yolo_factory=yolo_factory,
        )
    except ExportError as exc:
        stages.append(StageResult("export", "failed", str(exc)))
        output_fn(render_report(stages))
        return stages

    stages.append(
        StageResult(
            "export",
            "passed",
            f"param={exported.param_path} bin={exported.bin_path} imgsz={exported.imgsz}",
        )
    )

    def _skip_remaining(reason: str) -> list[StageResult]:
        for name in ("layout", "parity", "masking", "tracker_integration"):
            stages.append(StageResult(name, "skipped", reason))
        output_fn(render_report(stages))
        return stages

    if config.test_image is None:
        return _skip_remaining("no --test-image provided")

    try:
        frame = load_test_image(config.test_image)
    except ExportError as exc:
        return _skip_remaining(str(exc))

    try:
        import ncnn  # noqa: F401 - availability probe only
    except ImportError:
        return _skip_remaining(
            "ncnn-python is not installed; install it on this Mac to run the "
            "remaining decode-compatibility checks"
        )

    try:
        raw_output, transform = ncnn_forward_raw(
            exported.param_path, exported.bin_path, exported.imgsz, frame
        )
        layout = verify_output_layout(
            raw_output, expected_num_classes=config.num_classes, input_size=exported.imgsz
        )
    except LayoutError as exc:
        stages.append(StageResult("layout", "failed", str(exc)))
        for name in ("parity", "masking", "tracker_integration"):
            stages.append(StageResult(name, "skipped", "layout check failed"))
        output_fn(render_report(stages))
        return stages

    stages.append(
        StageResult(
            "layout",
            "passed",
            f"shape={layout.raw_shape} dtype={layout.raw_dtype} "
            f"orientation={layout.orientation} nc={layout.num_classes_detected} "
            f"anchors={layout.num_anchors}",
        )
    )

    try:
        pt_detections = run_pt_predictions(
            config.weights,
            frame,
            imgsz=exported.imgsz,
            conf_threshold=config.conf_threshold,
            nms_iou=config.nms_iou,
            yolo_factory=yolo_factory,
        )
        ncnn_detector = Detector(
            NcnnBackend(str(exported.param_path), str(exported.bin_path), input_size=exported.imgsz),
            input_size=exported.imgsz,
            conf_threshold=config.conf_threshold,
            nms_iou=config.nms_iou,
        )
        ncnn_detections = ncnn_detector.detect(frame)
        parity = compare_detections(
            pt_detections,
            ncnn_detections,
            box_pixel_tol=config.box_pixel_tol,
            score_tol=config.score_tol,
        )
    except ExportError as exc:
        stages.append(StageResult("parity", "failed", str(exc)))
    else:
        detail = (
            f"pt={parity.num_pt} ncnn={parity.num_ncnn} matched={parity.matched} "
            f"max_box_diff={parity.max_box_pixel_diff:.2f}px "
            f"max_score_diff={parity.max_score_diff:.4f}"
        )
        if not parity.passed:
            detail += f"; mismatches={list(parity.mismatches)}"
        stages.append(StageResult("parity", "passed" if parity.passed else "failed", detail))

    masking = verify_masking(
        raw_output, config.target_class_ids, conf_threshold=config.conf_threshold
    )
    stages.append(
        StageResult(
            "masking",
            "passed" if masking.passed else "failed",
            f"{masking.detail} (target_ids={masking.target_class_ids})",
        )
    )

    try:
        predictions = masked_argmax_predictions(
            raw_output, config.target_class_ids, conf_threshold=config.conf_threshold
        )
        detections = masked_predictions_to_detections(predictions, transform)
        for det in detections:
            if not isinstance(det, Detection):  # pragma: no cover - defensive
                raise TypeError("masked prediction did not produce a tracker.Detection")
        Tracker().update(detections, frame_idx=0)
    except Exception as exc:  # noqa: BLE001 - must be reported as a failed stage
        stages.append(StageResult("tracker_integration", "failed", str(exc)))
    else:
        stages.append(
            StageResult(
                "tracker_integration",
                "passed",
                f"{len(detections)} Detection object(s) accepted by Tracker.update",
            )
        )

    output_fn(render_report(stages))
    return stages


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _parse_int_list(value: str) -> tuple[int, ...]:
    """argparse converter for a comma-separated list of ints (--target-ids)."""
    try:
        ids = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a comma-separated list of integers"
        ) from None
    if not ids:
        raise argparse.ArgumentTypeError("--target-ids must contain at least one integer")
    return ids


def parse_args(argv: Sequence[str] | None = None) -> ExportConfig:
    """Parse CLI arguments into a typed `ExportConfig`."""
    parser = argparse.ArgumentParser(
        description=(
            "Export a trained YOLO11n .pt to ncnn and verify the exported "
            "output layout is decode-compatible with detector.py before it "
            "ever ships to the Pi."
        )
    )
    parser.add_argument("--weights", required=True, help="Path to best.pt.")
    parser.add_argument("--out", required=True, help="Output dir for the staged .ncnn.param/.bin.")
    parser.add_argument(
        "--test-image",
        default=None,
        help="Image for layout/parity/masking checks; omit to skip those stages.",
    )
    parser.add_argument(
        "--target-ids",
        type=_parse_int_list,
        default=None,
        help="Comma-separated briefed target class ids, e.g. 3,17,42,88.",
    )
    parser.add_argument("--num-classes", type=int, default=_EXPECTED_NUM_CLASSES)
    parser.add_argument("--imgsz", type=int, default=_EXPORT_IMGSZ)
    parser.add_argument("--conf-threshold", type=float, default=_DEFAULT_CONF_THRESHOLD)
    parser.add_argument("--nms-iou", type=float, default=_DEFAULT_NMS_IOU)
    parser.add_argument("--box-pixel-tol", type=float, default=_DEFAULT_BOX_PIXEL_TOL)
    parser.add_argument("--score-tol", type=float, default=_DEFAULT_SCORE_TOL)
    parser.add_argument("--quantize", type=int, choices=(16, 32), default=32)
    parser.add_argument("--stem", default=_STABLE_MODEL_STEM)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    target_ids = args.target_ids
    if target_ids is None:
        target_ids = _DEFAULT_TARGET_IDS
        log.warning(
            "no --target-ids given; using placeholder %s for the masking check "
            "(ASSUMPTION E5) - pass the real 4 briefed ids for a meaningful run",
            target_ids,
        )

    return ExportConfig(
        weights=Path(args.weights).expanduser(),
        out_dir=Path(args.out).expanduser(),
        test_image=Path(args.test_image).expanduser() if args.test_image else None,
        target_class_ids=target_ids,
        num_classes=args.num_classes,
        imgsz=args.imgsz,
        conf_threshold=args.conf_threshold,
        nms_iou=args.nms_iou,
        box_pixel_tol=args.box_pixel_tol,
        score_tol=args.score_tol,
        quantize=None if args.quantize == 32 else args.quantize,
        stem=args.stem,
        log_level=args.log_level,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `python export.py`."""
    config = parse_args(argv)
    configure_logging(level=config.log_level)
    stages = run_export_and_verify(config)
    return 1 if any(stage.status == "failed" for stage in stages) else 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI use
    raise SystemExit(main(sys.argv[1:]))
