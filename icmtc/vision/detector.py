# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   17/06/2026

"""detector.py - flag-detection inference wrapper (ICMTC 2026, Mission 2).

This is the *socket* a neural net plugs into, NOT a trained flag detector.
It owns the per-frame path: one BGR camera frame in -> a list of
`tracker.Detection` out, ready for `Tracker.update(...)`.  The actual model
is a swappable backend behind an abstract seam so the pipeline can be
developed and tested today with zero heavy runtimes installed.

    capture -> [ detector ] -> tracker -> geolocate -> mission

================================================================================
SCOPE - read first
================================================================================
NO TRAINED MODEL EXISTS YET.  There is no dataset and no weights file.  This
module deliberately fabricates nothing: it does not ship a .ncnn/.onnx/.pt
path pretending to work, and it makes no accuracy claim.  What it provides is
    * the exact input/output contract the tracker already expects,
    * a StubBackend that returns canned detections (tests + pipeline dev),
    * real Onnx/Torch/Ncnn backends that are honest *scaffolding* - they show
      exactly where weights load and where raw output is decoded, and are
      marked `# requires trained model` / UNVERIFIED until weights are
      produced from the dataset.

This file is INFERENCE ONLY.  It does not track, deduplicate, geo-locate,
select a target, or know which class is "the target flag".  It emits
"class 3, this box in original-frame pixels, score 0.82" - nothing about
meaning.  All that downstream reasoning lives in tracker/geolocate/mission.

Importing or testing this file needs no camera, no network, and no model
runtime - only numpy.  The heavy runtimes are imported *inside* their backend.

================================================================================
ASSUMPTIONS (collected here; also marked inline with # ASSUMPTION: <tag>)
================================================================================

D1  INPUT SIZE
    Model input defaults to a square 640x640 letterbox.  config.py has no
    detector-input-size field today, so `build_detector` uses 640 unless an
    explicit `input_size` is passed.  Change here once the trained model's
    input geometry is known.

D2  CONFIDENCE THRESHOLD SOURCE
    The threshold is class-aware, never a single hardcoded scalar in
    production: `build_detector` reads `core.config.DEFAULT_CONFIDENCE_THRESHOLD`
    (the fallback for every class) and `core.config.SPECIAL_CONFIDENCE_THRESHOLDS`
    (name-keyed per-class overrides - four classes with systematically lower
    validation confidence get a lower keep-threshold).  `cfg.mission.
    confidence_threshold` is NOT read here any more (superseded - config.py
    ASSUMPTION T3); the field is retained in MissionConfig only for schema
    back-compat.  A direct `conf_threshold=`/`special_thresholds=` argument to
    `Detector` overrides the resolved values (for experiments/tests).

D11 PER-CLASS THRESHOLD RESOLUTION (name -> id, once, at startup)
    `SPECIAL_CONFIDENCE_THRESHOLDS` is keyed by class NAME (ids are an
    export-time accident of alphabetical ordering - see data.yaml).
    `resolve_special_thresholds` translates name -> id once, in
    `build_detector`, against the `class_names` mapping the caller passes in
    (the model's own id->name mapping, e.g. loaded from the exported model's
    metadata).  A name with no match in `class_names` raises
    `DetectorConfigError` immediately - it never silently falls back to the
    default threshold for a class that should have had its own.  Per-frame
    filtering (`_postprocess`) only ever looks up by already-resolved
    integer class_id - no name lookups happen on the hot path.

D12 THRESHOLD-VS-NMS ORDERING
    Filtering happens before NMS in `_postprocess`, structurally unchanged
    from before this became class-aware.  This is intentionally equivalent
    to filtering after NMS: NMS is strictly per-class (D4) and the
    per-class threshold is constant within a class, so a sub-threshold box
    can never survive NMS only to be caught by a later filter, and it can
    never suppress a box that would otherwise have survived - the kept set
    is identical either way.  Filtering first is simply cheaper (fewer boxes
    reach the O(n^2) NMS pass).

D13 CLASS-MASKING HAPPENS BEFORE ARGMAX, IN DECODE - GAP-2, RESOLVED
    `Detector.target_class_ids` (settable any time via `set_target_classes`,
    no rebuild needed) is threaded into the live decode (`_decode_yolov8`,
    via `DetectorBackend.infer`) and applied by `_masked_class_argmax`
    BEFORE the per-anchor argmax runs - not as a filter on an
    already-decided winner.  This is the single canonical implementation of
    "restrict class-score columns to a target set, then argmax within that
    set"; `training/export.py`'s offline `masked_argmax_predictions` calls
    this exact function rather than keeping its own copy, so the two can
    never silently diverge.

    Proven failure mode this closes (formerly core/mission.py's GAP-2,
    ASSUMPTION MS9): a post-hoc filter applied AFTER an unmasked argmax
    drops a true target detection whenever some other (non-target) class
    happens to win that anchor's argmax across the *full* class field, even
    if the target's own score would have cleared ITS threshold.  Concretely:
    target score 0.30 (>= its 0.25 special threshold) at an anchor where a
    non-target class scores 0.35 (< that non-target's own 0.60 default
    threshold) - unmasked, the row is dropped entirely (0.35 loses to
    threshold, and the target never got to compete); masked to the briefed
    set, the target wins locally among just the briefed classes and
    survives at 0.30 >= 0.25.  When `target_class_ids` is `None` (the
    default), decode is byte-identical to the pre-masking behavior - every
    training/validation script (train.py, masked_validation_experiment.py,
    resolution_and_remap_report.py, threshold_sweep_report.py) depends on
    this and is never masked.

D3  NMS IoU
    Default NMS IoU is 0.45.  config.mission.iou_threshold is the *tracker's*
    association IoU - a semantically different knob - and is deliberately NOT
    reused here.  When a detector-NMS field is added to config, wire it in
    `build_detector`; until then this default stands.

D4  NMS IS PER-CLASS
    Boxes of different classes never suppress one another; NMS runs
    independently within each class_id.

D5  NMS COORDINATE SPACE
    NMS runs in model-input (letterboxed) space.  Letterboxing is a uniform
    scale plus translation, under which IoU is invariant, so the result is
    identical to running NMS in original-frame space.

D6  LETTERBOX / NORMALIZATION
    Preprocess letterbox-resizes BGR uint8 to `input_size`, preserving aspect
    ratio, padding with value 114 (the common YOLO grey).  Per-model
    normalization (BGR->RGB, /255, HWC->CHW, batch dim) is done *inside* each
    real backend's `infer`, because it is model-specific; the stub needs none.

D7  RESIZE INTERPOLATION
    The letterbox resize is nearest-neighbour in pure numpy to keep the
    module dependency-free (numpy only).  A real deployment may swap in
    cv2/Pillow for interpolation quality and speed; pixel quality is
    irrelevant to the stub and, until real weights exist, to correctness -
    only the coordinate transform matters, and that is exact.

D8  RAW OUTPUT LAYOUT (real backends)
    `_decode_yolov8` assumes a YOLOv8-style output tensor (cx,cy,w,h + per-
    class scores, no separate objectness).  This is UNVERIFIED scaffolding:
    it will be confirmed or rewritten against the actual exported model.  An
    unrecognized output shape raises, which `detect` catches and turns into
    an empty result rather than fabricating boxes.

D9  coord IS ALWAYS None
    Every emitted Detection has coord=None; the geolocate layer fills GPS
    later.  The detector has no notion of world coordinates.

D10 NEVER CRASH THE PIPELINE
    A None/empty/mis-shaped frame, or any backend inference error, is caught,
    logged via get_logger(__name__), and turned into `[]`.  detect() does not
    raise into the mission loop.

USAGE NOTE
----------
    detector = build_detector(cfg, backend="stub")   # or "ncnn" on the Pi
    # with a real model, pass its id->name mapping to enable the per-class
    # SPECIAL_CONFIDENCE_THRESHOLDS overrides (ASSUMPTION D11):
    detector = build_detector(
        cfg, backend="ncnn", param_path=..., bin_path=...,
        class_names={0: "Afghanistan", ..., 50: "Egypt", ...},
    )
    # per frame:
    detections = detector.detect(frame)              # frame: (H,W,3) BGR uint8
    tracker.update(detections, frame_idx)
`build_detector` is called once at startup; `detect` runs every frame.
================================================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence, TYPE_CHECKING

import numpy as np

from ioutils.logging_cfg import get_logger
from vision.tracker import Detection

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids importing config at runtime
    from core.config import AppConfig

log = get_logger(__name__)

Box = tuple[float, float, float, float]

# ASSUMPTION D1 / D3 / D6: module-wide defaults, all overridable.
_DEFAULT_INPUT_SIZE = 640
_DEFAULT_CONF_THRESHOLD = 0.25
_DEFAULT_NMS_IOU = 0.45
_LETTERBOX_PAD_VALUE = 114  # ASSUMPTION D6


# ---------------------------------------------------------------------------
# Letterbox transform - the single most bug-prone piece, so it is isolated,
# exact, and independently testable.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    """Maps boxes between original-frame pixels and model-input pixels.

    A letterbox is a uniform resize by `scale` followed by a constant pad
    offset `(pad_x, pad_y)` placing the resized image inside the square model
    canvas.  Forward maps original -> model; inverse maps model -> original.

    Attributes:
        scale: Uniform resize factor applied to the original frame.
        pad_x: Left padding, in model-input pixels.
        pad_y: Top padding, in model-input pixels.
        input_size: Side length of the square model canvas.
        orig_w: Original frame width in pixels.
        orig_h: Original frame height in pixels.
    """

    scale: float
    pad_x: float
    pad_y: float
    input_size: int
    orig_w: int
    orig_h: int

    def forward_box(self, box: Box) -> Box:
        """Map an original-frame box (x1,y1,x2,y2) into model-input pixels."""
        x1, y1, x2, y2 = box
        return (
            x1 * self.scale + self.pad_x,
            y1 * self.scale + self.pad_y,
            x2 * self.scale + self.pad_x,
            y2 * self.scale + self.pad_y,
        )

    def inverse_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """Map an (N,4) array of model-space boxes back to original pixels.

        The result is clamped to the original frame bounds so downstream code
        never receives coordinates outside the image.
        """
        if boxes.size == 0:
            return boxes.reshape(0, 4).astype(np.float64)
        out = boxes.astype(np.float64).copy()
        out[:, 0] = (out[:, 0] - self.pad_x) / self.scale
        out[:, 1] = (out[:, 1] - self.pad_y) / self.scale
        out[:, 2] = (out[:, 2] - self.pad_x) / self.scale
        out[:, 3] = (out[:, 3] - self.pad_y) / self.scale
        # Clamp to frame bounds (ASSUMPTION D10-adjacent: never emit off-frame px).
        out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0.0, float(self.orig_w))
        out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0.0, float(self.orig_h))
        return out


def letterbox(
    frame: np.ndarray, input_size: int = _DEFAULT_INPUT_SIZE
) -> tuple[np.ndarray, LetterboxTransform]:
    """Letterbox-resize a BGR frame to `input_size` x `input_size`.

    Aspect ratio is preserved; the shorter side is padded with
    `_LETTERBOX_PAD_VALUE`.  Returns the model-input canvas (HWC BGR uint8)
    and the exact `LetterboxTransform` used, so postprocess can invert it.

    Args:
        frame: Original frame, (H, W, 3) BGR uint8.
        input_size: Side length of the square model canvas.

    Returns:
        (model_input, transform).
    """
    orig_h, orig_w = int(frame.shape[0]), int(frame.shape[1])
    scale = min(input_size / orig_w, input_size / orig_h)
    # Integer content dimensions; pads are exact so the round-trip is exact.
    new_w = min(int(round(orig_w * scale)), input_size)
    new_h = min(int(round(orig_h * scale)), input_size)
    dw = input_size - new_w
    dh = input_size - new_h
    left = dw // 2
    top = dh // 2

    # ASSUMPTION D7: nearest-neighbour resize in pure numpy (no cv2 dependency).
    src_x = np.minimum((np.arange(new_w) / scale).astype(np.int64), orig_w - 1)
    src_y = np.minimum((np.arange(new_h) / scale).astype(np.int64), orig_h - 1)
    resized = frame[src_y][:, src_x]

    canvas = np.full(
        (input_size, input_size, 3), _LETTERBOX_PAD_VALUE, dtype=np.uint8
    )
    canvas[top : top + new_h, left : left + new_w] = resized

    transform = LetterboxTransform(
        scale=float(scale),
        pad_x=float(left),
        pad_y=float(top),
        input_size=int(input_size),
        orig_w=orig_w,
        orig_h=orig_h,
    )
    return canvas, transform


# ---------------------------------------------------------------------------
# Raw backend output - the small struct every backend returns.  Boxes are in
# MODEL-INPUT pixel space (x1,y1,x2,y2); the Detector inverts the letterbox.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RawDetections:
    """Un-postprocessed detector output in model-input coordinates.

    Attributes:
        boxes: (N, 4) float array of x1,y1,x2,y2 in model-input pixels.
        scores: (N,) float array of confidences in [0, 1].
        class_ids: (N,) integer array of class labels.
    """

    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray

    @classmethod
    def empty(cls) -> "RawDetections":
        """Return an empty RawDetections with correctly-shaped arrays."""
        return cls(
            boxes=np.zeros((0, 4), dtype=np.float64),
            scores=np.zeros((0,), dtype=np.float64),
            class_ids=np.zeros((0,), dtype=np.int64),
        )


# ---------------------------------------------------------------------------
# Backend seam.
# ---------------------------------------------------------------------------
class DetectorBackend(ABC):
    """Abstract inference backend: model-input frame -> RawDetections.

    Concrete backends are selected at construction.  A backend receives the
    letterboxed model-input canvas plus the `LetterboxTransform` that produced
    it and returns raw boxes in model-input pixel space.  Real model backends
    ignore the transform (their net already outputs in model space); the stub
    uses it to place scripted original-frame detections.
    """

    @abstractmethod
    def infer(
        self,
        model_input: np.ndarray,
        transform: LetterboxTransform,
        target_class_ids: Iterable[int] | None = None,
    ) -> RawDetections:
        """Run inference on one model-input canvas and return raw detections.

        Args:
            model_input: Letterboxed model-input canvas.
            transform: The `LetterboxTransform` that produced `model_input`.
            target_class_ids: `Detector.target_class_ids`, forwarded so a
                real backend's decode can mask BEFORE argmax (ASSUMPTION
                D13).  `None` means unmasked.  Backends with no raw
                class-score tensor to mask (e.g. `StubBackend`) ignore it.
        """
        raise NotImplementedError


class StubBackend(DetectorBackend):
    """Canned backend for tests and model-free pipeline development.

    Scripted with either a fixed list of `tracker.Detection` (in ORIGINAL
    frame pixels) or a callable that produces such a list from the frame.
    Those detections are forward-projected into model-input space so they flow
    through the real threshold / NMS / inverse-letterbox path unchanged - i.e.
    the stub exercises the same postprocess a real backend would, and (barring
    threshold/NMS) round-trips back to the scripted boxes.
    """

    def __init__(
        self,
        scripted: Sequence[Detection] | Callable[[np.ndarray], Sequence[Detection]],
    ) -> None:
        """Create a stub backend.

        Args:
            scripted: Either the detections to emit every frame, or a callable
                mapping the original frame to detections for that frame.
        """
        self._scripted = scripted

    def infer(
        self,
        model_input: np.ndarray,
        transform: LetterboxTransform,
        target_class_ids: Iterable[int] | None = None,
    ) -> RawDetections:
        """Return the scripted detections, projected into model-input space.

        `target_class_ids` is accepted (for ABC compatibility) but ignored:
        a scripted `Detection` already has its class_id fixed by the test
        author, so there is no raw class-score matrix here to mask before
        argmax - masking a stub would not exercise ASSUMPTION D13 anyway.
        """
        if callable(self._scripted):
            # The callable is scripted against the ORIGINAL frame; reconstruct
            # it from the model input is impossible, so callables receive the
            # model_input canvas.  Tests that need original-frame logic script
            # a fixed list instead.
            detections = list(self._scripted(model_input))
        else:
            detections = list(self._scripted)

        if not detections:
            return RawDetections.empty()

        boxes = np.array(
            [transform.forward_box(det.box) for det in detections], dtype=np.float64
        )
        scores = np.array([float(det.score) for det in detections], dtype=np.float64)
        class_ids = np.array([int(det.class_id) for det in detections], dtype=np.int64)
        return RawDetections(boxes=boxes, scores=scores, class_ids=class_ids)


class OnnxBackend(DetectorBackend):
    """ONNX Runtime backend for laptop dev on recorded video.

    UNVERIFIED SCAFFOLDING - # requires trained model.  This wires the runtime
    and shows exactly where weights load and where output is decoded; it makes
    no accuracy claim and cannot until real weights exist.
    """

    def __init__(self, weights_path: str, input_size: int = _DEFAULT_INPUT_SIZE) -> None:
        """Load an ONNX model.  Imports onnxruntime lazily.

        Args:
            weights_path: Path to the exported .onnx model.  # requires trained model
            input_size: Model input side length.
        """
        import onnxruntime  # lazy: not needed to import this module

        self._session = onnxruntime.InferenceSession(
            weights_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._input_size = input_size

    def infer(
        self,
        model_input: np.ndarray,
        transform: LetterboxTransform,
        target_class_ids: Iterable[int] | None = None,
    ) -> RawDetections:
        """Normalize, run the session, and decode raw YOLOv8-style output."""
        blob = _to_model_blob(model_input)  # BGR->RGB, /255, HWC->CHW, batch dim
        outputs = self._session.run(None, {self._input_name: blob})
        return _decode_yolov8(outputs[0], self._input_size, target_class_ids)


class TorchBackend(DetectorBackend):
    """PyTorch backend for laptop dev.

    UNVERIFIED SCAFFOLDING - # requires trained model.
    """

    def __init__(self, weights_path: str, input_size: int = _DEFAULT_INPUT_SIZE) -> None:
        """Load a TorchScript model.  Imports torch lazily.

        Args:
            weights_path: Path to a traced/scripted .pt model.  # requires trained model
            input_size: Model input side length.
        """
        import torch  # lazy

        self._torch = torch
        self._model = torch.jit.load(weights_path, map_location="cpu")
        self._model.eval()
        self._input_size = input_size

    def infer(
        self,
        model_input: np.ndarray,
        transform: LetterboxTransform,
        target_class_ids: Iterable[int] | None = None,
    ) -> RawDetections:
        """Normalize, run the model, and decode raw YOLOv8-style output."""
        blob = _to_model_blob(model_input)
        with self._torch.no_grad():
            tensor = self._torch.from_numpy(blob)
            output = self._model(tensor)
        array = output.cpu().numpy() if hasattr(output, "cpu") else np.asarray(output)
        return _decode_yolov8(array, self._input_size, target_class_ids)


class NcnnBackend(DetectorBackend):
    """ncnn backend for Raspberry Pi 5 deployment.

    UNVERIFIED SCAFFOLDING - # requires trained model.  ncnn splits a model
    into a .param (topology) and a .bin (weights); both come from exporting
    the trained model.
    """

    def __init__(
        self,
        param_path: str,
        bin_path: str,
        input_size: int = _DEFAULT_INPUT_SIZE,
        input_name: str = "in0",
        output_name: str = "out0",
    ) -> None:
        """Load an ncnn model.  Imports ncnn lazily.

        Args:
            param_path: Path to the .param topology.  # requires trained model
            bin_path: Path to the .bin weights.        # requires trained model
            input_size: Model input side length.
            input_name / output_name: Blob names in the exported graph.
        """
        import ncnn  # lazy

        self._ncnn = ncnn
        self._net = ncnn.Net()  # type: ignore[attr-defined]  # ncnn is a compiled C-extension with no type stubs — confirmed to work correctly at runtime (see diagnostics_report.md C-5)
        self._net.load_param(param_path)
        self._net.load_model(bin_path)
        self._input_size = input_size
        self._input_name = input_name
        self._output_name = output_name

    def infer(
        self,
        model_input: np.ndarray,
        transform: LetterboxTransform,
        target_class_ids: Iterable[int] | None = None,
    ) -> RawDetections:
        """Run the ncnn extractor and decode raw YOLOv8-style output."""
        # ncnn Mat construction / normalization is model-specific; this is the
        # seam where a verified decode goes once weights exist.
        ncnn = self._ncnn
        mat_in = ncnn.Mat.from_pixels(  # type: ignore[attr-defined]  # ncnn is a compiled C-extension with no type stubs — confirmed to work correctly at runtime (see diagnostics_report.md C-5)
            np.ascontiguousarray(model_input[:, :, ::-1]),  # BGR->RGB
            ncnn.Mat.PixelType.PIXEL_RGB,  # type: ignore[attr-defined]  # ncnn is a compiled C-extension with no type stubs — confirmed to work correctly at runtime (see diagnostics_report.md C-5)
            self._input_size,
            self._input_size,
        )
        mat_in.substract_mean_normalize([], [1 / 255.0, 1 / 255.0, 1 / 255.0])
        extractor = self._net.create_extractor()
        extractor.input(self._input_name, mat_in)
        _, mat_out = extractor.extract(self._output_name)
        array = np.array(mat_out)[None, ...]  # add batch dim for the shared decoder
        return _decode_yolov8(array, self._input_size, target_class_ids)


def _to_model_blob(model_input: np.ndarray) -> np.ndarray:
    """Convert an HWC BGR uint8 canvas to a normalized NCHW RGB float32 blob.

    ASSUMPTION D6: BGR->RGB, /255, HWC->CHW, batch dim.  Real models may want a
    different normalization; adjust when weights exist.
    """
    rgb = model_input[:, :, ::-1].astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    return np.ascontiguousarray(chw[None, ...])


def _orient_attrs(output: np.ndarray) -> np.ndarray:
    """Orient a raw (1, 4+nc, N) / (1, N, 4+nc) tensor to rows-are-attrs form.

    Squeezes the batch dim, then picks whichever axis is smaller as the
    attribute axis (4+nc is a few hundred at most; num_anchors is thousands
    for a 640 input) - valid as long as num_anchors comfortably exceeds
    4+nc.  Isolated from `_decode_yolov8` so export.py's layout inspection
    and masking checks share this exact orientation logic instead of
    re-deriving it - the orientation heuristic is the single most bug-prone
    piece of ASSUMPTION D8.
    """
    arr = np.asarray(output)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise RuntimeError(f"unexpected model output ndim: {arr.shape}")
    # Orient so rows are attributes (4+nc) and columns are anchors (N).
    if arr.shape[0] < arr.shape[1]:
        attrs = arr  # (4+nc, N)
    else:
        attrs = arr.T
    if attrs.shape[0] < 5:
        raise RuntimeError(f"unexpected model output layout: {arr.shape}")
    return attrs


def _masked_class_argmax(
    class_scores: np.ndarray, target_class_ids: Iterable[int] | None
) -> tuple[np.ndarray, np.ndarray]:
    """Score + winning class id per anchor, via argmax, optionally masked.

    THE canonical implementation of "restrict class-score columns to a
    target set, then argmax within that restricted set" (ASSUMPTION D13).
    Used identically by the live decode (`_decode_yolov8`, driven by
    `Detector.target_class_ids`) and by `training/export.py`'s offline
    `masked_argmax_predictions`, which imports and calls this exact
    function rather than keeping its own copy - there is exactly one
    implementation of this mechanism in the codebase.

    Masking happens BEFORE argmax: restricting the score matrix's columns to
    `target_class_ids` first means a non-target class is structurally
    incapable of winning a row - unlike a downstream filter applied to an
    already-decided (full-field) argmax winner, which can and does drop a
    valid target detection whenever some non-target class outscored it
    across the *entire* class field (see ASSUMPTION D13 for the proven
    failure mode this replaces).

    Args:
        class_scores: (N, nc) per-anchor per-class scores.
        target_class_ids: If given (non-empty), restrict the argmax to
            these class ids - the returned class_ids are always one of
            them, never any other.  `None` means unmasked: argmax over
            every column, byte-identical to the pre-masking decode.

    Returns:
        (scores, class_ids): each shape (N,), dtype float64 / int64.

    Raises:
        ValueError: if `target_class_ids` is given but empty, or any id in
            it is out of range for `class_scores`'s column count.
    """
    if target_class_ids is None:
        scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1).astype(np.int64)
        return scores.astype(np.float64), class_ids

    num_classes = class_scores.shape[1]
    target_ids = np.asarray(
        sorted({int(class_id) for class_id in target_class_ids}), dtype=np.int64
    )
    if target_ids.size == 0:
        raise ValueError(
            "target_class_ids must be non-empty; pass None for unmasked"
        )
    out_of_range = target_ids[(target_ids < 0) | (target_ids >= num_classes)]
    if out_of_range.size > 0:
        raise ValueError(
            f"target class id {int(out_of_range[0])} out of range [0, {num_classes})"
        )

    masked_scores = class_scores[:, target_ids]  # (N, K)
    local_argmax = masked_scores.argmax(axis=1)
    scores = masked_scores[np.arange(masked_scores.shape[0]), local_argmax]
    class_ids = target_ids[local_argmax]
    return scores.astype(np.float64), class_ids.astype(np.int64)


def _decode_yolov8(
    output: np.ndarray,
    input_size: int,
    target_class_ids: Iterable[int] | None = None,
) -> RawDetections:
    """Decode a YOLOv8-style output tensor into model-space RawDetections.

    UNVERIFIED (ASSUMPTION D8).  Expects (1, 4+nc, N) or (1, N, 4+nc) with box
    rows cx,cy,w,h (model-input pixels) followed by per-class scores and no
    separate objectness.  Score and class are decided by `_masked_class_argmax`
    (ASSUMPTION D13): the max/argmax class probability when `target_class_ids`
    is `None` (byte-identical to the original unmasked decode), or the
    masked argmax restricted to `target_class_ids` otherwise - a non-target
    class can never be emitted when masking is active.  An unrecognized
    shape raises RuntimeError, which `detect` turns into an empty result
    rather than inventing boxes.
    """
    attrs = _orient_attrs(output)

    boxes_cxcywh = attrs[:4, :].T  # (N, 4)
    class_scores = attrs[4:, :].T  # (N, nc)
    scores, class_ids = _masked_class_argmax(class_scores, target_class_ids)  # ASSUMPTION D13

    cx, cy, w, h = (
        boxes_cxcywh[:, 0],
        boxes_cxcywh[:, 1],
        boxes_cxcywh[:, 2],
        boxes_cxcywh[:, 3],
    )
    boxes = np.stack(
        [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=1
    )
    return RawDetections(
        boxes=boxes.astype(np.float64),
        scores=scores.astype(np.float64),
        class_ids=class_ids,
    )


# ---------------------------------------------------------------------------
# NMS - pure numpy, per class, deterministic.
# ---------------------------------------------------------------------------
def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one (4,) box against an (M,4) array of boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area + areas - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0.0, inter / union, 0.0)
    return iou


def _nms_single_class(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Greedy NMS within one class; returns kept indices into `boxes`.

    Deterministic: candidates are ranked by descending score with a stable
    sort, so equal scores keep input order.
    """
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        iou = _iou_one_to_many(boxes[i], boxes[rest])
        order = rest[iou <= iou_thresh]
    return keep


# ---------------------------------------------------------------------------
# Per-class threshold resolution (name -> id, once, at startup).  ASSUMPTION D11.
# ---------------------------------------------------------------------------
class DetectorConfigError(Exception):
    """Raised at startup when a briefed class name has no matching class id.

    Never raised per-frame - only from `resolve_special_thresholds`, called
    once by `build_detector` during initialization (ASSUMPTION D11).
    """


def resolve_special_thresholds(
    class_names: Mapping[int, str],
    special_thresholds_by_name: Mapping[str, float],
) -> dict[int, float]:
    """Translate name-keyed thresholds into id-keyed thresholds.

    `class_names` is the model's own id -> name mapping (e.g. loaded from the
    exported model's metadata - the "already used by the detector" mapping).
    Every key of `special_thresholds_by_name` (e.g.
    `core.config.SPECIAL_CONFIDENCE_THRESHOLDS`) must resolve to exactly one
    id in `class_names`.  Class ids are never hardcoded here or anywhere
    downstream - only names are, and only in config.py.

    Args:
        class_names: class_id -> class name, as known by the loaded model.
        special_thresholds_by_name: class name -> confidence threshold.

    Returns:
        class_id -> confidence threshold.

    Raises:
        DetectorConfigError: if any name in `special_thresholds_by_name` has
            no matching id in `class_names` - fails loud, never falls back
            to the default threshold for a class that should have had its
            own (ASSUMPTION D11).
    """
    name_to_id: dict[str, int] = {
        name: int(class_id) for class_id, name in class_names.items()
    }

    resolved: dict[int, float] = {}
    missing: list[str] = []
    for name, threshold in special_thresholds_by_name.items():
        class_id = name_to_id.get(name)
        if class_id is None:
            missing.append(name)
            continue
        resolved[class_id] = float(threshold)

    if missing:
        raise DetectorConfigError(
            "SPECIAL_CONFIDENCE_THRESHOLDS names not found in the model's "
            f"class-name mapping: {sorted(missing)!r}. Known class names "
            f"({len(class_names)}): {sorted(class_names.values())!r}"
        )
    return resolved


# ---------------------------------------------------------------------------
# Orchestrator.
# ---------------------------------------------------------------------------
class Detector:
    """Owns preprocess -> infer -> postprocess for one frame.

    Stateless across frames (the backend may hold a model session).  Emits
    `tracker.Detection` objects with boxes in ORIGINAL-frame pixels and
    coord=None.  Never raises out of `detect` (ASSUMPTION D10).
    """

    def __init__(
        self,
        backend: DetectorBackend,
        input_size: int = _DEFAULT_INPUT_SIZE,
        conf_threshold: float = _DEFAULT_CONF_THRESHOLD,
        nms_iou: float = _DEFAULT_NMS_IOU,
        special_thresholds: Mapping[int, float] | None = None,
        target_class_ids: Sequence[int] | None = None,
    ) -> None:
        """Create a detector around a backend.

        Args:
            backend: The inference backend (stub or a real model backend).
            input_size: Square model-input side length.  # ASSUMPTION D1
            conf_threshold: Default minimum score to keep a detection, used
                for every class not present in `special_thresholds`.  # ASSUMPTION D2
            nms_iou: IoU above which same-class boxes are suppressed.  # ASSUMPTION D3
            special_thresholds: Optional class_id -> minimum score overrides
                for specific classes (e.g. the four low-confidence briefed
                classes).  Already id-keyed and already resolved from names -
                see `resolve_special_thresholds` / `build_detector`.  Absent
                or empty means every class uses `conf_threshold`, identical
                to the pre-class-aware behavior.  # ASSUMPTION D11
            target_class_ids: Optional set of briefed class ids to mask the
                live decode to (ASSUMPTION D13) - general, any non-empty
                count, not hardcoded to any particular number.  `None`
                (default) means unmasked: decode argmaxes over every class
                the model knows, byte-identical to the original behavior.
                Update later without rebuilding via `set_target_classes`.
        """
        if input_size <= 0:
            raise ValueError(f"input_size must be positive; got {input_size!r}")
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError(f"conf_threshold must be in [0, 1]; got {conf_threshold!r}")
        if not 0.0 <= nms_iou <= 1.0:
            raise ValueError(f"nms_iou must be in [0, 1]; got {nms_iou!r}")
        if special_thresholds:
            for class_id, threshold in special_thresholds.items():
                if not 0.0 <= threshold <= 1.0:
                    raise ValueError(
                        f"special_thresholds[{class_id!r}] must be in [0, 1]; "
                        f"got {threshold!r}"
                    )

        self.backend = backend
        self.input_size = int(input_size)
        self.conf_threshold = float(conf_threshold)
        self.nms_iou = float(nms_iou)
        self.special_thresholds: dict[int, float] = (
            {int(class_id): float(threshold) for class_id, threshold in special_thresholds.items()}
            if special_thresholds
            else {}
        )
        self.target_class_ids: frozenset[int] | None = None
        self.set_target_classes(target_class_ids)

    def threshold_for_class(self, class_id: int) -> float:
        """Confidence threshold this detector applies to `class_id`.

        Mirrors: threshold = SPECIAL_THRESHOLDS.get(class_id, DEFAULT_THRESHOLD).
        """
        return self.special_thresholds.get(int(class_id), self.conf_threshold)

    def set_target_classes(self, target_class_ids: Sequence[int] | None) -> None:
        """Update the briefed target-class mask without rebuilding the detector.

        Masking happens in the live decode (before argmax - ASSUMPTION D13),
        so this takes effect on the very next `detect()` call.  Safe to call
        any time between frames - the briefed set is typically only known
        ~1 day before the competition.  Works for any non-empty count of
        target ids (1, 4, 10, ...) - nothing here hardcodes a specific
        count or any specific class id.

        Args:
            target_class_ids: The new briefed set, or `None` to revert to
                unmasked (full-class argmax, ASSUMPTION D8's original
                behavior).

        Raises:
            ValueError: if given an empty (but non-None) sequence - "mask
                to nothing" is never a valid request; pass `None` instead.
        """
        if target_class_ids is None:
            self.target_class_ids = None
            return
        ids = frozenset(int(class_id) for class_id in target_class_ids)
        if not ids:
            raise ValueError(
                "target_class_ids must be non-empty; pass None for unmasked"
            )
        self.target_class_ids = ids

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect flags in one frame.  Returns [] on any bad input or error.

        Args:
            frame: (H, W, 3) BGR uint8 image.

        Returns:
            List of `tracker.Detection` in original-frame pixels, coord=None.
        """
        if not self._is_valid_frame(frame):
            log.warning(
                "detector received an invalid frame (type=%s); returning no detections",
                type(frame).__name__,
            )
            return []

        try:
            model_input, transform = letterbox(frame, self.input_size)
            raw = self.backend.infer(model_input, transform, self.target_class_ids)
            return self._postprocess(raw, transform)
        except Exception:  # ASSUMPTION D10: never crash the pipeline
            log.exception("detector inference failed; returning no detections")
            return []

    @staticmethod
    def _is_valid_frame(frame: object) -> bool:
        """Return True only for a non-empty (H, W, 3) numpy image."""
        return (
            isinstance(frame, np.ndarray)
            and frame.ndim == 3
            and frame.shape[2] == 3
            and frame.shape[0] > 0
            and frame.shape[1] > 0
        )

    def _postprocess(
        self, raw: RawDetections, transform: LetterboxTransform
    ) -> list[Detection]:
        """Threshold, per-class NMS, invert letterbox, build Detections."""
        boxes, scores, class_ids = raw.boxes, raw.scores, raw.class_ids
        if boxes.shape[0] == 0:
            return []

        # 1. Confidence threshold - class-aware.  # ASSUMPTION D2 / D11 / D12
        if self.special_thresholds:
            thresholds = np.full(scores.shape, self.conf_threshold, dtype=np.float64)
            for class_id, threshold in self.special_thresholds.items():
                thresholds[class_ids == class_id] = threshold
            keep_mask = scores >= thresholds
        else:
            keep_mask = scores >= self.conf_threshold
        boxes = boxes[keep_mask]
        scores = scores[keep_mask]
        class_ids = class_ids[keep_mask]
        if boxes.shape[0] == 0:
            return []

        # 2. Per-class NMS in model space.  # ASSUMPTION D4 / D5
        kept: list[int] = []
        for cls in np.unique(class_ids):
            cls_idx = np.nonzero(class_ids == cls)[0]
            local_keep = _nms_single_class(boxes[cls_idx], scores[cls_idx], self.nms_iou)
            kept.extend(int(cls_idx[i]) for i in local_keep)

        if not kept:
            return []
        boxes = boxes[kept]
        scores = scores[kept]
        class_ids = class_ids[kept]

        # 3. Invert the letterbox back to original-frame pixels (+ clamp).
        orig_boxes = transform.inverse_boxes(boxes)

        # 4. Build Detections.  Cast to plain Python scalars: tracker.Detection
        #    rejects numpy scalar types.  Drop degenerate (zero-area) boxes.
        detections: list[Detection] = []
        for i in range(orig_boxes.shape[0]):
            x1, y1, x2, y2 = (float(v) for v in orig_boxes[i])
            if x2 <= x1 or y2 <= y1:
                log.debug("dropping degenerate box after clamp: %s", (x1, y1, x2, y2))
                continue
            detections.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    class_id=int(class_ids[i]),
                    score=float(scores[i]),
                    coord=None,  # ASSUMPTION D9
                )
            )
        return detections


# ---------------------------------------------------------------------------
# Factory.
# ---------------------------------------------------------------------------
def build_detector(
    cfg: "AppConfig",
    backend: str = "stub",
    *,
    input_size: int | None = None,
    scripted: Sequence[Detection] | Callable[[np.ndarray], Sequence[Detection]] | None = None,
    weights: str | None = None,
    param_path: str | None = None,
    bin_path: str | None = None,
    class_names: Mapping[int, str] | None = None,
) -> Detector:
    """Wire a `Detector` from config and a backend name.

    The confidence threshold is class-aware (ASSUMPTION D2 / D11):
    `core.config.DEFAULT_CONFIDENCE_THRESHOLD` is the fallback for every
    class, and `core.config.SPECIAL_CONFIDENCE_THRESHOLDS` (name-keyed)
    overrides it for specific classes.  `cfg.mission.confidence_threshold`
    is NOT read here (config.py ASSUMPTION T3).  NMS IoU uses the module
    default (ASSUMPTION D3) - config's `iou_threshold` is the tracker's
    association IoU and is NOT reused here.  Heavy backends are imported
    lazily inside their constructor.

    Args:
        cfg: Validated application config (see config.AppConfig).
        backend: One of "stub", "onnx", "torch", "ncnn".
        input_size: Override the default model input side length.  # ASSUMPTION D1
        scripted: Detections (or a frame->detections callable) for the stub.
            Defaults to an empty list, i.e. a no-op detector useful for wiring
            the pipeline before a model exists.
        weights: Path to .onnx or .pt weights (for "onnx"/"torch").
        param_path / bin_path: ncnn model files (for "ncnn").
        class_names: The model's own class_id -> class name mapping (e.g.
            loaded from the exported model's metadata.yaml `names:` section).
            When given, every name in `SPECIAL_CONFIDENCE_THRESHOLDS` is
            resolved to its id right now, at startup - a name with no match
            raises `DetectorConfigError` (ASSUMPTION D11).  When omitted
            (e.g. the "stub" backend during pipeline dev with no trained
            model yet), every class uses `DEFAULT_CONFIDENCE_THRESHOLD`.

    Returns:
        A configured `Detector`.

    Raises:
        ValueError: on an unknown backend name or missing required paths.
        DetectorConfigError: if `class_names` is given but is missing one of
            the names in `SPECIAL_CONFIDENCE_THRESHOLDS`.
    """
    from core.config import DEFAULT_CONFIDENCE_THRESHOLD, SPECIAL_CONFIDENCE_THRESHOLDS

    size = int(input_size) if input_size is not None else _DEFAULT_INPUT_SIZE
    conf_threshold = float(DEFAULT_CONFIDENCE_THRESHOLD)  # ASSUMPTION D2 / T1
    special_thresholds: dict[int, float] = {}
    if class_names is not None:
        special_thresholds = resolve_special_thresholds(
            class_names, SPECIAL_CONFIDENCE_THRESHOLDS
        )  # ASSUMPTION D11 - raises DetectorConfigError on a missing name
    name = backend.lower()

    impl: DetectorBackend
    if name == "stub":
        impl = StubBackend(scripted if scripted is not None else [])
    elif name == "onnx":
        if not weights:
            raise ValueError("backend 'onnx' requires weights=<path to .onnx>")
        impl = OnnxBackend(weights, input_size=size)  # requires trained model
    elif name == "torch":
        if not weights:
            raise ValueError("backend 'torch' requires weights=<path to .pt>")
        impl = TorchBackend(weights, input_size=size)  # requires trained model
    elif name == "ncnn":
        if not param_path or not bin_path:
            raise ValueError("backend 'ncnn' requires param_path=<.param> and bin_path=<.bin>")
        impl = NcnnBackend(param_path, bin_path, input_size=size)  # requires trained model
    else:
        raise ValueError(
            f"unknown backend {backend!r}; expected 'stub', 'onnx', 'torch', or 'ncnn'"
        )

    log.info(
        "built detector: backend=%s input_size=%d conf_threshold=%.3f nms_iou=%.3f "
        "special_thresholds=%s",
        name,
        size,
        conf_threshold,
        _DEFAULT_NMS_IOU,
        special_thresholds,
    )
    return Detector(
        backend=impl,
        input_size=size,
        conf_threshold=conf_threshold,
        nms_iou=_DEFAULT_NMS_IOU,  # ASSUMPTION D3
        special_thresholds=special_thresholds,
    )
