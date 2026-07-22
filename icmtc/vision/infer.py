# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   23/06/2026

"""infer.py - standalone inference/verification utility for Mission 2 flags.

ICMTC 2026, Mission 2. This script is ONLY for looking at a trained model's
predictions on real images. It does NOT train, does NOT export, and never
writes to the weights file it loads.

    python -m vision.infer --weights training/runs/mission2/yolo11n_flags/weights/best.pt \\
        --source test_images/

================================================================================
ASSUMPTIONS (collected here; also marked inline with # ASSUMPTION: <tag>)
================================================================================

I1  DEFAULT WEIGHTS
    training/runs/mission2/yolo11n_flags/weights/best.pt, relative to this
    file (the current Mission 2 training run's stable output). --weights
    overrides it.

I2  INFERENCE ENGINE
    Uses Ultralytics' own `YOLO.predict` for pre/post-process (letterbox,
    NMS, decode) - this file does not reimplement detector.py's ncnn decode
    path. It is a .pt-only sanity-check tool for the training Mac, not the
    Pi deployment path.

I3  DRAWING
    Boxes/labels are drawn manually with OpenCV (`cv2.rectangle` /
    `cv2.putText`), not Ultralytics' built-in `Results.plot()`, so the
    label format (name + confidence) and per-class colors are this file's
    own and stay stable across Ultralytics versions.

I4  OUTPUT LOCATION
    Every run gets a fresh, non-overwriting directory: runs/inference/exp,
    runs/inference/exp2, runs/inference/exp3, ... mirroring Ultralytics'
    own increment-path convention.

I5  DEVICE
    --device auto (default) passes device=None to Ultralytics, which picks
    CUDA/MPS/CPU itself. --device cpu/cuda/mps/0 etc. is passed through
    verbatim.

I6  IMAGE DISCOVERY
    --source may be a single image file or a directory. A directory is
    scanned non-recursively (glob, not rglob) for common image suffixes,
    sorted for deterministic order.

I7  LABEL OUTPUT (--save-txt / --save-conf)
    Written in the standard YOLO format (class x_center y_center width
    height, normalized 0-1), one .txt per image, alongside the saved image
    inside the run directory - never overwriting the loaded label data.

I8  NEVER CRASH ON A BAD IMAGE
    An unreadable/corrupt image within a folder run is reported and
    skipped; it does not abort the remaining images.

USAGE
-----
    python -m vision.infer --weights best.pt --source test_images/ \\
        --conf 0.25 --imgsz 640 --device auto --save-txt --save-conf
================================================================================
"""

from __future__ import annotations

import argparse
import colorsys
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

Box = tuple[float, float, float, float]

# ASSUMPTION I1/I5/I6: module-wide defaults, all CLI-overridable.
_DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parent
    / "training" / "runs" / "mission2" / "yolo11n_flags" / "weights" / "best.pt"
)
_DEFAULT_CONF = 0.25
_DEFAULT_IMGSZ = 640
_DEFAULT_DEVICE = "auto"
_DEFAULT_PROJECT = Path(__file__).resolve().parent / "runs" / "inference"
_DEFAULT_RUN_STEM = "exp"
_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
_BOX_THICKNESS = 2
_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX; avoids importing cv2 at module scope (I2)


class InferError(Exception):
    """Raised for any pre-flight or runtime failure infer.py cannot recover from."""


# ---------------------------------------------------------------------------
# Ultralytics seam (dependency-injectable so tests never need real weights).
# ---------------------------------------------------------------------------
class YoloModel(Protocol):
    """Small Protocol for the Ultralytics YOLO object used by this module."""

    names: dict[int, str]

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
        raise InferError(
            "Ultralytics is required for infer.py. Install it, then rerun infer.py."
        ) from exc
    try:
        return YOLO(weights)  # type: ignore[return-value]  # ultralytics 8.4.90's own stub for .predict()/.names is internally inconsistent (see diagnostics_report.md C-4) — not a project bug
    except Exception as exc:  # noqa: BLE001 - any load failure must be reported
        raise InferError(f"failed to load model weights {weights!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# Validation (requirement 10: fail loud, human-readable).
# ---------------------------------------------------------------------------
def validate_weights(path: str | Path) -> Path:
    """Confirm the weights file exists and is a file.  Raises InferError otherwise."""
    weights_path = Path(path).expanduser()
    if not weights_path.exists():
        raise InferError(
            f"weights file not found: {weights_path}\n"
            "Train a model first, or pass --weights with a valid .pt path."
        )
    if not weights_path.is_file():
        raise InferError(f"weights path is not a file: {weights_path}")
    return weights_path


def validate_source(path: str | Path) -> Path:
    """Confirm the source (image or folder) exists.  Raises InferError otherwise."""
    source_path = Path(path).expanduser()
    if not source_path.exists():
        raise InferError(
            f"source not found: {source_path}\n"
            "Pass --source with a valid image file or folder of images."
        )
    return source_path


def collect_images(source: Path) -> list[Path]:
    """Resolve --source into a sorted list of image paths.  # ASSUMPTION I6

    Raises:
        InferError: if a single-file source is not an image, or a folder
            contains no recognized image files.
    """
    if source.is_file():
        if source.suffix.lower() not in _IMAGE_SUFFIXES:
            raise InferError(
                f"source file is not a recognized image type: {source} "
                f"(expected one of {sorted(_IMAGE_SUFFIXES)})"
            )
        return [source]

    images = sorted(
        (p for p in source.glob("*") if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES),
        key=lambda p: p.name,
    )
    if not images:
        raise InferError(f"no images found in folder: {source}")
    return images


def resolve_device(device: str) -> str | None:
    """Map the CLI --device value to what Ultralytics expects.  # ASSUMPTION I5"""
    return None if device.lower() == "auto" else device


# ---------------------------------------------------------------------------
# Per-class color (requirement: color each class consistently).
# ---------------------------------------------------------------------------
def class_color(class_id: int) -> tuple[int, int, int]:
    """Deterministic, well-spread BGR color for a class id.

    Uses the golden-ratio hue increment so consecutive class ids land far
    apart on the color wheel instead of clustering, with fixed
    saturation/value for readable, non-washed-out boxes.
    """
    hue = (class_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR for OpenCV


# ---------------------------------------------------------------------------
# Detection extraction (mirrors export.py's Ultralytics-Results seam).
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DetectionResult:
    """One drawn/reported detection, in original-image pixel coordinates."""

    box: Box
    class_id: int
    class_name: str
    score: float


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a numpy array, torch tensor, or test double to a numpy array."""
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    to_numpy = getattr(value, "numpy", None)
    if callable(to_numpy):
        value = to_numpy()
    return np.asarray(value)


def extract_detections(result: Any, names: dict[int, str]) -> list[DetectionResult]:
    """Pull (box, class_id, class_name, score) tuples out of one Ultralytics Result."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
        return []

    xyxy = _to_numpy(boxes.xyxy)
    cls = _to_numpy(boxes.cls)
    conf = _to_numpy(boxes.conf)

    detections: list[DetectionResult] = []
    for i in range(xyxy.shape[0]):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        class_id = int(cls[i])
        detections.append(
            DetectionResult(
                box=(x1, y1, x2, y2),
                class_id=class_id,
                class_name=names.get(class_id, str(class_id)),
                score=float(conf[i]),
            )
        )
    return detections


# ---------------------------------------------------------------------------
# Drawing (requirement 4: boxes + class name + confidence, via OpenCV).
# ---------------------------------------------------------------------------
def draw_detections(frame: np.ndarray, detections: Sequence[DetectionResult]) -> np.ndarray:
    """Draw boxes and "name conf" labels onto a copy of `frame`; returns the copy."""
    import cv2  # ASSUMPTION I2/I3: OpenCV is this file's drawing dependency only

    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det.box)
        color = class_color(det.class_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, _BOX_THICKNESS)

        label = f"{det.class_name} {det.score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, _FONT, 0.5, 1)
        label_top = max(y1 - text_h - baseline, 0)
        cv2.rectangle(
            annotated, (x1, label_top), (x1 + text_w, label_top + text_h + baseline), color, -1
        )
        cv2.putText(
            annotated, label, (x1, label_top + text_h), _FONT, 0.5, (255, 255, 255), 1,
        )
    return annotated


# ---------------------------------------------------------------------------
# Output directory (requirement 5: never overwrite a previous run). # ASSUMPTION I4
# ---------------------------------------------------------------------------
def next_run_dir(project: Path, stem: str = _DEFAULT_RUN_STEM) -> Path:
    """Return the next unused `project/<stem>`, `project/<stem>2`, ... directory."""
    project.mkdir(parents=True, exist_ok=True)
    candidate = project / stem
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = project / f"{stem}{suffix}"
    candidate.mkdir(parents=True)
    return candidate


def write_label_txt(
    path: Path, detections: Sequence[DetectionResult], image_shape: tuple[int, int], *,
    save_conf: bool,
) -> None:
    """Write YOLO-format normalized labels for one image.  # ASSUMPTION I7"""
    height, width = image_shape
    lines = []
    for det in detections:
        x1, y1, x2, y2 = det.box
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        row = f"{det.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        if save_conf:
            row += f" {det.score:.6f}"
        lines.append(row)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ---------------------------------------------------------------------------
# Reporting (requirements 6, 9: clean per-image summary).
# ---------------------------------------------------------------------------
def render_detection_summary(detections: Sequence[DetectionResult]) -> str:
    """Render the "Detected N objects" + per-detection name/confidence report."""
    if not detections:
        return "No flags detected."

    ranked = sorted(detections, key=lambda d: -d.score)
    name_width = max(len(d.class_name) for d in ranked) + 4
    lines = [f"Detected {len(ranked)} objects", ""]
    lines.extend(f"{d.class_name.ljust(name_width)}{d.score:.2f}" for d in ranked)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class InferConfig:
    """Full configuration for one `python infer.py` run."""

    weights: Path
    source: Path
    conf: float = _DEFAULT_CONF
    imgsz: int = _DEFAULT_IMGSZ
    device: str = _DEFAULT_DEVICE
    save_txt: bool = False
    save_conf: bool = False
    project: Path = field(default_factory=lambda: _DEFAULT_PROJECT)


@dataclass(frozen=True, slots=True)
class ImageOutcome:
    """Per-image inference result."""

    image_path: Path
    detections: tuple[DetectionResult, ...]
    infer_ms: float
    image_shape: tuple[int, int]  # (height, width)


def print_model_info(model: YoloModel, output_fn: Callable[[str], None] = print) -> None:
    """Print a short human-readable summary of the loaded model."""
    names = getattr(model, "names", {}) or {}
    task = getattr(model, "task", "detect")
    output_fn(
        f"Model loaded: task={task}, classes={len(names)}"
    )


def run_inference_on_image(
    model: YoloModel,
    image_path: Path,
    config: InferConfig,
    run_dir: Path,
    *,
    output_fn: Callable[[str], None] = print,
) -> ImageOutcome | None:
    """Run inference on one image; draw, save, and report.  Returns None on a bad image."""
    import cv2  # ASSUMPTION I2/I3

    frame = cv2.imread(str(image_path))
    if frame is None:
        output_fn(f"  [SKIP] could not decode image: {image_path}")
        return None

    height, width = int(frame.shape[0]), int(frame.shape[1])
    output_fn(f"  Resolution: {width}x{height}")

    start = time.perf_counter()
    results = model.predict(
        source=frame,
        conf=config.conf,
        imgsz=config.imgsz,
        device=resolve_device(config.device),
        verbose=False,
    )
    infer_ms = (time.perf_counter() - start) * 1000.0

    result = results[0]
    names = getattr(model, "names", None) or getattr(result, "names", {})
    detections = extract_detections(result, names)

    annotated = draw_detections(frame, detections)
    out_image_path = run_dir / image_path.name
    cv2.imwrite(str(out_image_path), annotated)

    if config.save_txt:
        write_label_txt(
            out_image_path.with_suffix(".txt"), detections, (height, width),
            save_conf=config.save_conf,
        )

    output_fn(render_detection_summary(detections))
    output_fn(f"  Inference time: {infer_ms:.1f} ms")
    output_fn("")

    return ImageOutcome(
        image_path=image_path,
        detections=tuple(detections),
        infer_ms=infer_ms,
        image_shape=(height, width),
    )


def render_run_summary(outcomes: Sequence[ImageOutcome], run_dir: Path) -> str:
    """Render the final average-time/FPS/save-location report."""
    total_detections = sum(len(o.detections) for o in outcomes)
    avg_ms = sum(o.infer_ms for o in outcomes) / len(outcomes) if outcomes else 0.0
    avg_fps = (1000.0 / avg_ms) if avg_ms > 0 else 0.0

    return "\n".join(
        [
            "==================================================",
            "Inference Summary",
            "==================================================",
            f"Images processed : {len(outcomes)}",
            f"Total detections : {total_detections}",
            f"Average inference time : {avg_ms:.1f} ms",
            f"Average FPS : {avg_fps:.1f}",
            f"Results saved to : {run_dir}",
            "==================================================",
        ]
    )


def run_infer(
    config: InferConfig,
    *,
    yolo_factory: YoloFactory | None = None,
    output_fn: Callable[[str], None] = print,
) -> list[ImageOutcome]:
    """Validate inputs, load the model, and run inference over every image.

    Raises:
        InferError: on missing weights/source, a model load failure, or (for
            a single-image source) a failure to decode that image.
    """
    weights = validate_weights(config.weights)
    source = validate_source(config.source)
    images = collect_images(source)

    factory = yolo_factory or _default_yolo_factory
    model = factory(str(weights))
    print_model_info(model, output_fn)

    run_dir = next_run_dir(config.project)
    output_fn(f"Saving results to {run_dir}")
    output_fn("")

    outcomes: list[ImageOutcome] = []
    total = len(images)
    for index, image_path in enumerate(images, start=1):
        if total > 1:
            output_fn(f"[{index}/{total}] {image_path.name}")
        else:
            output_fn(image_path.name)
        outcome = run_inference_on_image(model, image_path, config, run_dir, output_fn=output_fn)
        if outcome is not None:
            outcomes.append(outcome)

    if not outcomes:
        raise InferError(
            "no image could be processed (all sources failed to decode); "
            "see the [SKIP] lines above"
        )

    output_fn(render_run_summary(outcomes, run_dir))
    return outcomes


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> InferConfig:
    """Parse CLI arguments into a typed `InferConfig`."""
    parser = argparse.ArgumentParser(
        description="Run YOLO flag-detection inference on an image or folder "
        "and save annotated results. Inference/testing only - never trains "
        "or exports."
    )
    parser.add_argument("--weights", default=str(_DEFAULT_WEIGHTS), help="Path to trained .pt weights.")
    parser.add_argument("--source", required=True, help="Image file or folder of images.")
    parser.add_argument("--conf", type=float, default=_DEFAULT_CONF, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=_DEFAULT_IMGSZ, help="Inference image size.")
    parser.add_argument("--device", default=_DEFAULT_DEVICE, help="'auto', 'cpu', 'cuda', 'mps', or a device id.")
    parser.add_argument("--save-txt", action="store_true", help="Also save YOLO-format label .txt files.")
    parser.add_argument("--save-conf", action="store_true", help="Include confidence in saved label .txt files.")
    parser.add_argument("--project", default=str(_DEFAULT_PROJECT), help="Base directory for run outputs.")
    args = parser.parse_args(argv)

    return InferConfig(
        weights=Path(args.weights).expanduser(),
        source=Path(args.source).expanduser(),
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=Path(args.project).expanduser(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `python infer.py`."""
    config = parse_args(argv)
    try:
        run_infer(config)
    except InferError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI use
    raise SystemExit(main(sys.argv[1:]))
