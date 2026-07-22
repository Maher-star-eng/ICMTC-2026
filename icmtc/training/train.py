# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   21/06/2026

"""YOLO11 training pipeline manager for ICMTC 2026 Mission 2.

This module owns the machine-learning training path from ``data.yaml``
verification through the final validation summary.  It deliberately does not
export NCNN artifacts; export belongs to ``export.py``.

The heavy dependencies (Torch and Ultralytics) are imported lazily so the
validation helpers and unit tests can run on any development machine without a
GPU or installed model runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from ioutils.logging_cfg import configure_logging, get_logger

log = get_logger(__name__)

_IMAGE_SUFFIXES = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
_DEFAULT_DATA_YAML = (
    Path(__file__).resolve().parent / "flag detection.v1i.yolov11" / "data.yaml"
)
_DEFAULT_WEIGHTS = "yolo11n.pt"
_DEFAULT_IMGSZ = 640
_DEFAULT_BATCH = 42
_DEFAULT_EPOCHS = 100
_DEFAULT_WORKERS = 8
_DEFAULT_SEED = 42
_DEFAULT_PATIENCE = 50
_DEFAULT_PROJECT = Path(__file__).resolve().parent / "runs" / "mission2"
_DEFAULT_NAME = "yolo11n_flags"
_STABLE_BEST_MODEL = Path("models/best.pt")
_TRAIN_SUMMARY_JSON = Path("logs/train_summary.json")
_NEGATIVE_RATIO_MIN = 0.10
_NEGATIVE_RATIO_MAX = 0.20
_LOW_CLASS_COUNT_RATIO = 0.25

ResumeMode = Literal["ask", "yes", "no"]


class DatasetError(ValueError):
    """Raised when ``data.yaml`` or YOLO labels fail validation."""


class TrainingError(RuntimeError):
    """Raised when the training pipeline cannot continue safely."""


class YoloModel(Protocol):
    """Small Protocol for the Ultralytics YOLO object used by this module."""

    trainer: Any

    def train(self, **kwargs: Any) -> Any:
        """Train the model using Ultralytics keyword arguments."""

    def val(self, **kwargs: Any) -> Any:
        """Validate the model using Ultralytics keyword arguments."""


class YoloFactory(Protocol):
    """Factory signature for dependency-injected YOLO model construction."""

    def __call__(self, weights: str) -> YoloModel:
        """Return a YOLO-like model loaded from weights."""
        ...


@dataclass(frozen=True, slots=True)
class SplitStats:
    """Verified statistics for one YOLO dataset split."""

    name: str
    source_paths: tuple[Path, ...]
    image_count: int
    label_count: int
    box_count: int
    empty_label_count: int
    missing_label_count: int
    class_counts: tuple[int, ...]

    @property
    def negative_count(self) -> int:
        """Total background images: missing label file or empty label file."""
        return self.missing_label_count + self.empty_label_count


@dataclass(frozen=True, slots=True)
class DatasetStats:
    """Verified dataset-level statistics collected before training."""

    data_yaml: Path
    root: Path
    class_names: tuple[str, ...]
    train: SplitStats
    val: SplitStats
    test: SplitStats | None = None

    @property
    def class_count(self) -> int:
        """Number of classes declared in ``data.yaml``."""
        return len(self.class_names)


@dataclass(frozen=True, slots=True)
class DatasetQualityStats:
    """Non-fatal dataset quality statistics for the training split."""

    negative_images: int
    negative_images_missing_labels: int
    negative_images_empty_labels: int
    positive_images: int
    negative_ratio: float
    per_class_instance_counts: tuple[int, ...]
    min_class_count: int
    max_class_count: int
    median_class_count: float
    mean_class_count: float
    largest_smallest_imbalance: float | None
    low_class_warning_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DeviceSelection:
    """Selected training device and a human-readable label."""

    value: str
    label: str


@dataclass(frozen=True, slots=True)
class ResumeDecision:
    """Decision about whether to resume from an existing ``last.pt``."""

    resume: bool
    checkpoint: Path | None


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """User-configurable training arguments for the Mission 2 run."""

    data_yaml: Path = _DEFAULT_DATA_YAML
    weights: str = _DEFAULT_WEIGHTS
    imgsz: int = _DEFAULT_IMGSZ
    batch: int = _DEFAULT_BATCH
    epochs: int = _DEFAULT_EPOCHS
    workers: int = _DEFAULT_WORKERS
    cache: bool = True
    optimizer: str = "auto"
    seed: int = _DEFAULT_SEED
    patience: int = _DEFAULT_PATIENCE
    project: Path = _DEFAULT_PROJECT
    name: str = _DEFAULT_NAME
    resume_mode: ResumeMode = "ask"
    log_level: str = "INFO"


@dataclass(frozen=True, slots=True)
class ValidationMetrics:
    """Validation metrics extracted from Ultralytics results."""

    precision: float | None
    recall: float | None
    map50: float | None
    map5095: float | None
    f1: float | None


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    """All information needed for the final Mission 2 training summary."""

    config: TrainingConfig
    dataset: DatasetStats
    device: DeviceSelection
    resume: ResumeDecision
    actual_workers: int | None
    train_seconds: float
    validation: ValidationMetrics
    best_epoch: int | None
    best_weights: Path | None


def _default_yolo_factory(weights: str) -> YoloModel:
    """Load a YOLO model from Ultralytics, with a clear missing-package error."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on local install
        raise TrainingError(
            "Ultralytics is required for training. Install the package, then "
            "rerun train.py."
        ) from exc
    return YOLO(weights)


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    """Read a YAML file and require its top-level value to be a mapping."""
    try:
        import yaml
    except ImportError:
        raw = _load_simple_yaml_mapping(path)
        if not isinstance(raw, Mapping):
            raise DatasetError(
                "data.yaml must contain a YAML mapping at the top level."
            )
        return raw

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DatasetError(f"Cannot read data.yaml at {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise DatasetError(f"data.yaml is not valid YAML: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise DatasetError("data.yaml must contain a YAML mapping at the top level.")
    return raw


def _strip_yaml_comment(line: str) -> str:
    """Remove a YAML comment while respecting simple quoted strings."""
    quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "#":
            return line[:index]
    return line


def _parse_simple_yaml_scalar(value: str) -> Any:
    """Parse a small scalar subset sufficient for common YOLO data.yaml files."""
    text = value.strip()
    if text == "":
        return ""
    if text in {"null", "Null", "NULL", "~"}:
        return None
    if text in {"true", "True", "TRUE"}:
        return True
    if text in {"false", "False", "FALSE"}:
        return False
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_simple_yaml_scalar(item) for item in inner.split(",")]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _load_simple_yaml_mapping(path: Path) -> Mapping[str, Any]:
    """Fallback parser for simple YOLO data.yaml when PyYAML is unavailable."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetError(f"Cannot read data.yaml at {path}: {exc}") from exc

    data: dict[str, Any] = {}
    index = 0
    while index < len(lines):
        raw_line = _strip_yaml_comment(lines[index]).rstrip()
        index += 1
        if not raw_line.strip():
            continue
        if raw_line[0].isspace():
            raise DatasetError(
                "Unsupported data.yaml indentation: top-level keys must not "
                f"be indented near line {index}."
            )
        if ":" not in raw_line:
            raise DatasetError(f"Unsupported data.yaml line {index}: {raw_line!r}")

        key, raw_value = raw_line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if not key:
            raise DatasetError(f"data.yaml has an empty key near line {index}.")
        if value:
            data[key] = _parse_simple_yaml_scalar(value)
            continue

        block_list: list[Any] = []
        block_dict: dict[Any, Any] = {}
        mode: Literal["list", "dict"] | None = None
        while index < len(lines):
            block_line = _strip_yaml_comment(lines[index]).rstrip()
            if not block_line.strip():
                index += 1
                continue
            if not block_line[0].isspace():
                break
            stripped = block_line.strip()
            index += 1
            if stripped.startswith("- "):
                if mode == "dict":
                    raise DatasetError(f"Mixed YAML block types under {key!r}.")
                mode = "list"
                block_list.append(_parse_simple_yaml_scalar(stripped[2:].strip()))
            elif ":" in stripped:
                if mode == "list":
                    raise DatasetError(f"Mixed YAML block types under {key!r}.")
                mode = "dict"
                item_key, item_value = stripped.split(":", 1)
                block_dict[_parse_simple_yaml_scalar(item_key.strip())] = (
                    _parse_simple_yaml_scalar(item_value.strip())
                )
            else:
                raise DatasetError(
                    f"Unsupported data.yaml block line {index}: {stripped!r}"
                )

        if mode == "list":
            data[key] = block_list
        elif mode == "dict":
            data[key] = block_dict
        else:
            data[key] = None

    return data


def _resolve_dataset_root(data_yaml: Path, raw: Mapping[str, Any]) -> Path:
    """Resolve the YOLO dataset root from the optional ``path`` key."""
    base = data_yaml.parent
    raw_root = raw.get("path", base)
    if not isinstance(raw_root, (str, Path)):
        raise DatasetError("data.yaml key 'path' must be a string path when present.")
    root = Path(raw_root).expanduser()
    if not root.is_absolute():
        root = base / root
    return root.resolve()


def _normalize_class_names(raw: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate and normalize YOLO class names from list or id:name mapping."""
    if "names" not in raw:
        raise DatasetError("data.yaml must define class names under key 'names'.")

    names_value = raw["names"]
    if isinstance(names_value, Mapping):
        parsed: dict[int, str] = {}
        for key, value in names_value.items():
            try:
                class_id = int(key)
            except (TypeError, ValueError):
                raise DatasetError(f"Class name key {key!r} is not an integer id.") from None
            if class_id < 0:
                raise DatasetError(f"Class name id {class_id} must be >= 0.")
            if not isinstance(value, str) or not value.strip():
                raise DatasetError(f"Class name for id {class_id} must be a non-empty string.")
            parsed[class_id] = value.strip()

        expected = set(range(len(parsed)))
        actual = set(parsed)
        if actual != expected:
            raise DatasetError(
                "Class name ids must be contiguous from 0 to "
                f"{len(parsed) - 1}; got {sorted(actual)}."
            )
        names = tuple(parsed[i] for i in range(len(parsed)))
    elif isinstance(names_value, Sequence) and not isinstance(names_value, (str, bytes)):
        names_list: list[str] = []
        for index, value in enumerate(names_value):
            if not isinstance(value, str) or not value.strip():
                raise DatasetError(
                    f"Class name at index {index} must be a non-empty string."
                )
            names_list.append(value.strip())
        names = tuple(names_list)
    else:
        raise DatasetError("data.yaml key 'names' must be a list or id:name mapping.")

    if not names:
        raise DatasetError("data.yaml must contain at least one class name.")
    if len(set(names)) != len(names):
        raise DatasetError("Class names in data.yaml must be unique.")

    nc = raw.get("nc")
    if nc is not None:
        if isinstance(nc, bool) or not isinstance(nc, int):
            raise DatasetError("data.yaml key 'nc' must be an integer when present.")
        if nc != len(names):
            raise DatasetError(
                f"data.yaml declares nc={nc}, but names contains {len(names)} entries."
            )
    return names


def _split_entries(
    raw: Mapping[str, Any],
    split_name: str,
    root: Path,
    *,
    required: bool,
) -> tuple[Path, ...]:
    """Resolve one split's path or path list from ``data.yaml``."""
    if split_name not in raw:
        if required:
            raise DatasetError(f"data.yaml is missing required key {split_name!r}.")
        return ()

    value = raw[split_name]
    if value in (None, ""):
        if required:
            raise DatasetError(f"data.yaml key {split_name!r} must not be empty.")
        return ()

    raw_entries: Sequence[Any]
    if isinstance(value, (str, Path)):
        raw_entries = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_entries = value
    else:
        raise DatasetError(
            f"data.yaml key {split_name!r} must be a string path or list of paths."
        )

    entries: list[Path] = []
    for item in raw_entries:
        if not isinstance(item, (str, Path)):
            raise DatasetError(f"{split_name} path entries must be strings; got {item!r}.")
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = root / path
        entries.append(path.resolve())
    return tuple(entries)


def _manifest_image_path(line: str, manifest: Path, root: Path) -> Path:
    """Resolve an image path from a YOLO text manifest."""
    image_path = Path(line).expanduser()
    if image_path.is_absolute():
        return image_path.resolve()

    candidates = ((manifest.parent / image_path), (root / image_path))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _collect_images(entry: Path, root: Path) -> tuple[Path, ...]:
    """Collect image files from a split directory, image file, or manifest."""
    if not entry.exists():
        raise DatasetError(f"Dataset split path does not exist: {entry}")

    if entry.is_dir():
        images = [
            path.resolve()
            for path in entry.rglob("*")
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        ]
        return tuple(sorted(images, key=lambda path: path.as_posix()))

    if entry.is_file() and entry.suffix.lower() in _IMAGE_SUFFIXES:
        return (entry.resolve(),)

    if entry.is_file() and entry.suffix.lower() == ".txt":
        images: list[Path] = []
        for line_number, raw_line in enumerate(
            entry.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            image = _manifest_image_path(line, entry, root)
            if image.suffix.lower() not in _IMAGE_SUFFIXES:
                raise DatasetError(
                    f"Manifest {entry}:{line_number} does not point to a known "
                    f"image type: {line!r}"
                )
            images.append(image)
        return tuple(images)

    raise DatasetError(
        f"Dataset split path must be an image directory, image file, or .txt "
        f"manifest: {entry}"
    )


def _label_candidates_for_image(image: Path) -> tuple[Path, ...]:
    """Return likely YOLO label paths for an image, most likely first."""
    candidates: list[Path] = []
    parts = image.parts
    image_indexes = [index for index, part in enumerate(parts) if part == "images"]
    for index in reversed(image_indexes):
        replaced = parts[:index] + ("labels",) + parts[index + 1 :]
        candidates.append(Path(*replaced).with_suffix(".txt"))

    candidates.append(image.with_suffix(".txt"))
    candidates.append(image.parent / "labels" / f"{image.stem}.txt")
    if image.parent.parent != image.parent:
        candidates.append(image.parent.parent / "labels" / f"{image.stem}.txt")
        candidates.append(
            image.parent.parent / "labels" / image.parent.name / f"{image.stem}.txt"
        )

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return tuple(unique)


def label_path_for_image(image: Path) -> Path:
    """Return the existing YOLO label path for ``image``, or the best candidate."""
    candidates = _label_candidates_for_image(image)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _parse_yolo_label_line(
    line: str,
    *,
    label_path: Path,
    line_number: int,
    class_count: int,
) -> int:
    """Validate one YOLO detection label row and return its class id."""
    fields = line.split()
    if len(fields) != 5:
        raise DatasetError(
            f"{label_path}:{line_number} must contain exactly 5 YOLO fields "
            "(class x_center y_center width height)."
        )

    class_token = fields[0]
    try:
        class_id = int(class_token)
    except ValueError:
        raise DatasetError(
            f"{label_path}:{line_number} has non-integer class id {class_token!r}."
        ) from None

    if class_id < 0 or class_id >= class_count:
        raise DatasetError(
            f"{label_path}:{line_number} has invalid class id {class_id}; "
            f"expected 0..{class_count - 1}."
        )

    for name, token in zip(("x_center", "y_center", "width", "height"), fields[1:]):
        try:
            value = float(token)
        except ValueError:
            raise DatasetError(
                f"{label_path}:{line_number} has non-numeric {name}: {token!r}."
            ) from None
        if not math.isfinite(value):
            raise DatasetError(
                f"{label_path}:{line_number} has non-finite {name}: {token!r}."
            )
        if name in {"x_center", "y_center"}:
            if value < 0.0 or value > 1.0:
                raise DatasetError(
                    f"{label_path}:{line_number} has {name}={value}; expected [0, 1]."
                )
        elif value <= 0.0 or value > 1.0:
            raise DatasetError(
                f"{label_path}:{line_number} has {name}={value}; expected (0, 1]."
            )

    return class_id


def _validate_label_file(label_path: Path, class_count: int) -> tuple[int, tuple[int, ...]]:
    """Validate one YOLO label file and return box count plus class counts.

    A missing label file is valid Ultralytics background-image shorthand: it
    is treated the same as a label file that exists but contains zero rows.
    """
    if not label_path.exists():
        return 0, tuple([0] * class_count)
    if not label_path.is_file():
        raise DatasetError(f"Label path is not a file: {label_path}")

    class_counts = [0] * class_count
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise DatasetError(f"Label file is not valid UTF-8: {label_path}") from exc
    except OSError as exc:
        raise DatasetError(f"Cannot read label file {label_path}: {exc}") from exc

    box_count = 0
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        class_id = _parse_yolo_label_line(
            line,
            label_path=label_path,
            line_number=line_number,
            class_count=class_count,
        )
        class_counts[class_id] += 1
        box_count += 1
    return box_count, tuple(class_counts)


def _verify_split(
    split_name: str,
    entries: tuple[Path, ...],
    root: Path,
    class_count: int,
) -> SplitStats:
    """Verify all images and labels for one dataset split."""
    images: list[Path] = []
    for entry in entries:
        images.extend(_collect_images(entry, root))

    if not images:
        raise DatasetError(f"Dataset split {split_name!r} contains no images.")

    seen_images: set[Path] = set()
    duplicates: set[Path] = set()
    for image in images:
        if image in seen_images:
            duplicates.add(image)
        seen_images.add(image)
        if not image.exists():
            raise DatasetError(f"Image listed in {split_name!r} does not exist: {image}")
        if not image.is_file():
            raise DatasetError(f"Image path in {split_name!r} is not a file: {image}")
    if duplicates:
        sample = sorted(path.as_posix() for path in duplicates)[:3]
        raise DatasetError(f"Duplicate images in split {split_name!r}: {sample}")

    total_class_counts = [0] * class_count
    total_boxes = 0
    empty_labels = 0
    missing_labels = 0
    label_paths: set[Path] = set()

    for image in images:
        label_path = label_path_for_image(image)
        box_count, class_counts = _validate_label_file(label_path, class_count)
        label_paths.add(label_path)
        total_boxes += box_count
        if box_count == 0:
            if label_path.exists():
                empty_labels += 1
            else:
                missing_labels += 1
        for index, count in enumerate(class_counts):
            total_class_counts[index] += count

    return SplitStats(
        name=split_name,
        source_paths=entries,
        image_count=len(images),
        label_count=len(label_paths),
        box_count=total_boxes,
        empty_label_count=empty_labels,
        missing_label_count=missing_labels,
        class_counts=tuple(total_class_counts),
    )


def verify_dataset(data_yaml: str | Path) -> DatasetStats:
    """Fail-loud verification of ``data.yaml`` and all referenced YOLO labels.

    The verifier checks that train and validation paths exist and images are
    present. An image without a label file is treated as a valid Ultralytics
    background image with zero objects, same as an image with an empty label
    file. When a label file does exist, its rows must be valid YOLO detection
    rows, class ids must be in range, class names must be unique, and ``nc``
    must match ``names`` when declared.
    """
    data_yaml_path = Path(data_yaml).expanduser().resolve()
    if not data_yaml_path.exists():
        raise DatasetError(f"data.yaml does not exist: {data_yaml_path}")
    if not data_yaml_path.is_file():
        raise DatasetError(f"data.yaml path is not a file: {data_yaml_path}")

    raw = _load_yaml_mapping(data_yaml_path)
    root = _resolve_dataset_root(data_yaml_path, raw)
    class_names = _normalize_class_names(raw)
    train_entries = _split_entries(raw, "train", root, required=True)
    val_entries = _split_entries(raw, "val", root, required=True)
    test_entries = _split_entries(raw, "test", root, required=False)

    train = _verify_split("train", train_entries, root, len(class_names))
    val = _verify_split("val", val_entries, root, len(class_names))
    test = (
        _verify_split("test", test_entries, root, len(class_names))
        if test_entries
        else None
    )

    return DatasetStats(
        data_yaml=data_yaml_path,
        root=root,
        class_names=class_names,
        train=train,
        val=val,
        test=test,
    )


def _torch_available_device(torch_module: object, name: str) -> bool:
    """Return whether ``torch.<name>.is_available()``-style API is true."""
    owner = torch_module
    for attr in name.split("."):
        owner = getattr(owner, attr, None)
        if owner is None:
            return False
    available = getattr(owner, "is_available", None)
    return bool(available()) if callable(available) else False


def select_device(
    *,
    torch_module: object | None = None,
    system_name: str | None = None,
    machine_name: str | None = None,
) -> DeviceSelection:
    """Automatically select Apple MPS, NVIDIA CUDA, or CPU for training."""
    if torch_module is None:
        try:
            import torch as imported_torch
        except ImportError:  # pragma: no cover - depends on local install
            imported_torch = None
        torch_module = imported_torch

    system = system_name or platform.system()
    machine = (machine_name or platform.machine()).lower()
    is_apple_silicon = system == "Darwin" and machine in {"arm64", "aarch64"}

    if torch_module is not None:
        if is_apple_silicon and _torch_available_device(torch_module, "backends.mps"):
            return DeviceSelection(value="mps", label="Apple MPS")
        if _torch_available_device(torch_module, "cuda"):
            return DeviceSelection(value="cuda", label="NVIDIA CUDA")

    return DeviceSelection(value="cpu", label="CPU")


def last_checkpoint_path(config: TrainingConfig) -> Path:
    """Return the expected Ultralytics ``last.pt`` path for this run name."""
    return (config.project / config.name / "weights" / "last.pt").expanduser()


def resolve_resume(
    last_checkpoint: str | Path,
    *,
    mode: ResumeMode = "ask",
    input_fn: Callable[[str], str] = input,
) -> ResumeDecision:
    """Resolve whether to resume when ``last.pt`` exists."""
    checkpoint = Path(last_checkpoint).expanduser()
    if not checkpoint.exists():
        return ResumeDecision(resume=False, checkpoint=None)
    if not checkpoint.is_file():
        raise TrainingError(f"Resume checkpoint exists but is not a file: {checkpoint}")

    if mode == "yes":
        return ResumeDecision(resume=True, checkpoint=checkpoint)
    if mode == "no":
        return ResumeDecision(resume=False, checkpoint=None)
    if mode != "ask":
        raise TrainingError(f"Unknown resume mode {mode!r}; expected ask, yes, or no.")

    prompt = f"Found existing checkpoint {checkpoint}. Resume training? [y/N]: "
    try:
        answer = input_fn(prompt).strip().lower()
    except EOFError as exc:
        raise TrainingError(
            "Cannot ask whether to resume in a non-interactive session. "
            "Pass --resume yes or --resume no."
        ) from exc

    if answer in {"y", "yes"}:
        return ResumeDecision(resume=True, checkpoint=checkpoint)
    if answer in {"", "n", "no"}:
        return ResumeDecision(resume=False, checkpoint=None)
    raise TrainingError("Resume answer must be yes or no.")


def _object_value(source: object, key: str) -> Any:
    """Read ``key`` from either a mapping or an object attribute."""
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _nested_value(source: object, path: Sequence[str]) -> Any:
    """Read a nested mapping/attribute path, returning None on any miss."""
    current: object | None = source
    for key in path:
        if current is None:
            return None
        current = _object_value(current, key)
    return current


def _as_optional_float(value: object) -> float | None:
    """Convert scalars, numpy scalars, or numeric sequences to a float."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (str, bytes)):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        mean = getattr(value, "mean", None)
        if callable(mean):
            try:
                # value is numpy-like here; callable() narrows `mean` to
                # Callable[..., object], losing numpy's real scalar return
                # type (see diagnostics_report.md C-10).
                parsed = float(cast(Any, mean)())
            except (TypeError, ValueError):
                return None
            return parsed if math.isfinite(parsed) else None
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            numbers = [_as_optional_float(item) for item in value]
            clean = [number for number in numbers if number is not None]
            if not clean:
                return None
            parsed = sum(clean) / len(clean)
            return parsed if math.isfinite(parsed) else None
        return None
    return parsed if math.isfinite(parsed) else None


def _as_optional_int(value: object) -> int | None:
    """Convert a numeric value to int when possible."""
    number = _as_optional_float(value)
    if number is None:
        return None
    return int(number)


def _metric_from_sources(sources: Sequence[object], keys: Sequence[str]) -> float | None:
    """Return the first metric value found in any source under any key."""
    for source in sources:
        for key in keys:
            value = _object_value(source, key)
            number = _as_optional_float(value)
            if number is not None:
                return number
    return None


def extract_validation_metrics(results: object) -> ValidationMetrics:
    """Extract precision, recall, mAP, and F1 from Ultralytics validation output."""
    box = _object_value(results, "box")
    results_dict = _object_value(results, "results_dict")
    box_results_dict = _object_value(box, "results_dict") if box is not None else None
    sources = tuple(source for source in (box, results, box_results_dict, results_dict) if source)

    precision = _metric_from_sources(
        sources,
        ("mp", "precision", "metrics/precision(B)", "metrics/precision"),
    )
    recall = _metric_from_sources(
        sources,
        ("mr", "recall", "metrics/recall(B)", "metrics/recall"),
    )
    map50 = _metric_from_sources(
        sources,
        ("map50", "mAP50", "metrics/mAP50(B)", "metrics/mAP50"),
    )
    map5095 = _metric_from_sources(
        sources,
        ("map", "map5095", "mAP50-95", "metrics/mAP50-95(B)", "metrics/mAP50-95"),
    )
    f1 = _metric_from_sources(sources, ("f1", "F1", "metrics/F1(B)", "metrics/F1"))
    if f1 is None and precision is not None and recall is not None and precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)

    return ValidationMetrics(
        precision=precision,
        recall=recall,
        map50=map50,
        map5095=map5095,
        f1=f1,
    )


def _extract_save_dir(train_results: object, model: YoloModel) -> Path | None:
    """Find Ultralytics save_dir from results or model.trainer."""
    trainer = getattr(model, "trainer", None)
    for source in (train_results, trainer):
        value = _object_value(source, "save_dir")
        if value is not None:
            return Path(value)
    return None


def extract_actual_workers(
    train_results: object,
    model: YoloModel,
    requested_workers: int,
) -> int | None:
    """Best-effort detection of the worker count Ultralytics actually used."""
    trainer = getattr(model, "trainer", None)
    candidates = (
        _nested_value(train_results, ("args", "workers")),
        _nested_value(trainer, ("args", "workers")),
        _nested_value(train_results, ("trainer", "args", "workers")),
        _nested_value(model, ("overrides", "workers")),
    )
    for candidate in candidates:
        workers = _as_optional_int(candidate)
        if workers is not None:
            return workers
    return requested_workers


def extract_best_weights(
    train_results: object,
    model: YoloModel,
    *,
    config: TrainingConfig,
) -> Path | None:
    """Find the best model checkpoint path from Ultralytics state."""
    trainer = getattr(model, "trainer", None)
    candidates = (
        _object_value(train_results, "best"),
        _object_value(trainer, "best"),
        _nested_value(train_results, ("trainer", "best")),
    )
    for candidate in candidates:
        if candidate:
            return Path(candidate)

    save_dir = _extract_save_dir(train_results, model)
    if save_dir is not None:
        return save_dir / "weights" / "best.pt"
    return config.project / config.name / "weights" / "best.pt"


def _best_epoch_from_csv(save_dir: Path) -> int | None:
    """Parse Ultralytics results.csv and return the epoch with best mAP50-95."""
    csv_path = save_dir / "results.csv"
    if not csv_path.exists():
        return None

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return None

    best_epoch: int | None = None
    best_score = -math.inf
    metric_keys = ("metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP50(B)")

    for raw_row in rows:
        row = {str(key).strip(): str(value).strip() for key, value in raw_row.items()}
        score: float | None = None
        for key in metric_keys:
            if key in row and row[key]:
                try:
                    score = float(row[key])
                except ValueError:
                    score = None
                break
        if score is None or not math.isfinite(score):
            continue
        if score > best_score:
            best_score = score
            best_epoch = _as_optional_int(row.get("epoch"))
    return best_epoch


def extract_best_epoch(train_results: object, model: YoloModel) -> int | None:
    """Extract best epoch from Ultralytics state or results.csv."""
    trainer = getattr(model, "trainer", None)
    candidates = (
        _object_value(train_results, "best_epoch"),
        _object_value(trainer, "best_epoch"),
        _nested_value(train_results, ("trainer", "best_epoch")),
    )
    for candidate in candidates:
        epoch = _as_optional_int(candidate)
        if epoch is not None:
            return epoch

    save_dir = _extract_save_dir(train_results, model)
    if save_dir is not None:
        return _best_epoch_from_csv(save_dir)
    return None


def _format_float(value: float | None, digits: int = 4) -> str:
    """Format optional metric values for summary output."""
    return "n/a" if value is None else f"{value:.{digits}f}"


def _format_duration(seconds: float) -> str:
    """Format elapsed seconds as compact hours/minutes/seconds."""
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def compute_dataset_quality(stats: DatasetStats) -> DatasetQualityStats:
    """Compute non-fatal training-split quality statistics."""
    counts = stats.train.class_counts
    min_count = min(counts)
    max_count = max(counts)
    median_count = float(statistics.median(counts))
    mean_count = float(statistics.mean(counts))

    if min_count == 0:
        imbalance = math.inf if max_count > 0 else 1.0
    else:
        imbalance = max_count / min_count

    low_warning_ids: tuple[int, ...] = ()
    if median_count > 0:
        threshold = median_count * _LOW_CLASS_COUNT_RATIO
        low_warning_ids = tuple(
            index for index, count in enumerate(counts) if count < threshold
        )

    total_images = stats.train.image_count
    negative_images_missing_labels = stats.train.missing_label_count
    negative_images_empty_labels = stats.train.empty_label_count
    negative_images = negative_images_missing_labels + negative_images_empty_labels
    positive_images = total_images - negative_images
    negative_ratio = negative_images / total_images if total_images else 0.0

    return DatasetQualityStats(
        negative_images=negative_images,
        negative_images_missing_labels=negative_images_missing_labels,
        negative_images_empty_labels=negative_images_empty_labels,
        positive_images=positive_images,
        negative_ratio=negative_ratio,
        per_class_instance_counts=counts,
        min_class_count=min_count,
        max_class_count=max_count,
        median_class_count=median_count,
        mean_class_count=mean_count,
        largest_smallest_imbalance=imbalance,
        low_class_warning_ids=low_warning_ids,
    )


def _format_percent(value: float) -> str:
    """Format a ratio as a percentage for human-readable reports."""
    return f"{value * 100.0:.1f}%"


def _format_imbalance(value: float | None) -> str:
    """Format largest/smallest class imbalance."""
    if value is None:
        return "n/a"
    if math.isinf(value):
        return "infinite"
    return f"{value:.2f}x"


def render_dataset_statistics(stats: DatasetStats) -> str:
    """Render a concise dataset verification report."""
    lines = [
        "==================================================",
        "Mission 2 Dataset Verification",
        "==================================================",
        f"data.yaml : {stats.data_yaml}",
        f"Root      : {stats.root}",
        f"Classes   : {stats.class_count}",
        f"Train     : {stats.train.image_count} images, "
        f"{stats.train.box_count} boxes, {stats.train.negative_count} negatives",
        f"Val       : {stats.val.image_count} images, "
        f"{stats.val.box_count} boxes, {stats.val.negative_count} negatives",
    ]
    if stats.test is not None:
        lines.append(
            f"Test      : {stats.test.image_count} images, "
            f"{stats.test.box_count} boxes, {stats.test.negative_count} negatives"
        )
    lines.append("==================================================")
    return "\n".join(lines)


def render_dataset_quality(stats: DatasetStats) -> str:
    """Render dataset quality statistics and non-fatal quality warnings."""
    quality = compute_dataset_quality(stats)
    lines = [
        "--------------------------------------------------",
        "Dataset Quality",
        "--------------------------------------------------",
        "Training split",
        f"Positive Images : {quality.positive_images}",
        f"Negative Images (missing labels) : {quality.negative_images_missing_labels}",
        f"Negative Images (empty labels) : {quality.negative_images_empty_labels}",
        f"Total Negatives : {quality.negative_images}",
        f"Negative Ratio : {_format_percent(quality.negative_ratio)}",
        f"Minimum Class Count : {quality.min_class_count}",
        f"Maximum Class Count : {quality.max_class_count}",
        f"Median Class Count : {quality.median_class_count:.2f}",
        f"Mean Class Count : {quality.mean_class_count:.2f}",
        "Largest/Smallest Class Imbalance : "
        f"{_format_imbalance(quality.largest_smallest_imbalance)}",
        "Per-class instance counts:",
    ]

    for class_id, (class_name, count) in enumerate(
        zip(stats.class_names, quality.per_class_instance_counts)
    ):
        lines.append(f"  {class_id:03d} {class_name} : {count}")

    if not (_NEGATIVE_RATIO_MIN <= quality.negative_ratio <= _NEGATIVE_RATIO_MAX):
        lines.extend(
            [
                "WARNING",
                "Negative image ratio is outside the recommended 10% to 20% range.",
            ]
        )

    if quality.low_class_warning_ids:
        low_classes = ", ".join(
            f"{class_id}:{stats.class_names[class_id]}"
            for class_id in quality.low_class_warning_ids
        )
        lines.extend(
            [
                "WARNING",
                "Some classes contain dramatically fewer instances than the "
                f"median: {low_classes}",
            ]
        )

    lines.append("--------------------------------------------------")
    return "\n".join(lines)


def render_summary(outcome: TrainingOutcome) -> str:
    """Render the final Mission 2 training summary."""
    workers = (
        str(outcome.actual_workers)
        if outcome.actual_workers is not None
        else "unknown"
    )
    best_model = str(outcome.best_weights) if outcome.best_weights is not None else "unknown"
    best_epoch = str(outcome.best_epoch) if outcome.best_epoch is not None else "unknown"

    lines = [
        "==================================================",
        "Mission 2 Training Summary",
        "==================================================",
        "Dataset",
        f"[OK] Classes : {outcome.dataset.class_count}",
        f"[OK] Train Images : {outcome.dataset.train.image_count}",
        f"[OK] Validation Images : {outcome.dataset.val.image_count}",
        f"[OK] Device : {outcome.device.label}",
        f"[OK] Batch : {outcome.config.batch}",
        f"[OK] Workers Requested : {outcome.config.workers}",
        f"[OK] Actual Workers Used : {workers}",
        f"[OK] Cache : {'Enabled' if outcome.config.cache else 'Disabled'}",
        "--------------------------------------------",
        f"Best Epoch : {best_epoch}",
        f"Best mAP50 : {_format_float(outcome.validation.map50)}",
        f"Best mAP50-95 : {_format_float(outcome.validation.map5095)}",
        f"Precision : {_format_float(outcome.validation.precision)}",
        f"Recall : {_format_float(outcome.validation.recall)}",
        f"F1 : {_format_float(outcome.validation.f1)}",
        f"Training Time : {_format_duration(outcome.train_seconds)}",
        f"Best Model : {best_model}",
        "Ready for export.py",
        "==================================================",
    ]
    return "\n".join(lines)


def _train_kwargs(
    config: TrainingConfig,
    device: DeviceSelection,
    resume: ResumeDecision,
) -> dict[str, Any]:
    """Build the exact Ultralytics training kwargs for this run."""
    return {
        "data": str(config.data_yaml),
        "imgsz": config.imgsz,
        "batch": config.batch,
        "epochs": config.epochs,
        "workers": config.workers,
        "cache": config.cache,
        "optimizer": config.optimizer,
        "seed": config.seed,
        "patience": config.patience,
        "project": str(config.project),
        "name": config.name,
        "device": device.value,
        "resume": resume.resume,
    }


def _val_kwargs(config: TrainingConfig, device: DeviceSelection) -> dict[str, Any]:
    """Build the exact Ultralytics validation kwargs for this run."""
    return {
        "data": str(config.data_yaml),
        "imgsz": config.imgsz,
        "batch": config.batch,
        "device": device.value,
        "split": "val",
    }


def copy_best_model(
    best_weights: str | Path | None,
    destination: str | Path = _STABLE_BEST_MODEL,
) -> Path:
    """Copy the final best checkpoint to the stable export handoff path."""
    if best_weights is None:
        raise TrainingError("Cannot copy best model because no best.pt path was found.")

    source = Path(best_weights).expanduser()
    destination_path = Path(destination).expanduser()
    try:
        resolved_source = source.resolve(strict=True)
    except OSError as exc:
        raise TrainingError(f"Best model checkpoint does not exist: {source}") from exc

    try:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.exists() and resolved_source.samefile(destination_path):
            return destination_path.resolve()
        shutil.copy2(resolved_source, destination_path)
    except (OSError, shutil.Error) as exc:
        raise TrainingError(
            f"Failed to copy best model from {resolved_source} to "
            f"{destination_path}: {exc}"
        ) from exc
    return destination_path.resolve()


def render_worker_mismatch_warning(
    requested_workers: int,
    actual_workers: int | None,
) -> str | None:
    """Render a non-fatal warning when Ultralytics changes worker count."""
    if actual_workers is None or actual_workers == requested_workers:
        return None
    return "\n".join(
        [
            "--------------------------------------------------",
            "WARNING",
            f"Requested workers : {requested_workers}",
            f"Actual workers : {actual_workers}",
            "Ultralytics reduced the worker count.",
            "This commonly happens on macOS/MPS.",
            "--------------------------------------------------",
        ]
    )


def _weights_used(outcome: TrainingOutcome) -> str:
    """Return the weights path actually used to construct the YOLO model."""
    if outcome.resume.resume and outcome.resume.checkpoint is not None:
        return str(outcome.resume.checkpoint)
    return outcome.config.weights


def _model_name(weights: str) -> str:
    """Infer a compact model name from a weights path or identifier."""
    stem = Path(weights).stem
    return stem if stem else weights


def build_training_summary_payload(outcome: TrainingOutcome) -> dict[str, Any]:
    """Build the machine-readable training summary JSON payload."""
    weights = _weights_used(outcome)
    return {
        "model": _model_name(outcome.config.weights),
        "weights": weights,
        "dataset_path": str(outcome.dataset.data_yaml),
        "number_of_classes": outcome.dataset.class_count,
        "train_images": outcome.dataset.train.image_count,
        "validation_images": outcome.dataset.val.image_count,
        "device": outcome.device.label,
        "batch": outcome.config.batch,
        "requested_workers": outcome.config.workers,
        "actual_workers": outcome.actual_workers,
        "epochs": outcome.config.epochs,
        "training_duration_seconds": outcome.train_seconds,
        "best_epoch": outcome.best_epoch,
        "precision": outcome.validation.precision,
        "recall": outcome.validation.recall,
        "F1": outcome.validation.f1,
        "mAP50": outcome.validation.map50,
        "mAP50-95": outcome.validation.map5095,
        "best_model_path": str(outcome.best_weights) if outcome.best_weights else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def write_training_summary_json(
    outcome: TrainingOutcome,
    destination: str | Path = _TRAIN_SUMMARY_JSON,
) -> Path:
    """Write the machine-readable training summary with an atomic replace."""
    destination_path = Path(destination).expanduser()
    temp_path = destination_path.with_name(
        f".{destination_path.name}.{os.getpid()}.tmp"
    )
    payload = build_training_summary_payload(outcome)

    try:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination_path)
    except OSError as exc:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise TrainingError(
            f"Failed to write training summary JSON to {destination_path}: {exc}"
        ) from exc
    return destination_path.resolve()


def run_training_pipeline(
    config: TrainingConfig,
    *,
    model_factory: YoloFactory | None = None,
    torch_module: object | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    clock: Callable[[], float] = time.perf_counter,
) -> TrainingOutcome:
    """Verify data, train YOLO11n with Ultralytics, validate, and summarize."""
    factory = model_factory or _default_yolo_factory

    log.info("Verifying dataset before training: %s", config.data_yaml)
    dataset = verify_dataset(config.data_yaml)
    output_fn(render_dataset_statistics(dataset))
    output_fn(render_dataset_quality(dataset))

    device = select_device(torch_module=torch_module)
    output_fn(f"Selected device: {device.label} ({device.value})")

    resume = resolve_resume(
        last_checkpoint_path(config),
        mode=config.resume_mode,
        input_fn=input_fn,
    )
    if resume.resume and resume.checkpoint is not None:
        weights = str(resume.checkpoint)
        output_fn(f"Resuming from checkpoint: {resume.checkpoint}")
    else:
        weights = config.weights
        output_fn(f"Starting fresh from pretrained weights: {weights}")

    model = factory(weights)
    train_args = _train_kwargs(config, device, resume)

    output_fn("Starting Ultralytics training. Native progress output follows.")
    output_fn(f"Cache {'Enabled' if config.cache else 'Disabled'}")
    start = clock()
    train_results = model.train(**train_args)
    train_seconds = clock() - start

    output_fn("Training finished. Running validation on the validation split.")
    val_results = model.val(**_val_kwargs(config, device))

    actual_workers = extract_actual_workers(train_results, model, config.workers)
    run_best_weights = extract_best_weights(train_results, model, config=config)
    best_weights = copy_best_model(run_best_weights)
    output_fn(f"Copied best model -> {_STABLE_BEST_MODEL}")
    best_epoch = extract_best_epoch(train_results, model)
    metrics = extract_validation_metrics(val_results)

    outcome = TrainingOutcome(
        config=config,
        dataset=dataset,
        device=device,
        resume=resume,
        actual_workers=actual_workers,
        train_seconds=train_seconds,
        validation=metrics,
        best_epoch=best_epoch,
        best_weights=best_weights,
    )
    worker_warning = render_worker_mismatch_warning(config.workers, actual_workers)
    if worker_warning is not None:
        output_fn(worker_warning)
    output_fn(render_summary(outcome))
    write_training_summary_json(outcome)
    return outcome


def _positive_int(value: str) -> int:
    """argparse converter for positive integers."""
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer.") from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{value!r} must be > 0.")
    return parsed


def _non_negative_int(value: str) -> int:
    """argparse converter for non-negative integers."""
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer.") from None
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"{value!r} must be >= 0.")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> TrainingConfig:
    """Parse CLI arguments into a typed ``TrainingConfig``."""
    parser = argparse.ArgumentParser(
        description="Train YOLO11n for ICMTC 2026 Mission 2 flags."
    )
    parser.add_argument("--data", default=str(_DEFAULT_DATA_YAML), help="Path to data.yaml.")
    parser.add_argument("--weights", default=_DEFAULT_WEIGHTS, help="Pretrained weights.")
    parser.add_argument("--epochs", type=_positive_int, default=_DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=_positive_int, default=_DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=_positive_int, default=_DEFAULT_BATCH)
    parser.add_argument("--workers", type=_non_negative_int, default=_DEFAULT_WORKERS)
    parser.add_argument("--seed", type=_non_negative_int, default=_DEFAULT_SEED)
    parser.add_argument("--patience", type=_non_negative_int, default=_DEFAULT_PATIENCE)
    parser.add_argument("--project", default=str(_DEFAULT_PROJECT))
    parser.add_argument("--name", default=_DEFAULT_NAME)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--resume", choices=("ask", "yes", "no"), default="ask")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable Ultralytics dataset caching.",
    )

    args = parser.parse_args(argv)
    return TrainingConfig(
        data_yaml=Path(args.data).expanduser(),
        weights=args.weights,
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        workers=args.workers,
        cache=not args.no_cache,
        optimizer=args.optimizer,
        seed=args.seed,
        patience=args.patience,
        project=Path(args.project).expanduser(),
        name=args.name,
        resume_mode=args.resume,
        log_level=args.log_level,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``python train.py``."""
    config = parse_args(argv)
    configure_logging(level=config.log_level)
    try:
        run_training_pipeline(config)
    except (DatasetError, TrainingError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI use
    raise SystemExit(main(sys.argv[1:]))
