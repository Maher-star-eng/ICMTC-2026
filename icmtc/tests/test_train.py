# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   21/06/2026

"""Hardware-free tests for train.py.

Ultralytics and Torch are mocked by dependency injection.  The dataset tests
use tiny synthetic YOLO directory trees; image bytes are not decoded by the
training manager, so simple placeholder files are enough.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from training.train import (
    TrainingError,
    DatasetError,
    ResumeDecision,
    TrainingConfig,
    TrainingOutcome,
    ValidationMetrics,
    compute_dataset_quality,
    copy_best_model,
    last_checkpoint_path,
    render_dataset_quality,
    render_summary,
    render_worker_mismatch_warning,
    resolve_resume,
    run_training_pipeline,
    select_device,
    verify_dataset,
)


def _fake_torch(*, mps: bool = False, cuda: bool = False) -> SimpleNamespace:
    """Return a minimal Torch-like object exposing MPS/CUDA availability."""
    return SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        cuda=SimpleNamespace(is_available=lambda: cuda),
    )


def _write_image(path: Path) -> None:
    """Create a placeholder image file; train.py only checks paths/suffixes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")


def _write_label(path: Path, content: str) -> None:
    """Create a YOLO label file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_dataset(tmp_path: Path) -> Path:
    """Build a small valid YOLO dataset and return its data.yaml path."""
    root = tmp_path / "dataset"

    _write_image(root / "images/train/a.jpg")
    _write_label(root / "labels/train/a.txt", "0 0.5 0.5 0.25 0.25\n")

    _write_image(root / "images/train/negative.jpg")
    _write_label(root / "labels/train/negative.txt", "")

    _write_image(root / "images/val/b.jpg")
    _write_label(root / "labels/val/b.txt", "1 0.4 0.4 0.2 0.2\n")

    _write_image(root / "images/test/c.jpg")
    _write_label(root / "labels/test/c.txt", "2 0.6 0.6 0.1 0.1\n")

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "nc: 3",
                "names:",
                "  - red",
                "  - green",
                "  - blue",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def _make_imbalanced_dataset(tmp_path: Path) -> Path:
    """Build a valid dataset with skewed class counts for quality reporting."""
    root = tmp_path / "imbalanced"
    _write_image(root / "images/train/a.jpg")
    lines: list[str] = []
    for class_id, count in enumerate((8, 8, 8, 1, 8)):
        lines.extend([f"{class_id} 0.5 0.5 0.1 0.1"] * count)
    _write_label(root / "labels/train/a.txt", "\n".join(lines) + "\n")

    _write_image(root / "images/val/b.jpg")
    _write_label(root / "labels/val/b.txt", "0 0.5 0.5 0.1 0.1\n")

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/val",
                "nc: 5",
                "names:",
                "  - c0",
                "  - c1",
                "  - c2",
                "  - c3",
                "  - c4",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_verify_dataset_collects_stats_and_allows_empty_negative_labels(tmp_path: Path):
    """Valid YOLO data.yaml verifies and reports image, box, and class stats."""
    data_yaml = _make_dataset(tmp_path)

    stats = verify_dataset(data_yaml)

    assert stats.class_count == 3
    assert stats.class_names == ("red", "green", "blue")
    assert stats.train.image_count == 2
    assert stats.train.label_count == 2
    assert stats.train.box_count == 1
    assert stats.train.empty_label_count == 1
    assert stats.train.class_counts == (1, 0, 0)
    assert stats.val.image_count == 1
    assert stats.test is not None
    assert stats.test.image_count == 1


def _make_mixed_negative_dataset(tmp_path: Path) -> Path:
    """Build a dataset with a positive image, a missing-label negative, and an
    empty-label negative, to exercise both Ultralytics background-image forms.
    """
    root = tmp_path / "mixed"

    _write_image(root / "images/train/a.jpg")
    _write_label(root / "labels/train/a.txt", "0 0.5 0.5 0.25 0.25\n")

    _write_image(root / "images/train/missing_negative.jpg")
    # Deliberately no labels/train/missing_negative.txt.

    _write_image(root / "images/train/empty_negative.jpg")
    _write_label(root / "labels/train/empty_negative.txt", "")

    _write_image(root / "images/val/b.jpg")
    _write_label(root / "labels/val/b.txt", "0 0.4 0.4 0.2 0.2\n")

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                "names:",
                "  - red",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_verify_dataset_accepts_missing_label_file_as_background_image(tmp_path: Path):
    """An image with no corresponding .txt label is a valid negative image."""
    data_yaml = _make_mixed_negative_dataset(tmp_path)

    stats = verify_dataset(data_yaml)

    assert stats.train.image_count == 3
    assert stats.train.box_count == 1
    assert stats.train.missing_label_count == 1
    assert stats.train.empty_label_count == 1
    assert stats.train.negative_count == 2


def test_verify_dataset_accepts_empty_label_file_as_background_image(tmp_path: Path):
    """An existing but empty label file remains a valid negative image."""
    data_yaml = _make_dataset(tmp_path)

    stats = verify_dataset(data_yaml)

    assert stats.train.empty_label_count == 1
    assert stats.train.missing_label_count == 0
    assert stats.train.negative_count == 1


def test_dataset_quality_reports_missing_and_empty_negatives_separately(tmp_path: Path):
    """Quality stats and report break negatives down by representation."""
    stats = verify_dataset(_make_mixed_negative_dataset(tmp_path))

    quality = compute_dataset_quality(stats)
    report = render_dataset_quality(stats)

    assert quality.positive_images == 1
    assert quality.negative_images_missing_labels == 1
    assert quality.negative_images_empty_labels == 1
    assert quality.negative_images == 2
    assert quality.negative_ratio == pytest.approx(2 / 3)

    assert "Negative Images (missing labels) : 1" in report
    assert "Negative Images (empty labels) : 1" in report
    assert "Total Negatives : 2" in report


def test_verify_dataset_malformed_existing_label_still_raises(tmp_path: Path):
    """An existing label file with malformed rows still fails loudly."""
    data_yaml = _make_mixed_negative_dataset(tmp_path)
    bad_label = data_yaml.parent / "labels/train/a.txt"
    bad_label.write_text("0 0.5 0.5 0.25\n", encoding="utf-8")

    with pytest.raises(DatasetError) as excinfo:
        verify_dataset(data_yaml)

    assert "5 YOLO fields" in str(excinfo.value)


def test_verify_dataset_invalid_class_id_raises(tmp_path: Path):
    """A label class outside names/nc fails before training begins."""
    data_yaml = _make_dataset(tmp_path)
    bad_label = data_yaml.parent / "labels/val/b.txt"
    bad_label.write_text("99 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(DatasetError) as excinfo:
        verify_dataset(data_yaml)

    assert "invalid class id" in str(excinfo.value)


def test_dataset_quality_report_warns_without_failing(tmp_path: Path):
    """Quality reporting surfaces negative-ratio and low-class warnings only."""
    stats = verify_dataset(_make_imbalanced_dataset(tmp_path))

    quality = compute_dataset_quality(stats)
    report = render_dataset_quality(stats)

    assert quality.negative_images == 0
    assert quality.positive_images == 1
    assert quality.negative_ratio == pytest.approx(0.0)
    assert quality.per_class_instance_counts == (8, 8, 8, 1, 8)
    assert quality.min_class_count == 1
    assert quality.max_class_count == 8
    assert quality.median_class_count == pytest.approx(8.0)
    assert quality.mean_class_count == pytest.approx(6.6)
    assert quality.largest_smallest_imbalance == pytest.approx(8.0)
    assert quality.low_class_warning_ids == (3,)

    assert "Dataset Quality" in report
    assert "Negative Ratio : 0.0%" in report
    assert "Largest/Smallest Class Imbalance : 8.00x" in report
    assert "003 c3 : 1" in report
    assert "outside the recommended 10% to 20% range" in report
    assert "dramatically fewer instances than the median" in report


def test_select_device_prefers_apple_mps_then_cuda_then_cpu():
    """Automatic device selection follows the requested Mission 2 priority."""
    mps_device = select_device(
        torch_module=_fake_torch(mps=True, cuda=True),
        system_name="Darwin",
        machine_name="arm64",
    )
    cuda_device = select_device(
        torch_module=_fake_torch(cuda=True),
        system_name="Linux",
        machine_name="x86_64",
    )
    cpu_device = select_device(
        torch_module=_fake_torch(),
        system_name="Linux",
        machine_name="x86_64",
    )

    assert mps_device.value == "mps"
    assert mps_device.label == "Apple MPS"
    assert cuda_device.value == "cuda"
    assert cuda_device.label == "NVIDIA CUDA"
    assert cpu_device.value == "cpu"


def test_resolve_resume_detects_last_checkpoint_and_honors_modes(tmp_path: Path):
    """last.pt detection supports explicit resume, explicit fresh, and ask."""
    checkpoint = tmp_path / "runs/exp/weights/last.pt"

    assert resolve_resume(checkpoint) == ResumeDecision(False, None)

    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")

    assert resolve_resume(checkpoint, mode="yes") == ResumeDecision(True, checkpoint)
    assert resolve_resume(checkpoint, mode="no") == ResumeDecision(False, None)

    asked = resolve_resume(checkpoint, mode="ask", input_fn=lambda _prompt: "yes")
    assert asked == ResumeDecision(True, checkpoint)


def test_render_summary_contains_mission_metrics(tmp_path: Path):
    """Final summary includes dataset, worker, metric, model, and export handoff."""
    stats = verify_dataset(_make_dataset(tmp_path))
    config = TrainingConfig(project=tmp_path / "runs", name="exp")
    outcome = TrainingOutcome(
        config=config,
        dataset=stats,
        device=select_device(
            torch_module=_fake_torch(cuda=True),
            system_name="Linux",
            machine_name="x86_64",
        ),
        resume=ResumeDecision(False, None),
        actual_workers=6,
        train_seconds=125.0,
        validation=ValidationMetrics(
            precision=0.81,
            recall=0.72,
            map50=0.9,
            map5095=0.66,
            f1=0.762352941,
        ),
        best_epoch=11,
        best_weights=tmp_path / "runs/exp/weights/best.pt",
    )

    summary = render_summary(outcome)

    assert "Mission 2 Training Summary" in summary
    assert "[OK] Classes : 3" in summary
    assert "[OK] Actual Workers Used : 6" in summary
    assert "[OK] Cache : Enabled" in summary
    assert "Best Epoch : 11" in summary
    assert "Best mAP50 : 0.9000" in summary
    assert "Best mAP50-95 : 0.6600" in summary
    assert "Precision : 0.8100" in summary
    assert "Recall : 0.7200" in summary
    assert "Ready for export.py" in summary


def test_copy_best_model_preserves_original_and_raises_on_missing(tmp_path: Path):
    """The stable export handoff is copied, not moved, and fails loudly."""
    source = tmp_path / "runs/exp/weights/best.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"model")
    destination = tmp_path / "models/best.pt"

    copied = copy_best_model(source, destination)

    assert copied == destination.resolve()
    assert source.read_bytes() == b"model"
    assert destination.read_bytes() == b"model"

    with pytest.raises(TrainingError):
        copy_best_model(tmp_path / "missing.pt", destination)


def test_worker_warning_only_when_actual_differs():
    """Worker mismatch reporting is non-fatal and silent on exact matches."""
    assert render_worker_mismatch_warning(8, 8) is None
    assert render_worker_mismatch_warning(8, None) is None

    warning = render_worker_mismatch_warning(8, 0)

    assert warning is not None
    assert "WARNING" in warning
    assert "Requested workers : 8" in warning
    assert "Actual workers : 0" in warning
    assert "macOS/MPS" in warning


class _FakeModel:
    """Small Ultralytics-like model that records train/val kwargs."""

    def __init__(self, save_dir: Path, *, actual_workers: int = 6):
        self.train_kwargs: dict | None = None
        self.val_kwargs: dict | None = None
        best_path = save_dir / "weights/best.pt"
        best_path.parent.mkdir(parents=True, exist_ok=True)
        best_path.write_bytes(b"best model bytes")
        self.trainer = SimpleNamespace(
            args=SimpleNamespace(workers=actual_workers),
            best=best_path,
            best_epoch=7,
            save_dir=save_dir,
        )

    def train(self, **kwargs):
        self.train_kwargs = kwargs
        return SimpleNamespace(
            args=SimpleNamespace(workers=self.trainer.args.workers),
            save_dir=self.trainer.save_dir,
        )

    def val(self, **kwargs):
        self.val_kwargs = kwargs
        return SimpleNamespace(
            box=SimpleNamespace(mp=0.8, mr=0.7, map50=0.9, map=0.65)
        )


class _FakeFactory:
    """Factory that captures the weights path passed by run_training_pipeline."""

    def __init__(self, model: _FakeModel):
        self.model = model
        self.weights: list[str] = []

    def __call__(self, weights: str) -> _FakeModel:
        self.weights.append(weights)
        return self.model


def test_run_training_pipeline_passes_required_ultralytics_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The manager delegates to Ultralytics with the requested Mission 2 config."""
    monkeypatch.chdir(tmp_path)
    data_yaml = _make_dataset(tmp_path)
    save_dir = tmp_path / "runs/exp"
    model = _FakeModel(save_dir)
    factory = _FakeFactory(model)
    ticks = iter((10.0, 25.5))
    outputs: list[str] = []
    config = TrainingConfig(
        data_yaml=data_yaml,
        project=tmp_path / "runs",
        name="exp",
        epochs=3,
        seed=123,
        patience=4,
        resume_mode="no",
    )

    outcome = run_training_pipeline(
        config,
        model_factory=factory,
        torch_module=_fake_torch(cuda=True),
        output_fn=outputs.append,
        clock=lambda: next(ticks),
    )

    assert factory.weights == ["yolo11n.pt"]
    assert model.train_kwargs is not None
    assert model.train_kwargs["data"] == str(data_yaml)
    assert model.train_kwargs["imgsz"] == 640
    assert model.train_kwargs["batch"] == 42
    assert model.train_kwargs["epochs"] == 3
    assert model.train_kwargs["workers"] == 8
    assert model.train_kwargs["cache"] is True
    assert model.train_kwargs["optimizer"] == "auto"
    assert model.train_kwargs["seed"] == 123
    assert model.train_kwargs["patience"] == 4
    assert model.train_kwargs["project"] == str(tmp_path / "runs")
    assert model.train_kwargs["name"] == "exp"
    assert model.train_kwargs["device"] == "cuda"
    assert model.train_kwargs["resume"] is False

    assert model.val_kwargs is not None
    assert model.val_kwargs["split"] == "val"
    assert model.val_kwargs["device"] == "cuda"

    assert outcome.actual_workers == 6
    assert outcome.best_epoch == 7
    assert outcome.best_weights == (tmp_path / "models/best.pt").resolve()
    assert outcome.train_seconds == pytest.approx(15.5)
    assert outcome.validation.map50 == pytest.approx(0.9)
    assert (tmp_path / "models/best.pt").read_bytes() == b"best model bytes"

    joined_outputs = "\n".join(outputs)
    assert "Cache Enabled" in joined_outputs
    assert "Copied best model -> models/best.pt" in joined_outputs
    assert "Requested workers : 8" in joined_outputs
    assert "Actual workers : 6" in joined_outputs
    assert "Mission 2 Training Summary" in joined_outputs
    assert "[OK] Cache : Enabled" in joined_outputs

    summary_path = tmp_path / "logs/train_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["model"] == "yolo11n"
    assert summary["weights"] == "yolo11n.pt"
    assert summary["dataset_path"] == str(data_yaml.resolve())
    assert summary["number_of_classes"] == 3
    assert summary["train_images"] == 2
    assert summary["validation_images"] == 1
    assert summary["device"] == "NVIDIA CUDA"
    assert summary["batch"] == 42
    assert summary["requested_workers"] == 8
    assert summary["actual_workers"] == 6
    assert summary["epochs"] == 3
    assert summary["training_duration_seconds"] == pytest.approx(15.5)
    assert summary["best_epoch"] == 7
    assert summary["precision"] == pytest.approx(0.8)
    assert summary["recall"] == pytest.approx(0.7)
    assert summary["F1"] == pytest.approx(2 * 0.8 * 0.7 / (0.8 + 0.7))
    assert summary["mAP50"] == pytest.approx(0.9)
    assert summary["mAP50-95"] == pytest.approx(0.65)
    assert summary["best_model_path"] == str((tmp_path / "models/best.pt").resolve())
    assert "timestamp" in summary
    assert list((tmp_path / "logs").glob(".train_summary.json.*.tmp")) == []


def test_run_training_pipeline_resumes_from_last_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """When resume is explicit, the model is loaded from the detected last.pt."""
    monkeypatch.chdir(tmp_path)
    data_yaml = _make_dataset(tmp_path)
    config = TrainingConfig(
        data_yaml=data_yaml,
        project=tmp_path / "runs",
        name="exp",
        resume_mode="yes",
        cache=False,
    )
    checkpoint = last_checkpoint_path(config)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")

    model = _FakeModel(tmp_path / "runs/exp")
    factory = _FakeFactory(model)
    ticks = iter((0.0, 1.0))
    outputs: list[str] = []

    run_training_pipeline(
        config,
        model_factory=factory,
        torch_module=_fake_torch(),
        output_fn=outputs.append,
        clock=lambda: next(ticks),
    )

    assert factory.weights == [str(checkpoint)]
    assert model.train_kwargs is not None
    assert model.train_kwargs["resume"] is True
    assert model.train_kwargs["cache"] is False
    assert "Cache Disabled" in "\n".join(outputs)
