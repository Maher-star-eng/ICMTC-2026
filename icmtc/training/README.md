<!--
Author: Maher Ahmed Maher Shola
Email:  maher.shola.officer@gmail.com
        M.shoala61308@student.aast.edu
Date:   14/07/2026
-->

# training/

Everything off the live flight path: dataset handling, model training,
export, validation studies, and dev/smoke-test tooling. Nothing in this
folder runs during a mission — `core/`, `geo/`, `ioutils/`, and `vision/`
never import from here except read-only, at the edges (e.g. reusing a
decode helper for verification).

Dataset details (class list, split counts, background-image composition,
how to point `train.py` at a fresh download) live in
[`DATASET.md`](DATASET.md), not here.

## Production pipeline scripts

| File | Role |
|---|---|
| `train.py` | Dataset validation (`data.yaml` + label integrity, fail-loud `DatasetError`), dataset-quality statistics, device auto-selection, and the actual Ultralytics training run. Writes `logs/train_summary.json` and stages the best checkpoint to `models/best.pt`. Does **not** export ncnn — that's `export.py`'s job. |
| `export.py` | Exports `models/best.pt` → ncnn and verifies the export is *decode-compatible*, not just "exports without error": output-tensor layout, `.pt`-vs-ncnn detection parity, and that class-masking-before-argmax works correctly on the real exported tensor. |

## Validation studies (read the pipeline, write a report; not part of production)

These three build on each other in order and are the evidence behind two of
the decisions in the root [README](../README.md#key-engineering-decisions):

1. `masked_validation_experiment.py` → `masked_validation_report.md` — does
   before-argmax class masking recover recall on the 4 visually-similar
   flags (Egypt, Iraq, Syria, Yemen)?
2. `resolution_and_remap_report.py` → `resolution_and_remap_report.md` —
   builds on (1): is the shortfall a mislabeling problem, would a
   confusion-remap layer help, does a higher input resolution help?
3. `threshold_sweep_report.py` → `threshold_sweep_report.md` — builds on
   (1) and (2): sweeps confidence thresholds to find the operating point,
   checked against both the weak flags and the other 192 classes.

## Dev / smoke-test tooling (Mac-only, not part of a mission run)

| File | Role |
|---|---|
| `dry_run.py` → `dry_run_report.md` | First full-pipeline wiring test: real detector/tracker/geolocator/mission objects against synthetic footage and synthetic telemetry (including fault injection). Proves the wiring, not detection accuracy — that's the validation studies' job. |
| `webcam_live_test.py` | Live webcam smoke test of the real trained, exported detector — unmasked by default, to see raw per-class behavior. Not a mission run: no tracker, no geolocation. |

## Data / artifact directories

| Directory | Contents |
|---|---|
| `flag detection.v1i.yolov11/` | The dataset itself — see [`DATASET.md`](DATASET.md). |
| `models/` | Staged deployment artifacts (`best.pt`, ncnn `.bin`/`.param`) — what `export.py` and `vision/detector.py` actually load. |
| `runs/mission2/yolo11n_flags/` | Raw Ultralytics training-run output (curves, batch previews, `args.yaml`, `results.csv`, `weights/`) for the run that produced `models/best.pt`. |
| `runs/detect/val/` | A separate, unreferenced validation-run output — not part of the wired pipeline; flagged as a stray artifact, not deleted. |
| `logs/` | `train_summary.json`, the machine-readable summary `train.py` writes at the end of a run. |
| `dry_run_footage/`, `dry_run_judge_log*.json`, `dry_run_raw_log.txt` | Synthetic input and logged output backing `dry_run_report.md`'s evidence. |
| `webcam_captures/` | Output directory for `webcam_live_test.py`; empty until that script is run. |
| `yolo11n.pt` | Pretrained COCO base weights, the starting point `train.py` fine-tunes from. |

Every production and dev script above has a matching hardware-free test in
[`tests/`](../tests/) except the validation studies and `dry_run.py` /
`webcam_live_test.py`, which are one-off analysis/demo scripts rather than
importable library code.
