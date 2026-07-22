<!--
Author: Maher Ahmed Maher Shola
Email:  maher.shola.officer@gmail.com
        M.shoala61308@student.aast.edu
Date:   14/07/2026
-->

# tests/

288 tests, one file per source module, all hardware-free. Run the whole
suite from the repo root:

```bash
pytest
```

## Layout

| Test file | Covers |
|---|---|
| `conftest.py` | Shared fixtures — notably a minimal `yaml` stand-in so config-loading tests still work in environments where PyYAML isn't importable. |
| `test_config.py` | `core/config.py` |
| `test_mission.py` | `core/mission.py` |
| `test_geolocate.py` | `geo/geolocate.py` |
| `test_telemetry.py` | `geo/telemetry.py` |
| `test_logging_cfg.py` | `ioutils/logging_cfg.py` |
| `test_output.py` | `ioutils/output.py` |
| `test_capture.py` | `vision/capture.py` |
| `test_detector.py` | `vision/detector.py` |
| `test_tracker.py` | `vision/tracker.py` |
| `test_infer.py` | `vision/infer.py` |
| `test_webcam_infer.py` | `vision/webcam_infer.py` |
| `test_train.py` | `training/train.py` |
| `test_export.py` | `training/export.py` |

`training/dry_run.py`, `training/masked_validation_experiment.py`,
`training/resolution_and_remap_report.py`, and
`training/threshold_sweep_report.py` are one-off analysis/demo scripts
rather than importable library code, and don't have a corresponding test
file — see [`training/README.md`](../training/README.md).

## Conventions

- **No hardware, no network.** No real camera, MAVLink link, GPU, or model
  runtime (`onnxruntime` / `torch` / `ncnn`) is imported by the test suite
  itself. Cameras are faked via injectable capture factories, models via
  dependency-injected stub/fake backends, telemetry via synthetic `Pose`
  samples fed directly into `TelemetryBuffer`.
- **Real I/O where it's cheap and meaningful.** Image decode/draw/save paths
  use real small PNGs written to pytest's `tmp_path` with OpenCV rather than
  mocking the I/O itself — only the model/camera/network boundary is faked.
- **One test file per source module**, named `test_<module>.py`, so a
  source file and its tests are trivially easy to find side by side.
