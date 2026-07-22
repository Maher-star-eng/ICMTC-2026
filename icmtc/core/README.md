<!--
Author: Maher Ahmed Maher Shola
Email:  maher.shola.officer@gmail.com
        M.shoala61308@student.aast.edu
Date:   14/07/2026
-->

# core/

Configuration and mission orchestration — the two modules everything else in
the pipeline is wired up by or configured through.

| File | Role |
|---|---|
| `config.py` | Dependency-free leaf. Loads and validates `camera.yaml` / `mission.yaml` / `frames.yaml` into frozen, fully-typed dataclasses (`CameraConfig`, `MissionConfig`, `FramesConfig`, `AppConfig`). Fails loud on malformed required fields, warns (never fails) on unknown keys. Also defines the per-class confidence-threshold constants (`DEFAULT_CONFIDENCE_THRESHOLD`, `SPECIAL_CONFIDENCE_THRESHOLDS`) that `vision/detector.py` reads. |
| `mission.py` | Top-level orchestrator. Wires `VisionSystem` (`ioutils/output.py`), the capture loop, and the mission state machine (`INIT → SEARCHING ↔ TRACKING → COVERAGE_COMPLETE → HANDOFF_COMPLETE → DONE`, plus `ABORT`). Defines `MedianGeolocatorAdapter`, which wraps `geo/geolocate.py`'s per-frame math with median aggregation to suppress single-frame gimbal-glitch outliers (see the root [README](../README.md#2-the-median-geolocator-adapter)). `main()` holds the real-hardware wiring, outside the hardware-free test suite. |

Everything here is unit-tested with in-memory fakes — see
[`tests/test_config.py`](../tests/test_config.py) and
[`tests/test_mission.py`](../tests/test_mission.py). No camera, model
runtime, or MAVLink link involved.
