<!--
Author: Maher Ahmed Maher Shola
Email:  maher.shola.officer@gmail.com
        M.shoala61308@student.aast.edu
Date:   07/07/2026
-->

# ICMTC 2026 — UAVC-9 Mission 2: Autonomous Flag Detection & Geolocation

The vision subsystem for **ICMTC 2026, UAV Challenge Mission 2**: an
autonomous drone pipeline that detects national flags from the air,
identifies which country each one belongs to, geolocates each flag to a GPS
coordinate, and selects the best target for a payload drop. Vision runs
entirely on-board (Raspberry Pi 5) and hands a plain, dependency-free result
object to the autonomy stack — it never touches flight control directly.

This is a completed-competition archive: everything below is written and
tested, but the two pieces that require real flight hardware to verify —
live RTSP capture and an actual SIYI A8 telemetry link — are honestly
stubbed rather than faked (see [Hardware](#hardware) and the `capture.py`
notes below).

## Architecture

```
capture → detector → tracker → geolocator → output (VisionSystem) → mission → autonomy handoff
```

| Package | Role |
|---|---|
| [`core/`](core/README.md) | `config.py` — YAML config loading/validation (camera, mission, frame-convention settings), the only dependency-free leaf everyone imports. `mission.py` — the top-level orchestrator: wires every subsystem, runs the capture loop, drives the mission state machine, hands off to autonomy. |
| [`geo/`](geo/README.md) | `geolocate.py` — pure math: pixel + drone/gimbal pose + camera intrinsics → GPS coordinate, following ArduPilot NED conventions. `telemetry.py` — MAVLink pose ingestion on the Pi 5, exposing timestamp-interpolated pose lookups. |
| [`ioutils/`](ioutils/README.md) | `logging_cfg.py` — centralized diagnostic logging. `output.py` — the *only* file autonomy is allowed to depend on: `VisionSystem` (detect → track → geolocate → GPS-dedup → target-select, never raises) and `write_judge_log` (the competition result record). |
| [`vision/`](vision/README.md) | `capture.py` — frame sources (`FileFrameSource` for dev/test; `RtspFrameSource` deferred). `detector.py` — letterbox → backend inference → class-masked, per-class-threshold postprocess. `tracker.py` — IoU-gated multi-object tracking with per-track majority-vote class and median coordinate. `infer.py` / `webcam_infer.py` — standalone inspection tools for a trained model, no relation to the production decode path. |
| [`training/`](training/README.md) | `train.py` — dataset validation + YOLO11 training. `export.py` — `.pt` → ncnn export with decode-compatibility verification. `masked_validation_experiment.py`, `resolution_and_remap_report.py`, `threshold_sweep_report.py` — the three validation studies behind the confidence-threshold and masking decisions below. `dry_run.py` — first full-pipeline wiring test. |
| [`tests/`](tests/README.md) | 288 hardware-free tests, one file per source module. |

**Data flow in detail:** `Mission` (core/mission.py) pulls frames from a
`FrameSource` via a `LatestFrameQueue`, and for each frame calls
`VisionSystem.process_frame` (ioutils/output.py), which runs
`Detector.detect` → `Tracker.update` → `MedianGeolocatorAdapter.locate` →
GPS-dedup/target-select, and never raises into its caller — a vision fault
degrades the mission (`DEGRADED` status, last-known coordinates preserved),
it never crashes it. `Mission` folds each `MissionResult` into a state
machine (`SEARCHING` → `TRACKING` → `COVERAGE_COMPLETE` → `HANDOFF_COMPLETE`
→ `DONE`, plus `ABORT`), writes the judge log, and hands the final result to
autonomy as a plain in-process Python call — no ROS2, no sockets, since
autonomy runs in the same process.

## Hardware

- **Camera:** SIYI A8 mini gimbal camera, RTSP/H.265 stream.
- **Companion computer:** Raspberry Pi 5, running the vision pipeline and
  YOLO11n inference via **ncnn**.
- **Flight controller:** Cube (ArduPilot), connected to the Pi over MAVLink.
- **Model:** YOLO11n, trained on 196 country-flag classes (see
  [`training/DATASET.md`](training/DATASET.md)), exported to ncnn for
  on-Pi inference.

### RTSP / H.265 decode — problem space, not (yet) code

Real SIYI A8 + Raspberry Pi 5 hardware was never available during
development, so `vision/capture.py`'s `RtspFrameSource` deliberately raises
`NotImplementedError` instead of shipping an unverified pipeline — the
codebase ships `FileFrameSource` (video file / image-directory replay) as
the honest dev/test stand-in (see `capture.py` ASSUMPTION CAP10).

Decoding an H.265 RTSP stream on a Pi 5 has two real options, each with a
real cost, and the choice matters before this gets wired up against actual
hardware:

- **Hardware decode** (the Pi 5's V4L2 M2M HEVC decoder, e.g. a GStreamer
  `v4l2h265dec`-style element): low CPU cost, which matters a lot when the
  same core budget also has to run YOLO11n inference and the tracker every
  frame — but it's tied to the specific kernel/driver build shipped with
  whatever Raspberry Pi OS image is on the SD card. HEVC hardware-decode
  drivers on the Pi have historically had kernel-version-specific bugs
  (stalls/deadlocks under sustained streaming), so "hardware decode works"
  is really "hardware decode works *on this exact OS image*," and a routine
  `apt upgrade` can silently break it in the field.
- **Software decode** (e.g. GStreamer's `avdec_h265`, libavcodec-based):
  portable across any Pi OS/kernel version, no driver-specific breakage —
  but it's CPU-bound, and on a Pi 5 that CPU is already shared with
  inference and tracking. Under sustained 4K/high-bitrate H.265 at flight
  framerates, software decode risks eating enough CPU to cause dropped
  frames or added latency right when low, predictable latency matters most
  (target acquisition, payload-drop timing).

In practice, the mitigation that mattered most wasn't picking one decoder
over the other in code — it was pinning the Raspberry Pi OS image/kernel
version so the hardware-decode path stays stable, since re-flashing to a
different OS release was what actually resolved a decode-side stall
encountered during hardware bring-up. `RtspFrameSource` stays unimplemented
in this repo until it can be built and verified against the real camera and
a pinned OS image; wiring it up is the first task once that hardware is
available again.

## Key engineering decisions

### 1. Class-masking happens *before* argmax, not after

Early in development, class-restriction (only accept the 4 briefed flags)
was applied as a **post-hoc filter** on top of the detector's own unmasked,
196-class argmax — a class `_ClassMaskedDetector` wrapper. That's wrong:
if the correct class isn't already the *global* argmax winner, filtering
its output after the fact can only ever throw the detection away, never
recover it.

The fix (`vision/detector.py` ASSUMPTION D13, `_masked_class_argmax`)
restricts the score matrix to the target class columns **before** argmax
runs, so a non-target class is structurally incapable of winning a row —
the model's confidence in the *correct* class can win even when it isn't
the global maximum. `training/export.py` verifies this against the real
exported ncnn tensor (`masked_argmax_predictions`), and
`training/masked_validation_experiment.py` empirically proves the
distinction matters: it runs both the correct before-argmax masking and the
naive after-argmax filter side by side and shows the naive approach is a
strictly pessimistic lower bound. `_ClassMaskedDetector` has been deleted;
`core/mission.py` (ASSUMPTION MS9) and `training/export.py` (ASSUMPTION
E10) both call this **GAP-2, resolved**.

### 2. The median geolocator adapter

A single bad telemetry or gimbal-attitude sample, projected directly through
the pixel→GPS math, can put a flag's reported coordinate kilometres from
where it actually is — a nadir-adjacent gimbal-lock singularity or one noisy
attitude sample is enough. `MedianGeolocatorAdapter`
(`core/mission.py`, ASSUMPTION MS1) wraps `geo.geolocate.pixel_to_gps` with
a bounded per-track history (window of 5) and reports the per-axis
**median**, not the latest single-frame estimate.

`training/dry_run.py`'s fault-injection run reproduces this directly: a
synthetic gimbal glitch produces a single-frame estimate diverging
**1086 m** from the median-aggregated one at the same instant (see
`training/dry_run_report.md`) — concrete, logged evidence that the adapter
absorbs a kilometre-scale single-frame outlier in real time, not just in
theory.

### 3. Class-specific confidence thresholds for the 4 look-alike flags

Egypt, Iraq, Syria, and Yemen are all horizontal red-white-black tricolors
and collapsed into each other under the default 0.60 confidence threshold
at full 196-class scale. `training/threshold_sweep_report.py` swept
confidence in {0.25 … 0.60} under true before-argmax masking and found the
4 weak flags' recall recovers sharply as the threshold drops toward 0.25
(e.g. Iraq 0.774, Syria 0.742 recall at 0.25), while the 192 healthy classes
stay safe (worst-case F1 ≥ 0.936) and the 444 background/no-flag validation
images produce **zero** false positives at every threshold tested.

`core/config.py` (ASSUMPTION T1/T2) ships this as a **class-specific**
threshold — 0.25 for `{Egypt, Iraq, Syria, Yemen}`, 0.60 for everyone else —
applied by name, not id, in `vision/detector.py`. Worth flagging honestly:
`threshold_sweep_report.py`'s own written recommendation argues a *global*
0.25 would have been "sufficient" and that a class-specific rule "would add
complexity without a measurable benefit" for the classes it tested; the
shipped config is more conservative than that recommendation (keeping 0.60
as the default for the 192 classes the sweep only spot-checked 20 of,
rather than lowering the threshold for all 196). The sweep's numbers and the
shipped behavior aren't in conflict — the shipped code is just more
cautious than the report's narrow recommendation.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .              # core pipeline (numpy, scipy, PyYAML)
pip install -e ".[training]"  # + torch, torchvision, ultralytics, opencv-python, ncnn, onnxruntime
pip install -e ".[mavlink]"   # + pymavlink, for real telemetry
```

Python ≥ 3.10 required (see `pyproject.toml`).

## Running things

```bash
# Train (see training/DATASET.md for dataset details / fresh-download instructions)
python -m training.train --data "training/flag detection.v1i.yolov11/data.yaml"

# Export the trained .pt to ncnn and verify decode-compatibility
python -m training.export --weights training/models/best.pt

# Validate the class-masking / threshold decisions against the held-out val set
python -m training.masked_validation_experiment
python -m training.resolution_and_remap_report
python -m training.threshold_sweep_report

# First full-pipeline wiring check (synthetic footage + telemetry, no hardware)
python -m training.dry_run

# Inspect a trained model's predictions on real images
python -m vision.infer --weights training/runs/mission2/yolo11n_flags/weights/best.pt --source path/to/images/

# Live webcam demo (Mac dev machine, not a mission run)
python -m vision.webcam_infer
```

## Test suite

288 tests across `tests/`, one file per source module plus `conftest.py` for
shared fixtures. Every test is hardware-free — synthetic frames, in-memory
fakes for detector/tracker/geolocator/telemetry, and dependency-injected
stub backends stand in for the camera, MAVLink link, and any real model
runtime. Run the whole suite with:

```bash
pytest
```

## Reports and further reading

- [`training/DATASET.md`](training/DATASET.md) — dataset composition,
  class/split counts, how `train.py` consumes it.
- [`diagnostics_report.md`](diagnostics_report.md) — full-repo type-checker
  diagnostic sweep, root-caused into real bugs vs. cosmetic stub gaps.
- [`training/dry_run_report.md`](training/dry_run_report.md) — first
  full-pipeline wiring proof, including the median-geolocator glitch
  evidence above.
- [`training/masked_validation_report.md`](training/masked_validation_report.md),
  [`training/resolution_and_remap_report.md`](training/resolution_and_remap_report.md),
  [`training/threshold_sweep_report.md`](training/threshold_sweep_report.md) —
  the three studies behind the class-masking and confidence-threshold
  decisions above.

## Attribution

Author: Maher Ahmed Maher Shola
Email: maher.shola.officer@gmail.com, M.shoala61308@student.aast.edu
