<!--
Author: Maher Ahmed Maher Shola
Email:  maher.shola.officer@gmail.com
        M.shoala61308@student.aast.edu
Date:   14/07/2026
-->

# vision/

Frame sourcing, detection, and tracking — the per-frame path from a camera
frame to tracked, GPS-ready flag detections.

| File | Role |
|---|---|
| `capture.py` | Frame sources. `FileFrameSource` reads a recorded video file or a directory of sequential images (the dev/test stand-in). `LatestFrameQueue` wraps any frame source on a background thread as a single-slot "always newest frame" buffer. `RtspFrameSource` is a documented `NotImplementedError` stub — no SIYI A8 + Raspberry Pi 5 hardware has been available to build or verify it against (see root [README](../README.md#rtsp--h265-decode--problem-space-not-yet-code)). |
| `detector.py` | Letterbox preprocess → pluggable backend inference (`StubBackend` for tests; `OnnxBackend` / `TorchBackend` / `NcnnBackend` scaffolds for real models) → class-masked, per-class-threshold, per-class-NMS postprocess. Class-masking is applied **before** argmax, not after (see root README, decision #1) — this is the socket a trained model plugs into, not a trained detector itself. |
| `tracker.py` | Pure computation, no I/O. Associates per-frame detections into stable per-track flags via greedy IoU-gated matching, confirms a track once `hits >= min_hits`, and commits a per-track class (majority vote) and coordinate (per-axis median). |
| `infer.py` | Standalone CLI (`python -m vision.infer`) for inspecting a *trained* model's predictions on real images via Ultralytics' own pre/postprocess — read-only with respect to weights, no relation to `detector.py`'s ncnn decode path. |
| `webcam_infer.py` | Live MacBook-webcam demo, reusing `infer.py`'s detection/drawing code with a live device-selection banner (MPS/CUDA/CPU). Dev/demo tooling only, not part of the mission pipeline. |

Every test here is hardware- and model-free — synthetic frames, stub
backends, and dependency-injected camera/GUI seams. See
[`tests/test_capture.py`](../tests/test_capture.py),
[`tests/test_detector.py`](../tests/test_detector.py),
[`tests/test_tracker.py`](../tests/test_tracker.py),
[`tests/test_infer.py`](../tests/test_infer.py), and
[`tests/test_webcam_infer.py`](../tests/test_webcam_infer.py).
