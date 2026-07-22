<!--
Author: Maher Ahmed Maher Shola
Email:  maher.shola.officer@gmail.com
        M.shoala61308@student.aast.edu
Date:   14/07/2026
-->

# ioutils/

Logging setup and the vision ↔ autonomy boundary.

| File | Role |
|---|---|
| `logging_cfg.py` | Centralized diagnostic logging for the whole pipeline. `configure_logging()` is idempotent (safe to call more than once — it replaces its own handlers rather than stacking them) and attaches a console handler and/or a size-rotated file handler. `get_logger(name)` is the thin wrapper every other module uses. Operational/diagnostic logging only — not the judge log, not mission results. |
| `output.py` | **The only file autonomy is allowed to depend on.** Defines `MissionResult` / `FlagReport` (the immutable, plain-serializable result contract), `VisionSystem` (the facade that runs detect → track → geolocate → GPS-dedup → target-select behind duck-typed Protocols, and never raises into its caller), and `write_judge_log` (the atomic, structured per-flag GPS record written for competition submission). |

`output.py` never imports `vision/detector.py`, `vision/tracker.py`, or
`geo/geolocate.py` directly — it depends on local Protocol types
(`DetectorLike`, `TrackerLike`, `GeolocatorLike`, ...) so it stays testable
and hardware-free in isolation. See
[`tests/test_logging_cfg.py`](../tests/test_logging_cfg.py) and
[`tests/test_output.py`](../tests/test_output.py).
