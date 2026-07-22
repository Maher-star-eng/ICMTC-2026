# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   03/06/2026

"""config.py - typed configuration loader & validator (ICMTC 2026, Mission 2).

Loads the mission's YAML config files (camera.yaml, mission.yaml, frames.yaml)
from disk, validates every field, and exposes an immutable, fully-typed
`AppConfig` that the rest of the in-process vision pipeline (capture / detect /
track / geo-locate / mission decisions) reads.  Every tunable - search
altitude, ground speed, confidence thresholds, tracker IoU/min_hits/max_age,
camera intrinsics, the GPS-dedup merge distance, the target flag class, and the
logging primitives - lives here as data, so the vehicle can be retuned in the
field without touching code.

SCOPE BOUNDARY
--------------
This module ONLY loads, validates, and exposes configuration.  It is a
dependency-free *leaf*: everyone imports it; it imports none of our pipeline
modules (geolocate, telemetry, tracker, detector, mission, output,
logging_cfg).  That is what keeps the import graph acyclic.  It performs no
logging setup, no geo-location math, no mission logic; it never touches
hardware or the network.  Pure: files/dicts in -> validated typed config out.

FAIL LOUD, FAIL EARLY
---------------------
A missing required field or an out-of-range value raises `ConfigError` at
startup - naming the offending key and its source - NOT a `KeyError` three
minutes into a flight.  This is deliberately the opposite of logging_cfg's
graceful-degradation rule: flying with wrong parameters is worse than not
flying, so bad config stops startup.  (Bad *logging* config still degrades
gracefully - see L2 below - because logging_cfg, not this module, owns that.)

================================================================================
ASSUMPTIONS (collected here; also marked inline with # ASSUMPTION: <tag>)
================================================================================

GENERAL
G1  FAIL-LOUD.  Any missing required key, wrong type, or out-of-range value
    raises `ConfigError` with the offending key and its source file/label.
    Required fields have no default; optional fields (documented below) do.
G2  UNKNOWN-KEY POLICY.  Keys we do not recognise are *warned* (RuntimeWarning)
    and ignored, never fatal - so a newer config file stays loadable by an
    older binary and vice-versa (forward/backward compatible).
G3  PyYAML is imported lazily, only inside the file-loading path.  The pure
    `from_dicts()` path needs no PyYAML and touches no filesystem, so unit
    tests can validate everything from in-memory dicts.
G4  Every leaf value must be *finite* (no NaN/inf); numeric fields declared as
    `int` reject `float` and `bool`; numeric fields declared as `float` reject
    `bool`.  (`bool` is an `int` subclass in Python; a stray `true` must not
    silently pass as 1.)

CAMERA (camera.yaml -> CameraConfig)
C1  fx, fy (focal lengths) must be finite and > 0.  cx, cy (principal point)
    must be finite; they are NOT bounds-checked against image size - a valid
    principal point may legitimately sit slightly outside the frame after
    calibration, so we only require finiteness, not [0, width]/[0, height].
C2  dist_coeffs is OPTIONAL; default () (empty tuple) meaning "no distortion".
    When present it must be a list/tuple of finite numbers (any length: the
    OpenCV 4/5/8/12/14-coefficient models are all accepted as-is).
C3  image_width, image_height must be int and > 0.

MISSION (mission.yaml -> MissionConfig)
M1  search_altitude_m must be finite and within [10, 150] m (outside -> error).
    Below 50 m it still loads but emits a RuntimeWarning, per the competition's
    -5 pt low-altitude rule; 50 m is the warn threshold, not a hard floor.
M2  ground_speed_mps must be finite and > 0.
M3  target_flag_class is REQUIRED (the audit found it missing) - int and >= 0.
M4  confidence_threshold must be finite and within [0, 1] (inclusive).
M5  dedup_merge_distance_m is OPTIONAL; default 5.0 m.  It is the GPS-space
    distance below which two confirmed tracks are merged as the same physical
    flag (the tracker gap we identified).  Must be finite and > 0.
M6  iou_threshold must be finite and within [0, 1] (inclusive).
M7  min_hits must be int and > 0.
M8  max_age must be int and >= 0 (0 = retire the moment a track is missed).
M9  lane_overlap must be finite and within [0, 1) - HALF-OPEN: 1.0 is rejected
    because a lane cannot fully overlap its neighbour.

FRAMES (frames.yaml -> FramesConfig)
F1  gimbal_reference_frame must be exactly "body" or "earth" (the choice fixed
    after the bench test); anything else -> ConfigError.  This is the semantic
    reference frame the operator declares.  telemetry.py's per-sample
    `gimbal_frame`, decoded live from MAVLink, uses this same "body"/"earth"
    vocabulary - MAVLink's own "vehicle" terminology is translated to "body"
    once, at telemetry.py's module boundary, and never appears past it.

DETECTOR CONFIDENCE THRESHOLDS (code constants, not YAML - DEFAULT_CONFIDENCE_THRESHOLD / SPECIAL_CONFIDENCE_THRESHOLDS)
T1  Validation found four classes - Egypt, Iraq, Syria, Yemen - with
    systematically lower confidence than the rest of the 196-class flag
    dataset (see training/masked_validation_experiment.py,
    training/threshold_sweep_report.py).  They keep a lower keep-threshold
    (0.25); every other class uses DEFAULT_CONFIDENCE_THRESHOLD (0.60).
    These are fixed operating constants from that validation, not a
    per-flight tunable, so they live here as code rather than in
    mission.yaml.
T2  SPECIAL_CONFIDENCE_THRESHOLDS is keyed by class NAME, never a numeric id:
    ids are an export-time accident of alphabetical class ordering (see
    training/flag detection.v1i.yolov11/data.yaml).  vision.detector resolves
    each name to its id once at startup, against the model's own
    class-name mapping, and raises loudly if a name can't be found - it
    never silently maps to the wrong id or falls back to the default
    threshold for a name that should have had its own.
T3  Supersedes MissionConfig.confidence_threshold (ASSUMPTION M4) for
    detector filtering.  That YAML field is left in place for schema
    back-compat (existing configs/tests still validate it) but is no
    longer read by vision.detector.build_detector.

LOGGING (primitives only - stored, never acted on here)
L1  log_level / log_dir live in mission.yaml under an optional `logging:`
    section (or may be passed explicitly to from_dicts()).  Defaults:
    log_level="INFO", log_dir=None (console-only).  They are handed verbatim to
    logging_cfg.configure_logging by mission.py; this module never calls it.
L2  An unrecognised log_level is WARNED, not fatal - logging_cfg owns the
    graceful fallback to INFO, and config must not be stricter about logging
    than logging is about itself.  (Contrast G1: everything else fails loud.)

================================================================================
USAGE
-----
mission.py calls `load_config()` once at startup, then wires the pieces out:

    cfg = load_config("config/")
    configure_logging(level=cfg.log_level, log_dir=cfg.log_dir)   # logging_cfg
    intrinsics = cfg.camera            # -> geolocate
    tracker = Tracker(cfg.mission.iou_threshold,
                      cfg.mission.min_hits, cfg.mission.max_age)   # -> tracker
    ...                                # cfg.mission.* drive the mission loop

Everything downstream reads the frozen `AppConfig`; nobody passes a
dict-of-magic-strings around.
================================================================================
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Validation bounds (every one carries an ASSUMPTION tag from the header above)

_ALT_MIN_M = 10.0            # ASSUMPTION M1
_ALT_MAX_M = 150.0           # ASSUMPTION M1
_ALT_WARN_BELOW_M = 50.0     # ASSUMPTION M1

_DEFAULT_DEDUP_MERGE_DISTANCE_M = 5.0   # ASSUMPTION M5

_DEFAULT_LOG_LEVEL = "INFO"  # ASSUMPTION L1
# Level names logging_cfg recognises; an unknown one warns (L2), never fails.
_KNOWN_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR"})

_VALID_GIMBAL_FRAMES = ("body", "earth")   # ASSUMPTION F1

# ASSUMPTION T1-T3: detector per-class confidence thresholds.  Public (no
# leading underscore) - vision.detector.build_detector imports these
# directly to translate names -> class ids at startup.
DEFAULT_CONFIDENCE_THRESHOLD = 0.60   # ASSUMPTION T1
SPECIAL_CONFIDENCE_THRESHOLDS: dict[str, float] = {   # ASSUMPTION T1 / T2
    "Egypt": 0.25,
    "Iraq": 0.25,
    "Syria": 0.25,
    "Yemen": 0.25,
}


class ConfigError(Exception):
    """Raised for any missing, malformed, or out-of-range configuration.

    The message always names the offending key and its source (file path when
    loaded from disk, or a logical label such as ``"mission.yaml"`` on the
    pure-dict path) so a failure is diagnosable at a glance at startup.
    """


# ---------------------------------------------------------------------------
# Frozen, fully-typed config objects
#
# Fields with a default are OPTIONAL (dataclass ordering forces them last);
# all others are REQUIRED.  `dist_coeffs` is a tuple, not a list, so the frozen
# guarantee is real - a list field would still be mutable in place.

@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Validated camera intrinsics (camera.yaml).  See ASSUMPTIONS C1-C3."""

    fx: float
    fy: float
    cx: float
    cy: float
    image_width: int
    image_height: int
    dist_coeffs: tuple[float, ...] = ()   # ASSUMPTION C2 (default = no distortion)


@dataclass(frozen=True, slots=True)
class MissionConfig:
    """Validated mission + tracker tunables (mission.yaml).  See M1-M9."""

    search_altitude_m: float
    ground_speed_mps: float
    target_flag_class: int          # ASSUMPTION M3 (required; the audit gap)
    confidence_threshold: float
    iou_threshold: float
    min_hits: int
    max_age: int
    lane_overlap: float
    dedup_merge_distance_m: float = _DEFAULT_DEDUP_MERGE_DISTANCE_M  # ASSUMPTION M5


@dataclass(frozen=True, slots=True)
class FramesConfig:
    """Validated reference-frame declaration (frames.yaml).  See F1."""

    gimbal_reference_frame: str   # "body" | "earth"


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Top-level immutable config composed from the three files + logging.

    `log_level`/`log_dir` are stored primitives (L1); this module never calls
    logging_cfg - mission.py hands them to configure_logging at startup.
    """

    camera: CameraConfig
    mission: MissionConfig
    frames: FramesConfig
    log_level: str = _DEFAULT_LOG_LEVEL   # ASSUMPTION L1
    log_dir: str | None = None            # ASSUMPTION L1


# ---------------------------------------------------------------------------
# Low-level validators.  Each takes the value, its key name, and a `source`
# label, and raises ConfigError naming both on failure (G1).

def _require(data: Mapping, key: str, source: str):
    """Return data[key] or raise ConfigError naming the missing key + source."""
    if key not in data:
        raise ConfigError(f"missing required key {key!r} in {source}")
    return data[key]


def _as_float(value: object, key: str, source: str) -> float:
    """Coerce to a finite float; reject bool (G4) and NaN/inf/non-numeric."""
    if isinstance(value, bool):
        raise ConfigError(
            f"{key} in {source} must be a number, not a bool; got {value!r}"
        )
    try:
        scalar = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ConfigError(
            f"{key} in {source} must be a number; got {value!r}"
        ) from None
    if not math.isfinite(scalar):
        raise ConfigError(f"{key} in {source} must be finite; got {value!r}")
    return scalar


def _as_int(value: object, key: str, source: str) -> int:
    """Require a true int; reject bool (G4) and float (e.g. 640.5 pixels)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            f"{key} in {source} must be an integer; got {value!r}"
        )
    return value


def _bounded_float(
    value: object,
    key: str,
    source: str,
    *,
    lo: float | None = None,
    hi: float | None = None,
    lo_inclusive: bool = True,
    hi_inclusive: bool = True,
) -> float:
    """Validate a finite float against optional lower/upper bounds."""
    scalar = _as_float(value, key, source)
    if lo is not None:
        too_low = scalar < lo if lo_inclusive else scalar <= lo
        if too_low:
            op = ">=" if lo_inclusive else ">"
            raise ConfigError(f"{key} in {source} must be {op} {lo}; got {scalar!r}")
    if hi is not None:
        too_high = scalar > hi if hi_inclusive else scalar >= hi
        if too_high:
            op = "<=" if hi_inclusive else "<"
            raise ConfigError(f"{key} in {source} must be {op} {hi}; got {scalar!r}")
    return scalar


def _bounded_int(
    value: object,
    key: str,
    source: str,
    *,
    lo: int | None = None,
    hi: int | None = None,
    lo_inclusive: bool = True,
    hi_inclusive: bool = True,
) -> int:
    """Validate a true int against optional lower/upper bounds."""
    number = _as_int(value, key, source)
    if lo is not None:
        too_low = number < lo if lo_inclusive else number <= lo
        if too_low:
            op = ">=" if lo_inclusive else ">"
            raise ConfigError(f"{key} in {source} must be {op} {lo}; got {number!r}")
    if hi is not None:
        too_high = number > hi if hi_inclusive else number >= hi
        if too_high:
            op = "<=" if hi_inclusive else "<"
            raise ConfigError(f"{key} in {source} must be {op} {hi}; got {number!r}")
    return number


def _warn_unknown(data: Mapping, known: set[str], source: str) -> None:
    """Warn (never fail) about unrecognised keys - forward-compat (G2)."""
    extra = set(data) - known
    if extra:
        warnings.warn(
            f"unknown key(s) {sorted(extra)} in {source} ignored "
            "(forward-compatible; not a fatal error)",
            RuntimeWarning,
            stacklevel=3,
        )


def _require_mapping(data: object, source: str) -> Mapping:
    """Ensure a section is a mapping/dict before we index into it."""
    if not isinstance(data, Mapping):
        raise ConfigError(
            f"{source} must be a mapping of key: value; got "
            f"{type(data).__name__}"
        )
    return data


# ---------------------------------------------------------------------------
# Per-file builders (dict -> validated dataclass)

_CAMERA_KEYS = {
    "fx", "fy", "cx", "cy", "image_width", "image_height", "dist_coeffs",
}
_MISSION_KEYS = {
    "search_altitude_m", "ground_speed_mps", "target_flag_class",
    "confidence_threshold", "dedup_merge_distance_m", "iou_threshold",
    "min_hits", "max_age", "lane_overlap",
    "logging",  # consumed for log_level/log_dir (L1); not a MissionConfig field
}
_FRAMES_KEYS = {"gimbal_reference_frame"}
_LOGGING_KEYS = {"log_level", "log_dir"}


def _build_camera(camera: object, source: str) -> CameraConfig:
    data = _require_mapping(camera, source)

    fx = _bounded_float(_require(data, "fx", source), "fx", source,
                        lo=0.0, lo_inclusive=False)   # ASSUMPTION C1
    fy = _bounded_float(_require(data, "fy", source), "fy", source,
                        lo=0.0, lo_inclusive=False)   # ASSUMPTION C1
    cx = _as_float(_require(data, "cx", source), "cx", source)  # ASSUMPTION C1
    cy = _as_float(_require(data, "cy", source), "cy", source)  # ASSUMPTION C1
    image_width = _bounded_int(_require(data, "image_width", source),
                               "image_width", source, lo=0, lo_inclusive=False)  # C3
    image_height = _bounded_int(_require(data, "image_height", source),
                                "image_height", source, lo=0, lo_inclusive=False)  # C3

    dist_coeffs = _build_dist_coeffs(data.get("dist_coeffs"), source)  # ASSUMPTION C2

    _warn_unknown(data, _CAMERA_KEYS, source)
    return CameraConfig(
        fx=fx, fy=fy, cx=cx, cy=cy,
        image_width=image_width, image_height=image_height,
        dist_coeffs=dist_coeffs,
    )


def _build_dist_coeffs(raw: object, source: str) -> tuple[float, ...]:
    """Optional distortion coefficients -> tuple of finite floats (C2)."""
    if raw is None:
        return ()
    # A str is a Sequence but never a valid coefficient list - exclude it.
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise ConfigError(
            f"dist_coeffs in {source} must be a list of numbers; got {raw!r}"
        )
    return tuple(
        _as_float(value, f"dist_coeffs[{i}]", source)
        for i, value in enumerate(raw)
    )


def _build_mission(mission: object, source: str) -> MissionConfig:
    data = _require_mapping(mission, source)

    search_altitude_m = _bounded_float(
        _require(data, "search_altitude_m", source), "search_altitude_m", source,
        lo=_ALT_MIN_M, hi=_ALT_MAX_M,
    )  # ASSUMPTION M1
    if search_altitude_m < _ALT_WARN_BELOW_M:  # ASSUMPTION M1 (-5 pt rule)
        warnings.warn(
            f"search_altitude_m={search_altitude_m} m is below "
            f"{_ALT_WARN_BELOW_M} m; expect the competition -5 pt low-altitude "
            "penalty. Loading anyway.",
            RuntimeWarning,
            stacklevel=3,
        )

    ground_speed_mps = _bounded_float(
        _require(data, "ground_speed_mps", source), "ground_speed_mps", source,
        lo=0.0, lo_inclusive=False,
    )  # ASSUMPTION M2
    target_flag_class = _bounded_int(
        _require(data, "target_flag_class", source), "target_flag_class", source,
        lo=0,
    )  # ASSUMPTION M3
    confidence_threshold = _bounded_float(
        _require(data, "confidence_threshold", source), "confidence_threshold",
        source, lo=0.0, hi=1.0,
    )  # ASSUMPTION M4
    iou_threshold = _bounded_float(
        _require(data, "iou_threshold", source), "iou_threshold", source,
        lo=0.0, hi=1.0,
    )  # ASSUMPTION M6
    min_hits = _bounded_int(
        _require(data, "min_hits", source), "min_hits", source, lo=0,
        lo_inclusive=False,
    )  # ASSUMPTION M7
    max_age = _bounded_int(
        _require(data, "max_age", source), "max_age", source, lo=0,
    )  # ASSUMPTION M8
    lane_overlap = _bounded_float(
        _require(data, "lane_overlap", source), "lane_overlap", source,
        lo=0.0, hi=1.0, hi_inclusive=False,   # ASSUMPTION M9 (half-open [0, 1))
    )
    dedup_merge_distance_m = _bounded_float(
        data.get("dedup_merge_distance_m", _DEFAULT_DEDUP_MERGE_DISTANCE_M),
        "dedup_merge_distance_m", source, lo=0.0, lo_inclusive=False,
    )  # ASSUMPTION M5

    _warn_unknown(data, _MISSION_KEYS, source)
    return MissionConfig(
        search_altitude_m=search_altitude_m,
        ground_speed_mps=ground_speed_mps,
        target_flag_class=target_flag_class,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        min_hits=min_hits,
        max_age=max_age,
        lane_overlap=lane_overlap,
        dedup_merge_distance_m=dedup_merge_distance_m,
    )


def _build_frames(frames: object, source: str) -> FramesConfig:
    data = _require_mapping(frames, source)

    frame = _require(data, "gimbal_reference_frame", source)
    if frame not in _VALID_GIMBAL_FRAMES:   # ASSUMPTION F1
        raise ConfigError(
            f"gimbal_reference_frame in {source} must be one of "
            f"{list(_VALID_GIMBAL_FRAMES)}; got {frame!r}"
        )

    _warn_unknown(data, _FRAMES_KEYS, source)
    return FramesConfig(gimbal_reference_frame=frame)


def _build_logging(section: object, source: str) -> tuple[str, str | None]:
    """Resolve (log_level, log_dir) from an optional logging section (L1/L2)."""
    if section is None:
        return _DEFAULT_LOG_LEVEL, None
    data = _require_mapping(section, source)

    log_level = data.get("log_level", _DEFAULT_LOG_LEVEL)
    if not isinstance(log_level, str):
        raise ConfigError(
            f"log_level in {source} must be a string; got {log_level!r}"
        )
    if log_level.upper() not in _KNOWN_LOG_LEVELS:   # ASSUMPTION L2 (warn, not fatal)
        warnings.warn(
            f"log_level={log_level!r} in {source} is not one of "
            f"{sorted(_KNOWN_LOG_LEVELS)}; logging_cfg will fall back to INFO.",
            RuntimeWarning,
            stacklevel=3,
        )

    log_dir = data.get("log_dir", None)
    if log_dir is not None and not isinstance(log_dir, str):
        raise ConfigError(
            f"log_dir in {source} must be a string path or null; got {log_dir!r}"
        )

    _warn_unknown(data, _LOGGING_KEYS, source)
    return log_level, log_dir


# ---------------------------------------------------------------------------
# Public API

def from_dicts(
    camera: dict,
    mission: dict,
    frames: dict,
    logging: dict | None = None,
    *,
    camera_source: str = "camera.yaml",
    mission_source: str = "mission.yaml",
    frames_source: str = "frames.yaml",
) -> AppConfig:
    """Validate three (already-parsed) config dicts into an `AppConfig`.

    This is the single pure dict -> AppConfig path: it touches no filesystem
    and needs no PyYAML, so all validation is unit-testable directly (G3).
    `load_config` is a thin YAML-reading wrapper around it.

    Args:
        camera:  Parsed contents of camera.yaml (see CameraConfig / C1-C3).
        mission: Parsed contents of mission.yaml (see MissionConfig / M1-M9).
        frames:  Parsed contents of frames.yaml (see FramesConfig / F1).
        logging: Optional explicit logging primitives ({"log_level", "log_dir"}).
            When None, they are read from mission["logging"] if present, else
            defaults apply (log_level="INFO", log_dir=None).  # ASSUMPTION L1
        camera_source / mission_source / frames_source: Human labels used in
            error messages; `load_config` passes the real file paths.

    Returns:
        A fully-populated, immutable `AppConfig`.

    Raises:
        ConfigError: on any missing required key, wrong type, or out-of-range
            value, naming the offending key and its source (G1).
    """
    camera_cfg = _build_camera(camera, camera_source)
    mission_cfg = _build_mission(mission, mission_source)
    frames_cfg = _build_frames(frames, frames_source)

    # Logging primitives: explicit arg wins; otherwise the mission file's
    # optional `logging:` section; otherwise defaults.  # ASSUMPTION L1
    if logging is not None:
        logging_source = "logging (explicit argument)"
        logging_section: object = logging
    else:
        logging_source = f"{mission_source}::logging"
        logging_section = mission.get("logging") if isinstance(mission, Mapping) else None
    log_level, log_dir = _build_logging(logging_section, logging_source)

    return AppConfig(
        camera=camera_cfg,
        mission=mission_cfg,
        frames=frames_cfg,
        log_level=log_level,
        log_dir=log_dir,
    )


def load_config(
    config_dir: str | Path,
    camera_file: str = "camera.yaml",
    mission_file: str = "mission.yaml",
    frames_file: str = "frames.yaml",
) -> AppConfig:
    """Load, parse, and validate the three YAML config files into an AppConfig.

    A missing directory or file, unreadable file, or malformed YAML raises
    `ConfigError` with a clear message - `FileNotFoundError`/`YAMLError` never
    leak out (G1).  Validation is delegated to `from_dicts`.

    Args:
        config_dir:  Directory holding the three YAML files.
        camera_file / mission_file / frames_file: File names within config_dir.

    Returns:
        A fully-populated, immutable `AppConfig`.

    Raises:
        ConfigError: if a file is missing/unreadable/malformed, or any value
            fails validation.
    """
    base = Path(config_dir)
    camera_path = base / camera_file
    mission_path = base / mission_file
    frames_path = base / frames_file

    camera = _load_yaml_mapping(camera_path)
    mission = _load_yaml_mapping(mission_path)
    frames = _load_yaml_mapping(frames_path)

    return from_dicts(
        camera, mission, frames,
        camera_source=str(camera_path),
        mission_source=str(mission_path),
        frames_source=str(frames_path),
    )


def _load_yaml_mapping(path: Path) -> dict:
    """Read one YAML file into a dict, converting every failure to ConfigError.

    PyYAML is imported here, lazily (G3), so the pure `from_dicts` path never
    depends on it.  An empty file parses to {} (so required-key checks fire with
    a precise message rather than a confusing "None has no key").
    """
    try:
        import yaml  # lazy: only the on-disk path needs PyYAML (ASSUMPTION G3)
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ConfigError(
            "PyYAML is required to load YAML config files but is not installed; "
            "`pip install pyyaml` (or use from_dicts() for the file-free path)"
        ) from exc

    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"malformed YAML in {path}: {exc}") from exc

    if data is None:   # empty file
        return {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"{path} must contain a YAML mapping at the top level; got "
            f"{type(data).__name__}"
        )
    return data
