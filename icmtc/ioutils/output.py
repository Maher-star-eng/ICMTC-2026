# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   15/06/2026

"""output.py - the vision <-> autonomy boundary (ICMTC 2026, Mission 2).

This is the ONLY file autonomy is allowed to depend on.  It defines:

  * `MissionResult` / `FlagReport` - the immutable, plain-serializable result
    contract vision hands to autonomy.
  * `VisionSystem` - the facade that runs the per-frame pipeline (detect ->
    track -> geolocate -> dedup-merge -> select) behind four dependency-
    injected, duck-typed collaborators, and NEVER raises into the caller.
  * `write_judge_log` - the atomic, structured per-flag GPS record written
    for competition submission.

Autonomy runs in the same Python process (no ROS2, no sockets) and calls a
plain Python method.  ArduPilot on the flight controller owns every flight-
critical failsafe, so a vision fault must degrade the mission, never crash
autonomy: `VisionSystem.process_frame` / `get_result` / `finalize` catch
their own exceptions and return a DEGRADED result instead.  `MissionResult`
is a plain, frozen, serializable data object (not a socket message) so the
transport can be swapped later (ZeroMQ/ROS2) without changing its shape.

SCOPE BOUNDARY
--------------
This module imports `config` (types) and `logging_cfg` only - never
`detector`/`tracker`/`geolocate`/`telemetry`.  Those four are dependency-
injected into `VisionSystem` and used only through the minimal, local
`typing.Protocol` contracts below (`DetectorLike`, `TrackerLike`, `TrackLike`,
`GeolocatorLike`, `TelemetryLike`), so this file is testable with plain
fakes and never touches hardware, models, or MAVLink.  This is also the
results/judge-log output - NOT the operational diagnostic log (that is
logging_cfg's job; see its own module docstring, which explicitly carves
this file out of its scope).

================================================================================
ASSUMPTIONS (collected here; also marked inline with # ASSUMPTION: <tag>)
================================================================================

O1  NEVER RAISE.  `process_frame`, `get_result`, and `finalize` never raise.
    Any internal exception is logged via `get_logger(__name__)` and converted
    to a DEGRADED `MissionResult`.  This is the fault-isolation guarantee:
    ArduPilot, not this module, owns flight-critical failsafes.

O2  DEGRADED SEMANTICS.  On DEGRADED, `target_flag_class`/`target_coord`/
    `target_confidence` are always None - a degraded frame withholds its
    target decision rather than handing autonomy a stale or unverified one.
    `all_flags` is preserved from the last known-good result (best-effort
    continuity for the judge log): a transient fault does not retroactively
    un-confirm flags already found.

O3  GEOLOCATION TIMING (deliberate deviation from tracker.py's own PIPELINE
    NOTE).  tracker.py's docstring describes "the mission loop" filling each
    Detection.coord before calling Tracker.update() so Track.committed_coord()
    aggregates it.  This file cannot do that: it may not import tracker.py,
    so it can never construct or mutate a `tracker.Detection` (whose type
    Tracker.update() also isinstance-checks).  Geolocation therefore runs
    AFTER tracking here, per confirmed track, via `geolocator.locate(track,
    pose)` - the geolocator (a separate, not-yet-written adapter around
    geolocate.py) reads whatever it needs (e.g. the track's last box) off the
    duck-typed track object itself.  `Track.committed_coord()` is simply
    never called by this file; that remains valid for other callers.

O4  POSE LOOKUP TIME.  `process_frame(frame, frame_idx)` receives no capture
    timestamp, so the monotonic instant used for `telemetry.pose_at(t)` is
    `time.monotonic()` sampled once at the top of `process_frame`,
    approximating "now" as capture time.  Acceptable because this pipeline is
    synchronous and in-process (no queueing between capture and processing);
    a future revision could thread a real capture timestamp through
    detector -> tracker -> output for exact per-frame pose alignment.

O5  LAST-KNOWN-COORD CACHE.  `VisionSystem` keeps a small per-track-id cache
    of the last successful geolocation.  If `telemetry.pose_at()` or
    `geolocator.locate()` cannot produce a coordinate this frame (e.g. a
    momentary telemetry gap, ASSUMPTION A8 in telemetry.py), a track already
    geolocated on a prior frame keeps reporting its last known coordinate
    instead of vanishing from `all_flags`.  A track never once geolocated is
    omitted from `all_flags` until it is.

O6  CONFIDENCE PROXY.  tracker.py deliberately does not aggregate
    Detection.score (see tracker.py ASSUMPTION T4); `Track` exposes only
    per-class hit counts (`class_votes`) and total `hits`.
    `FlagReport.confidence` is therefore the committed class's vote share
    (`class_votes[committed_class] / hits`) - a measure of class-identity
    stability, not detector confidence.  It is computed from Track's already-
    public fields (its own docstring: "public fields make the tracker easy to
    inspect") and needs no tracker.py change.  Not bounds-checked to [0, 1]
    beyond "finite", since the only producer of this value is this module's
    own division, which is structurally always in (0, 1].

O7  TARGET-SELECTION RULE.  `config.mission.target_flag_class` names a single
    class id, but more than one *distinct* confirmed flag may share that
    class (a briefing may post more than one physical target-class flag).
    When multiple confirmed flags (post-dedup) match, the highest-confidence
    match is reported as the target; ties are broken by the lowest
    `track_id` for determinism.  Every matching flag remains visible in
    `all_flags` regardless of which one is selected.

O8  SEARCHING VS NO_TARGET.  SEARCHING means no confirmed flag exists yet at
    all (`all_flags` empty).  NO_TARGET means one or more confirmed flags
    exist but none matches `target_flag_class`.  This lets autonomy tell
    "still building evidence" apart from "evidence says it isn't here."

O9  GPS DEDUP METRIC.  Flags are merged in GPS space using the same flat-
    earth equirectangular approximation as geolocate.py (111320.0 m/deg
    latitude, scaled by cos(mean latitude) for longitude; see geolocate.py
    conventions #9-#10).  The constant is duplicated, not imported (this
    module may not import geolocate.py); both are valid at the <=~100 m
    displacement scale relevant here.  Haversine would be more exact at
    global scale but is unneeded complexity at this scale.

O10 DEDUP MERGE GROUPING.  The "within dist_m" relation is treated as
    transitive: flags merge by connected components (union-find), not just
    pairwise-adjacent pairs, so an A-B-C chain of near neighbors merges into
    one flag rather than a partial/inconsistent merge.  Merging considers GPS
    proximity only, not class agreement - two physically-coincident tracks
    are assumed to be one physical flag even if their committed classes
    differ (possible with ambiguous viewing angles); the survivor's class is
    whichever member had higher confidence.

O11 DEDUP TIE-BREAK.  Within a merged group, the surviving `FlagReport` is
    the one with the highest confidence; ties are broken by the lowest
    `track_id` (same rule as O7, for consistency).

O12 JUDGE-LOG FORMAT.  `write_judge_log` serializes
    `{"flags": [FlagReport.to_dict(), ...]}` as indented JSON - deliberately
    narrower than the full `MissionResult` (no status/timestamp/target),
    because the spec scopes this file to "the per-flag GPS record" for
    competition submission, not a full mission dump.

O13 JUDGE-LOG ATOMICITY.  `write_judge_log` writes to a temp file in the SAME
    directory as the destination (so the final `os.replace()` is one
    filesystem/volume and therefore atomic on POSIX and Windows), fsync-ing
    the temp file before the rename so the write is durable across a crash,
    not just visually complete.  Directory-entry fsync (durability of the
    rename itself across power loss) is not performed; acceptable since a
    retry after a crash reproduces the same content deterministically.

O14 FINALIZE / JUDGE-LOG PATH.  `AppConfig` has no judge-log-output-path
    field today, so `finalize()`'s `judge_log_path` is an optional parameter
    (default None = skip writing) rather than assumed from `cfg`.  Call
    `finalize(judge_log_path=...)` explicitly at end-of-mission, or call
    `write_judge_log(path, vision.get_result())` directly.

O15 DUCK-TYPED DEPENDENCY CONTRACT.  detector/tracker/geolocator/telemetry
    are never imported; their expected shape is documented locally as
    minimal `typing.Protocol` classes covering only the methods/attributes
    `VisionSystem` actually calls, so real implementations and test fakes are
    both checked structurally, never via isinstance.

O16 NUMERIC VALIDATION.  `FlagReport`/`MissionResult` coerce every numeric
    leaf to a plain Python float/int (rejecting bool, non-finite values) in
    `__post_init__`, mirroring config.py/tracker.py's own frozen-dataclass
    validation style.  This is load-bearing, not decorative: it guarantees
    `to_dict()` never leaks a numpy scalar into `json.dumps`, and it means
    malformed data from any upstream dependency surfaces as a clean DEGRADED
    result (via O1) instead of a corrupt object reaching autonomy.

================================================================================
USAGE
-----
    vision = VisionSystem(cfg, detector=detector, tracker=tracker,
                           geolocator=geolocator, telemetry=telemetry)
    result = vision.process_frame(frame, frame_idx)   # once per frame
    ...
    final = vision.finalize(judge_log_path="results/judge_log.json")

Autonomy only ever calls `vision.get_result()` and reads the returned
`MissionResult`; it never imports detector/tracker/geolocate/telemetry.
================================================================================
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

import numpy as np

from core.config import AppConfig
from ioutils.logging_cfg import get_logger

log = get_logger(__name__)

#: Meters per degree of latitude, mirroring geolocate.py's own constant.
#: Duplicated, not imported - see ASSUMPTION O9.
_METERS_PER_DEG_LAT = 111320.0


class MissionStatus(Enum):
    """Vision-pipeline mission status handed to autonomy."""

    SEARCHING = "SEARCHING"
    TARGET_FOUND = "TARGET_FOUND"
    NO_TARGET = "NO_TARGET"
    DEGRADED = "DEGRADED"


# ---------------------------------------------------------------------------
# Validation helpers (mirrors tracker.py's/config.py's frozen-dataclass style)

def _validate_finite_float(name: str, value: object) -> float:
    """Coerce to a finite plain-Python float; reject bool and NaN/inf."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number, not a bool; got {value!r}")
    try:
        scalar = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a number; got {value!r}") from exc
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return scalar


def _validate_int(name: str, value: object) -> int:
    """Require a true int (rejects bool and float); returns a plain int."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer; got {value!r}")
    return int(value)


def _validate_coord(name: str, coord: object) -> tuple[float, float]:
    """Validate a (lat, lon) pair from any length-2 indexable (tuple/list/array)."""
    try:
        length = len(coord)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be a (lat, lon) pair; got {coord!r}") from exc
    if length != 2:
        raise TypeError(f"{name} must contain exactly two values; got {coord!r}")
    lat = _validate_finite_float(f"{name}[0]", coord[0])  # type: ignore[index]
    lon = _validate_finite_float(f"{name}[1]", coord[1])  # type: ignore[index]
    return lat, lon


def _validate_optional_int(name: str, value: object) -> int | None:
    return None if value is None else _validate_int(name, value)


def _validate_optional_finite_float(name: str, value: object) -> float | None:
    return None if value is None else _validate_finite_float(name, value)


def _validate_optional_coord(name: str, coord: object) -> tuple[float, float] | None:
    return None if coord is None else _validate_coord(name, coord)


# ---------------------------------------------------------------------------
# The contract: frozen, fully-typed, trivially serializable result objects.

@dataclass(frozen=True, slots=True)
class FlagReport:
    """One distinct confirmed physical flag, ready for the judge log.

    Attributes:
        flag_class: The confirmed (committed) detector class id.
        coord: `(lat, lon)` decimal degrees.
        track_id: The originating tracker id (see tracker.py `Track.id`).
        confidence: Committed-class vote share in (0, 1].  # ASSUMPTION O6
    """

    flag_class: int
    coord: tuple[float, float]
    track_id: int
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "flag_class", _validate_int("flag_class", self.flag_class))
        object.__setattr__(self, "coord", _validate_coord("coord", self.coord))
        object.__setattr__(self, "track_id", _validate_int("track_id", self.track_id))
        object.__setattr__(
            self, "confidence", _validate_finite_float("confidence", self.confidence)
        )

    def to_dict(self) -> dict[str, object]:
        """Plain-dict view: ints/floats/lists only, safe for `json.dumps`."""
        return {
            "flag_class": self.flag_class,
            "coord": list(self.coord),
            "track_id": self.track_id,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class MissionResult:
    """Immutable snapshot of the vision pipeline's mission-level judgement.

    This is the entire public data contract between vision and autonomy.
    See ASSUMPTION O2 for DEGRADED semantics and O7/O8 for how `status` and
    `target_*` are derived from `all_flags`.

    Attributes:
        status: One of SEARCHING / TARGET_FOUND / NO_TARGET / DEGRADED.
        target_flag_class: The briefed target's class id once found, else None.
        target_coord: `(lat, lon)` of the target flag, else None.
        target_confidence: Confidence of the selected target flag, else None.
        all_flags: Every distinct confirmed flag found so far (post-dedup).
        timestamp: `time.monotonic()` seconds when this result was produced.
    """

    status: MissionStatus
    target_flag_class: int | None
    target_coord: tuple[float, float] | None
    target_confidence: float | None
    all_flags: tuple[FlagReport, ...]
    timestamp: float

    def __post_init__(self) -> None:
        if not isinstance(self.status, MissionStatus):
            raise TypeError(f"status must be a MissionStatus; got {self.status!r}")
        object.__setattr__(
            self,
            "target_flag_class",
            _validate_optional_int("target_flag_class", self.target_flag_class),
        )
        object.__setattr__(
            self, "target_coord", _validate_optional_coord("target_coord", self.target_coord)
        )
        object.__setattr__(
            self,
            "target_confidence",
            _validate_optional_finite_float("target_confidence", self.target_confidence),
        )
        flags = tuple(self.all_flags)
        for flag in flags:
            if not isinstance(flag, FlagReport):
                raise TypeError(f"all_flags must contain FlagReport instances; got {flag!r}")
        object.__setattr__(self, "all_flags", flags)
        object.__setattr__(self, "timestamp", _validate_finite_float("timestamp", self.timestamp))

    def to_dict(self) -> dict[str, object]:
        """Plain-dict view: ints/floats/lists/strings only, safe for `json.dumps`."""
        return {
            "status": self.status.value,
            "target_flag_class": self.target_flag_class,
            "target_coord": (
                list(self.target_coord) if self.target_coord is not None else None
            ),
            "target_confidence": self.target_confidence,
            "all_flags": [flag.to_dict() for flag in self.all_flags],
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Duck-typed dependency contracts (ASSUMPTION O15).  Never imported for real;
# structural only, so fakes/stubs in tests satisfy them with no inheritance.

class TrackLike(Protocol):
    """What `VisionSystem` reads off a confirmed track (see tracker.py `Track`).

    `class_votes` is declared as a read-only `@property` rather than a plain
    attribute: pyright treats a plain Protocol attribute as read-write and
    requires the implementer's type to match invariantly, which `tracker.Track`
    (whose real field is `dict[int, int]`, mutated internally by
    `Track.add_detection`) can never satisfy against `Mapping[int, int]` -
    not because of a real substitution risk (this file only ever reads
    `class_votes`, never assigns to it - see `_build_flag_reports`), but
    because of that invariance rule.  A property is the idiomatic way to
    tell the type checker a Protocol member is read-only, which is the
    actual contract here; it does not add, remove, or widen what any real
    or fake track needs to expose - `dict[int, int]` already satisfies
    `Mapping[int, int]` covariantly once read-only is declared correctly.
    """

    id: int
    hits: int

    @property
    def class_votes(self) -> Mapping[int, int]: ...

    def committed_class(self) -> int | None: ...


class DetectorLike(Protocol):
    """One BGR frame in, opaque per-frame detections out.

    `VisionSystem` never inspects a detection's fields; it only forwards the
    returned sequence, unmodified, into `TrackerLike.update()`.
    """

    def detect(self, frame: np.ndarray) -> Sequence[object]: ...


class TrackerLike(Protocol):
    """Per-frame association.  Returns every CONFIRMED track found so far.

    Matches tracker.py's `Tracker.update()`, whose return value is already
    `confirmed_flags()` (tracker.py ASSUMPTION T9) - `VisionSystem` does not
    re-filter for confirmation.
    """

    def update(self, detections: Sequence[object], frame_idx: int) -> Sequence[TrackLike]: ...


class GeolocatorLike(Protocol):
    """Per-confirmed-track geolocation, called once per frame (ASSUMPTION O3).

    A real implementation wraps geolocate.py's pixel -> GPS math (plus camera
    intrinsics from `cfg.camera`) around the track's own most recent detector
    box.  Returns None when no coordinate can be computed right now (e.g. no
    ground intersection, or `pose` is None).
    """

    def locate(self, track: TrackLike, pose: object) -> tuple[float, float] | None: ...


class TelemetryLike(Protocol):
    """Monotonic time -> pose, or None if out of the buffered range (see
    telemetry.py ASSUMPTION A8).  `VisionSystem` treats `pose` as opaque and
    only ever passes it straight to `GeolocatorLike.locate()`.
    """

    def pose_at(self, t: float) -> object | None: ...


# ---------------------------------------------------------------------------
# Pure helpers: target selection and GPS-space dedup (independently testable,
# no dependency injection involved).

def _select_target(
    all_flags: tuple[FlagReport, ...], target_flag_class: int
) -> tuple[MissionStatus, int | None, tuple[float, float] | None, float | None]:
    """Pick the mission status and target fields from the confirmed flag set.

    See ASSUMPTION O7 (selection rule for multiple matches) and O8
    (SEARCHING vs NO_TARGET).
    """
    if not all_flags:
        return MissionStatus.SEARCHING, None, None, None

    matches = [flag for flag in all_flags if flag.flag_class == target_flag_class]
    if not matches:
        return MissionStatus.NO_TARGET, None, None, None

    # ASSUMPTION O7: highest confidence wins; ties broken by lowest track_id.
    best = max(matches, key=lambda flag: (flag.confidence, -flag.track_id))
    return MissionStatus.TARGET_FOUND, best.flag_class, best.coord, best.confidence


def _gps_distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Equirectangular flat-earth distance between two (lat, lon) points, meters.

    Mirrors geolocate.py conventions #9-#10.  # ASSUMPTION O9
    """
    lat1, lon1 = a
    lat2, lon2 = b
    mean_lat_rad = math.radians((lat1 + lat2) / 2.0)
    north_m = (lat2 - lat1) * _METERS_PER_DEG_LAT
    east_m = (lon2 - lon1) * _METERS_PER_DEG_LAT * math.cos(mean_lat_rad)
    return math.hypot(north_m, east_m)


def _merge_duplicate_flags(
    flags: Sequence[FlagReport], dist_m: float
) -> tuple[FlagReport, ...]:
    """Collapse confirmed flags that are the same physical flag in GPS space.

    Two flags whose coordinates are within `dist_m` of each other are
    considered duplicate observations of one physical flag (the cross-lane
    duplicate problem).  Merge groups are the connected components of the
    "within dist_m" relation, not just pairwise-adjacent pairs (ASSUMPTION
    O10), so an A-B-C chain merges into a single flag.  Merging looks only at
    GPS proximity, never at class agreement (O10).  Within a merged group,
    the surviving flag is the one with the highest confidence, ties broken by
    the lowest `track_id` (ASSUMPTION O11).

    Args:
        flags: Confirmed flags for the current frame, any order.
        dist_m: Merge distance in meters (`config.mission.dedup_merge_distance_m`).

    Returns:
        A tuple of distinct `FlagReport`s, sorted by `track_id`, with no two
        survivors within `dist_m` of each other.
    """
    flags = list(flags)
    n = len(flags)
    if n <= 1:
        return tuple(flags)

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for i in range(n):
        for j in range(i + 1, n):
            if _gps_distance_m(flags[i].coord, flags[j].coord) <= dist_m:
                union(i, j)

    groups: dict[int, list[FlagReport]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(flags[i])

    merged = [
        max(group, key=lambda flag: (flag.confidence, -flag.track_id))
        for group in groups.values()
    ]
    return tuple(sorted(merged, key=lambda flag: flag.track_id))


# ---------------------------------------------------------------------------
# The facade.

class VisionSystem:
    """The single public entry point between the vision subsystem and autonomy.

    Dependencies are injected (duck-typed against the `*Like` Protocols
    above), so this file imports none of detector/tracker/geolocate/telemetry
    and is fully testable with fakes.  `process_frame`, `get_result`, and
    `finalize` never raise (ASSUMPTION O1); autonomy must depend only on this
    class and the `MissionResult`/`FlagReport` contract above.
    """

    def __init__(
        self,
        cfg: AppConfig,
        *,
        detector: DetectorLike,
        tracker: TrackerLike,
        geolocator: GeolocatorLike,
        telemetry: TelemetryLike,
    ) -> None:
        self._cfg = cfg
        self._detector = detector
        self._tracker = tracker
        self._geolocator = geolocator
        self._telemetry = telemetry
        self._last_known_coord: dict[int, tuple[float, float]] = {}  # ASSUMPTION O5
        self._last_result = MissionResult(
            status=MissionStatus.SEARCHING,
            target_flag_class=None,
            target_coord=None,
            target_confidence=None,
            all_flags=(),
            timestamp=time.monotonic(),
        )

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> MissionResult:
        """Run detect -> track -> geolocate -> dedup-merge -> select for one frame.

        Never raises (ASSUMPTION O1): any internal failure is logged and
        turned into a DEGRADED `MissionResult` that preserves the last known
        `all_flags` (ASSUMPTION O2).

        Args:
            frame: The captured BGR frame, forwarded opaquely to the detector.
            frame_idx: Strictly increasing frame index, forwarded to the tracker.

        Returns:
            The `MissionResult` for this frame (or a DEGRADED one on fault).
        """
        try:
            now = time.monotonic()  # ASSUMPTION O4: approximates capture time
            detections = self._detector.detect(frame)
            confirmed_tracks = self._tracker.update(detections, frame_idx)
            pose = self._telemetry.pose_at(now)

            flags = self._build_flag_reports(confirmed_tracks, pose)
            merged_flags = _merge_duplicate_flags(
                flags, self._cfg.mission.dedup_merge_distance_m
            )
            status, target_class, target_coord, target_confidence = _select_target(
                merged_flags, self._cfg.mission.target_flag_class
            )
            result = MissionResult(
                status=status,
                target_flag_class=target_class,
                target_coord=target_coord,
                target_confidence=target_confidence,
                all_flags=merged_flags,
                timestamp=now,
            )
        except Exception:
            log.exception(
                "process_frame failed on frame_idx=%s; degrading mission result",
                frame_idx,
            )
            return self._degrade()

        self._last_result = result
        return result

    def _build_flag_reports(
        self, tracks: Sequence[TrackLike], pose: object
    ) -> tuple[FlagReport, ...]:
        """Geolocate each confirmed track and package it as a `FlagReport`.

        Tracks with no coordinate available yet (this frame or ever) are
        omitted; see ASSUMPTION O5 for the last-known-coord cache and
        ASSUMPTION O3 for why geolocation runs here, after tracking.
        """
        reports = []
        for track in tracks:
            coord = None
            if pose is not None:
                coord = self._geolocator.locate(track, pose)
                if coord is not None:
                    self._last_known_coord[track.id] = coord
            if coord is None:
                coord = self._last_known_coord.get(track.id)
            if coord is None:
                continue  # never geolocated yet; not reportable this frame

            committed_class = track.committed_class()
            if committed_class is None:
                continue  # confirmed but no class vote yet (shouldn't happen)

            confidence = track.class_votes[committed_class] / track.hits  # ASSUMPTION O6
            reports.append(
                FlagReport(
                    flag_class=committed_class,
                    coord=coord,
                    track_id=track.id,
                    confidence=confidence,
                )
            )
        return tuple(reports)

    def get_result(self) -> MissionResult:
        """Return the latest `MissionResult` without processing a new frame.

        Never raises: before the first `process_frame` call this returns the
        initial SEARCHING result set up in `__init__`.
        """
        return self._last_result

    def finalize(self, judge_log_path: str | Path | None = None) -> MissionResult:
        """End-of-mission: optionally flush the judge log, return the final result.

        Never raises (ASSUMPTION O1).  `judge_log_path` is optional because
        `AppConfig` has no judge-log-path field today (ASSUMPTION O14); pass
        it explicitly to have this call `write_judge_log` for you, or write
        it yourself from the returned result.

        Args:
            judge_log_path: Destination path for `write_judge_log`, or None
                (default) to skip writing.

        Returns:
            The final `MissionResult` (or a DEGRADED one if flushing fails).
        """
        try:
            result = self._last_result
            if judge_log_path is not None:
                write_judge_log(judge_log_path, result)
            return result
        except Exception:
            log.exception("finalize failed while flushing the judge log")
            return self._degrade()

    def _degrade(self) -> MissionResult:
        """Build, record, and return a DEGRADED result (ASSUMPTION O2)."""
        result = MissionResult(
            status=MissionStatus.DEGRADED,
            target_flag_class=None,
            target_coord=None,
            target_confidence=None,
            all_flags=self._last_result.all_flags,
            timestamp=time.monotonic(),
        )
        self._last_result = result
        return result


# ---------------------------------------------------------------------------
# Judge-log writer (results output; NOT the operational log - see module
# docstring SCOPE BOUNDARY and logging_cfg.py's own scope note).

def write_judge_log(path: str | Path, result: MissionResult) -> None:
    """Atomically write the per-flag GPS record for competition submission.

    Writes `{"flags": [FlagReport.to_dict(), ...]}` as indented JSON
    (ASSUMPTION O12).  The write is atomic: content goes to a temp file in
    the destination's own directory, is fsync-ed, and is then moved into
    place with `os.replace` (ASSUMPTION O13), so a crash mid-write can never
    leave a corrupt or partial file at `path`.

    Args:
        path: Destination file path.  Parent directories are created if needed.
        result: The `MissionResult` whose `all_flags` to serialize.

    Raises:
        OSError: if the directory cannot be created or the write/replace fails.
            (Unlike `VisionSystem`'s methods, this function is a plain I/O
            primitive and is allowed to raise; `VisionSystem.finalize` is
            what catches it to preserve the never-raise guarantee.)
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = {"flags": [flag.to_dict() for flag in result.all_flags]}

    fd, tmp_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, destination)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_name)
        raise
