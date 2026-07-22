# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   19/06/2026

"""mission.py - top-level mission orchestrator (ICMTC 2026, Mission 2).

    capture -> detector -> tracker -> geolocate -> output -> [ mission ] -> autonomy

`output.VisionSystem` already orchestrates the per-frame vision pipeline
(detect -> track -> geolocate -> GPS-dedup -> target-select) and never raises
into its caller.  This file sits one level up: it (1) builds the geolocator
adapter `VisionSystem` depends on, (2) wires and starts every subsystem in the
correct order, (3) runs the capture loop, (4) drives the mission state
machine, and (5) hands off to autonomy.  Everything is dependency-injected so
the whole thing runs on a Mac with a `FileFrameSource` and synthetic
telemetry, with zero hardware (see `test_mission.py`).

SCOPE BOUNDARY
--------------
Mission scope ends at HANDOFF_COMPLETE: this file owns
INIT -> SEARCHING <-> TRACKING -> COVERAGE_COMPLETE -> HANDOFF_COMPLETE -> DONE
(plus ABORT).  It does NOT implement autonomy's TARGETING/DROPPING/RETURNING
states - those live downstream, past the one clean handoff point exposed by
`Mission.final_result()`.

================================================================================
ASSUMPTIONS (collected here; also marked inline with # ASSUMPTION: <tag>)
================================================================================

THE FIX - MedianGeolocatorAdapter
----------------------------------
`output.VisionSystem` calls `geolocator.locate(track, pose)` once per frame
per confirmed track and reports whatever comes back (output.py ASSUMPTION
O3) - it does not aggregate across frames.  `tracker.Track.committed_coord()`
is never usable either, because every `Detection.coord` this pipeline
produces is `None` (detector.py ASSUMPTION D9) - `Track.coords` never fills.
A single bad gimbal frame therefore corrupts the reported coordinate by
kilometres.  `MedianGeolocatorAdapter` re-introduces median aggregation at
the geolocation seam: it keeps a bounded per-track history of single-frame
pixel->GPS estimates and reports `geolocate.aggregate_coords()` (the
per-axis median), never the latest frame alone.

MS1  MEDIAN HISTORY WINDOW.  Default 5 samples per track (`history_window=`
     overrides).  Deliberately small rather than a large fixed constant like
     50: flags are static but a track may only be confirmed and visible for a
     short pass, so the median should converge within a handful of frames of
     confirmation rather than needing dozens; 5 still means a single wild
     frame is always outvoted by at least 4 good ones.  Memory is flat over a
     long flight either way once a track's history fills.  Choose a larger
     `history_window=` for a slower/smoother aggregate if desired.

MS2  DIST COEFFS.  `cfg.camera.dist_coeffs` defaults to `()` (config.py
     ASSUMPTION C2, "no distortion").  An empty tuple is treated as `None` so
     `geolocate.pixel_to_ray` never enters its (cv2-dependent) undistort
     branch for the common no-distortion case.

MS3  GROUND-CONTACT PIXEL.  `u = (x1+x2)/2`, `v = y2` - the bottom-center of
     the track's last box, per geolocate.py convention #13 (the flag lies
     flat on the ground; the box center is NOT the ground-contact point).

MS4  GIMBAL FRAME DEFAULT.  `pose.gimbal_frame` is only `None` when telemetry
     never received gimbal-frame metadata at all.  That case defaults to
     "body" (frames.yaml's own default vocabulary, ASSUMPTION F1 in
     config.py) and logs exactly one warning per adapter instance, not one
     per frame.

MS5  EULER SINGULARITY.  Nadir (`gimbal_pitch = -pi/2`), this mission's
     primary search attitude, is the ZYX-Euler gimbal-lock singularity for
     `_quat_to_euler`.  scipy's "Gimbal lock detected" `UserWarning` is
     expected there and is suppressed rather than spamming the flight log
     every frame: the *composed* rotation `Rz(yaw)@Ry(pitch)@Rx(roll)` is
     reconstructed exactly regardless of which of the (infinitely many)
     equivalent yaw/roll splits scipy picks, so the projected ray - and
     therefore the reported coordinate - is unaffected by the ambiguity.

MS6  GROUND-MISS / BAD-TELEMETRY HANDLING.  A `ValueError` from geolocate.py
     (the ray never meets the ground - e.g. a bad tilt frame) or a
     `TypeError` from an unusable gimbal quaternion (e.g. `None`) are treated
     identically: that frame contributes nothing to history; the adapter
     returns the previously-computed median (or `None` if the track has
     never geolocated at all).  `locate()` never raises.

MS7  ALTITUDE REFERENCE.  `pose.alt_agl` is `GLOBAL_POSITION_INT.relative_alt`
     - metres above the ArduPilot Home waypoint (telemetry.py ASSUMPTION A3),
     NOT true terrain AGL.  It is passed straight through to geolocate.py's
     `altitude_agl` unmodified; if Home's elevation differs from the flag's
     local ground plane, that mismatch becomes projection error.  No
     correction is applied here - flagged, not silently treated as correct.

Two real gaps (surfaced, not papered over)
-------------------------------------------
MS8  GAP-1: BRIEFED TARGET IDS HAVE NO CONFIG HOME.
     `config.MissionConfig.target_flag_class` is a single int (the drop
     target); there is no config field for the 4 briefed class ids that
     class-masking and target reporting both need.  `Mission.__init__`
     therefore requires `briefed_target_class_ids` as an explicit,
     non-defaulted constructor argument and raises `ValueError` if
     `target_flag_class` is not one of them - this gap can never be
     silently missed.  The real fix is a new list field on
     `mission.yaml`/`MissionConfig`.

MS9  GAP-2: LIVE CLASS-MASKING - RESOLVED.
     `detector.py`'s live decode (`_decode_yolov8`, via
     `Detector.target_class_ids` / `Detector.set_target_classes`) now masks
     class-score columns to the briefed set BEFORE argmax, not after - the
     same mechanism `export.py`'s offline `masked_argmax_predictions`
     verifies (both call `vision.detector._masked_class_argmax`; see
     detector.py ASSUMPTION D13).  A masked class is structurally
     incapable of winning the argmax, so it can never suppress a true
     briefed-class box at the same anchor the way the old post-hoc filter
     could.  The former stopgap, `_ClassMaskedDetector` (a filter applied
     AFTER the wrapped detector's own unmasked argmax), has been deleted -
     `Mission` now passes `detector` straight through to `VisionSystem`
     unwrapped, trusting that whoever constructed `detector` (see `main()`
     below) already called `set_target_classes(briefed_target_class_ids)`
     on it.  `Mission` itself does no class filtering of any kind any more.

Orchestration
-------------
MS10 COVERAGE-COMPLETE TRIGGER (autonomy interface - STUB).  This fires when
     autonomy decides the search pattern is done.  It is an injectable
     zero-argument predicate (`coverage_complete_check`); absent an explicit
     autonomy handshake, the default is "the frame source is exhausted" - the
     only signal available in a Mac file-replay dry run.
     # TODO: autonomy coverage-complete interface.

MS11 STATE MACHINE SCOPE.  `MissionState` covers INIT..DONE/ABORT only;
     TARGETING/DROPPING/RETURNING belong to autonomy and are never modeled
     here (see SCOPE BOUNDARY above).

MS12 DEGRADED TOLERANCE.  A DEGRADED `MissionResult` never drives a state
     transition and never raises; it only increments a counter and logs, so
     a transient vision fault - already isolated by output.py's own
     never-raise guarantee (ASSUMPTION O1) - cannot stall or crash the
     mission loop.

MS13 PROTOCOL-HONEST BOX/POSE ACCESS.  `MedianGeolocatorAdapter.locate` is
     declared as `(track: TrackLike, pose: object)`, matching
     `output.GeolocatorLike` exactly, so it structurally satisfies the
     Protocol it is passed as at the `Mission(...)` construction site - no
     widened local Protocol, no `# type: ignore`.  `TrackLike` does not
     guarantee `.box`, and `GeolocatorLike` treats `pose` as fully opaque,
     but this adapter genuinely needs both: `track.box` for the
     ground-contact pixel (MS3) and every `Pose` field for the pixel->GPS
     math.  Both are therefore read defensively instead of assumed:
     `track.box` via `getattr(track, "box", None)` - tracks seen in tests
     (and any hand-written `TrackerLike` fake) are plain structural stand-ins,
     not `vision.tracker.Track` instances, so an `isinstance` check would be
     wrong here; `pose` via `isinstance(pose, Pose)` - `pose` always
     originates from `TelemetryLike.pose_at()`, so "a real `Pose`, or
     nothing usable" is the only realistic shape, and `isinstance` narrows
     every field for the rest of the method in one check.  Either gap is
     treated exactly like a ground-miss/bad-telemetry frame (MS6): the frame
     contributes nothing, the existing per-track median (or `None`) is
     returned, and a warning is logged once per adapter instance, not once
     per frame - `locate()` still never raises.  `Tracker.update()`'s own
     parameter type has the same fix applied on the tracker.py side - see
     tracker.py ASSUMPTION T12.

USAGE
-----
    cfg = load_config("config/")
    mission = Mission(cfg, detector=..., tracker=..., geolocator=...,
                       telemetry=..., frame_queue=...,
                       briefed_target_class_ids=(0, 1, 2, 3))
    mission.run()
    result = mission.final_result()
Real hardware wiring lives behind `main()` / `if __name__ == "__main__"`
below; the `Mission` class itself is the fully dependency-injected,
hardware-free, testable core (see test_mission.py).
================================================================================
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Iterable
from enum import Enum
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation

import geo.geolocate as geo
from core.config import AppConfig, load_config
from geo.telemetry import Pose
from ioutils.logging_cfg import configure_logging, get_logger
from ioutils.output import (
    DetectorLike,
    GeolocatorLike,
    MissionResult,
    MissionStatus,
    TelemetryLike,
    TrackerLike,
    TrackLike,
    VisionSystem,
)

log = get_logger(__name__)

GPSCoord = tuple[float, float]

_DEFAULT_HISTORY_WINDOW = 5  # ASSUMPTION MS1
_FRAME_POLL_INTERVAL_S = 0.005  # busy-poll backoff when the queue has no frame yet


# ---------------------------------------------------------------------------
# Structural contract local to mission.py (output.py's *Like Protocols cover
# everything VisionSystem itself touches; this one additionally covers the
# capture-side frame queue).  `MedianGeolocatorAdapter.locate` deliberately
# stays typed against `output.TrackLike`/`object` rather than getting its own
# widened track Protocol here - see ASSUMPTION MS13.

class FrameQueueLike(Protocol):
    """Structural contract for the frame source `Mission` drives (capture.LatestFrameQueue)."""

    def read(self) -> tuple[np.ndarray, float] | None: ...

    @property
    def exhausted(self) -> bool: ...


# ---------------------------------------------------------------------------
# Gimbal quaternion <-> Euler conversion.  telemetry.py explicitly delegates
# this conversion (and body-vs-earth routing) to the integration layer -
# this file (telemetry.py ASSUMPTION A11).

def _quat_to_euler(
    quat: tuple[float, float, float, float] | None,
) -> tuple[float, float, float]:
    """(w, x, y, z) gimbal quaternion -> (pitch, roll, yaw) radians, ZYX order.

    # ASSUMPTION MS5: suppresses scipy's benign gimbal-lock warning at nadir.
    """
    # ASSUMPTION MS6: quat=None deliberately raises TypeError here, which
    # MedianGeolocatorAdapter.locate() catches alongside ValueError and
    # treats as "this frame contributes nothing" (see diagnostics_report.md C-1).
    w, x, y, z = quat  # type: ignore[misc]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yaw, pitch, roll = Rotation.from_quat([x, y, z, w]).as_euler("ZYX")
    return float(pitch), float(roll), float(yaw)


# ---------------------------------------------------------------------------
# THE CENTERPIECE: median-aggregating geolocator adapter.

class MedianGeolocatorAdapter:
    """Adapts geolocate.py's per-frame pixel->GPS math to `output.GeolocatorLike`.

    Fixes the ~2 km bug (see module header): reports the per-axis median of
    every single-frame estimate seen for a track, not just the latest frame.
    """

    def __init__(
        self, cfg: AppConfig, history_window: int = _DEFAULT_HISTORY_WINDOW
    ) -> None:
        if history_window <= 0:
            raise ValueError(f"history_window must be positive; got {history_window!r}")
        self._K = geo.build_intrinsics(
            cfg.camera.fx, cfg.camera.fy, cfg.camera.cx, cfg.camera.cy
        )
        self._dist_coeffs = cfg.camera.dist_coeffs or None  # ASSUMPTION MS2
        self._history_window = history_window
        self._history: dict[int, list[GPSCoord]] = {}
        self._warned_default_gimbal_frame = False
        self._warned_missing_box = False  # ASSUMPTION MS13
        self._warned_unusable_pose = False  # ASSUMPTION MS13

    def locate(self, track: TrackLike, pose: object) -> GPSCoord | None:
        """Return the per-axis median GPS coordinate for `track`, or None.

        Never raises (ASSUMPTION MS6 / MS13): a telemetry gap (`pose is
        None`) returns `None` without touching history; a ground-miss,
        unusable gimbal reading, a `pose` that isn't `Pose`-shaped, or a
        `track` with no usable `.box` are all treated the same way - the
        frame is skipped and the existing per-track median (or `None`) is
        returned.  `track`/`pose` are typed to match `output.GeolocatorLike`
        exactly (`TrackLike`/`object`); `.box` and the concrete `Pose`
        fields this adapter actually needs are read defensively rather than
        assumed.  # ASSUMPTION MS13
        """
        if pose is None:
            return None  # telemetry gap; VisionSystem falls back to last-known-coord (O5)

        if not isinstance(pose, Pose):
            if not self._warned_unusable_pose:
                log.warning(
                    "pose is not a telemetry.Pose instance (got %s); "
                    "MedianGeolocatorAdapter needs a Pose-shaped pose "
                    "(mission.py ASSUMPTION MS13)", type(pose).__name__,
                )
                self._warned_unusable_pose = True
            history = self._history.get(track.id)
            return geo.aggregate_coords(history) if history else None

        box = getattr(track, "box", None)  # ASSUMPTION MS13: TrackLike doesn't guarantee .box
        if not isinstance(box, tuple) or len(box) != 4:
            if not self._warned_missing_box:
                log.warning(
                    "track %s has no usable .box; MedianGeolocatorAdapter needs "
                    "a .box-bearing track (mission.py ASSUMPTION MS13)", track.id,
                )
                self._warned_missing_box = True
            history = self._history.get(track.id)
            return geo.aggregate_coords(history) if history else None
        x1, y1, x2, y2 = box
        u = (x1 + x2) / 2.0
        v = y2  # ASSUMPTION MS3: bottom-center ground-contact pixel

        gimbal_frame = pose.gimbal_frame
        if gimbal_frame is None:
            gimbal_frame = "body"  # ASSUMPTION MS4
            if not self._warned_default_gimbal_frame:
                log.warning(
                    "pose.gimbal_frame is None; defaulting to body-relative "
                    "gimbal routing (mission.py ASSUMPTION MS4)"
                )
                self._warned_default_gimbal_frame = True

        try:
            gimbal_pitch, gimbal_roll, gimbal_yaw = _quat_to_euler(pose.gimbal_quat)
            if gimbal_frame == "earth":
                ray_cam = geo.pixel_to_ray(u, v, self._K, self._dist_coeffs)
                ray_world = geo.ray_cam_to_world_from_earth_attitude(
                    ray_cam, gimbal_pitch, gimbal_roll, gimbal_yaw,
                )
                north_m, east_m = geo.ray_to_ground(ray_world, pose.alt_agl)
                coord = geo.local_ned_to_gps(north_m, east_m, pose.lat, pose.lon)
            else:
                coord = geo.pixel_to_gps(
                    u, v, pose.lat, pose.lon, pose.alt_agl,
                    gimbal_pitch, gimbal_roll, gimbal_yaw,
                    pose.roll, pose.pitch, pose.yaw,
                    self._K, self._dist_coeffs,
                )
        except (ValueError, TypeError):
            # ASSUMPTION MS6: ray never hit the ground, or gimbal telemetry
            # was unusable this frame - skip the frame, never raise.
            log.debug(
                "geolocation skipped for track %s this frame (ground-miss or "
                "bad telemetry)", track.id, exc_info=True,
            )
            history = self._history.get(track.id)
            return geo.aggregate_coords(history) if history else None

        history = self._history.setdefault(track.id, [])
        history.append(coord)
        del history[: -self._history_window]  # ASSUMPTION MS1: bounded sliding window
        return geo.aggregate_coords(history)


# ---------------------------------------------------------------------------
# Mission state machine.

class MissionState(Enum):
    """Mission-level state.  Scope ends at HANDOFF_COMPLETE (ASSUMPTION MS11)."""

    INIT = "INIT"
    SEARCHING = "SEARCHING"
    TRACKING = "TRACKING"
    COVERAGE_COMPLETE = "COVERAGE_COMPLETE"
    HANDOFF_COMPLETE = "HANDOFF_COMPLETE"
    DONE = "DONE"
    ABORT = "ABORT"


class IllegalTransitionError(Exception):
    """Raised by `transition()` for a state change not in the legal table."""


_LEGAL_TRANSITIONS: dict[MissionState, frozenset[MissionState]] = {
    MissionState.INIT: frozenset({MissionState.SEARCHING, MissionState.ABORT}),
    MissionState.SEARCHING: frozenset({
        MissionState.SEARCHING,
        MissionState.TRACKING,
        MissionState.COVERAGE_COMPLETE,
        MissionState.ABORT,
    }),
    MissionState.TRACKING: frozenset({
        MissionState.TRACKING,
        MissionState.SEARCHING,
        MissionState.COVERAGE_COMPLETE,
        MissionState.ABORT,
    }),
    MissionState.COVERAGE_COMPLETE: frozenset({
        MissionState.HANDOFF_COMPLETE,
        MissionState.ABORT,
    }),
    MissionState.HANDOFF_COMPLETE: frozenset({MissionState.DONE}),
    MissionState.DONE: frozenset(),
    MissionState.ABORT: frozenset(),
}

_TERMINAL_STATES = frozenset(
    {MissionState.HANDOFF_COMPLETE, MissionState.DONE, MissionState.ABORT}
)

_STATUS_TO_STATE: dict[MissionStatus, MissionState] = {
    MissionStatus.TARGET_FOUND: MissionState.TRACKING,
    MissionStatus.NO_TARGET: MissionState.SEARCHING,
    MissionStatus.SEARCHING: MissionState.SEARCHING,
}


def transition(current: MissionState, target: MissionState) -> MissionState:
    """Return `target` if `current -> target` is legal, else raise.

    Raises:
        IllegalTransitionError: if `target` is not in the legal set for `current`.
    """
    if target not in _LEGAL_TRANSITIONS.get(current, frozenset()):
        raise IllegalTransitionError(
            f"illegal mission state transition: {current.name} -> {target.name}"
        )
    return target


# ---------------------------------------------------------------------------
# The orchestrator.

class Mission:
    """Wires `VisionSystem`, runs the capture loop, and drives the state machine.

    Fully dependency-injected: every collaborator is passed in, so this class
    runs with zero hardware in tests (see test_mission.py).  Real hardware
    wiring lives in `main()` below, not here.
    """

    def __init__(
        self,
        cfg: AppConfig,
        *,
        detector: DetectorLike,
        tracker: TrackerLike,
        geolocator: GeolocatorLike,
        telemetry: TelemetryLike,
        frame_queue: FrameQueueLike,
        briefed_target_class_ids: Iterable[int],
        judge_log_path: str | Path | None = None,
        coverage_complete_check: Callable[[], bool] | None = None,
    ) -> None:
        """Wire one mission run.

        Args:
            cfg: Validated application config.
            detector: Raw per-frame detector, used directly - NOT wrapped
                or re-filtered here.  Class-masking now happens inside the
                detector itself (`Detector.target_class_ids`, applied
                before argmax in decode - ASSUMPTION MS9 / detector.py
                ASSUMPTION D13); `Mission` trusts whoever constructed
                `detector` (see `main()` below) already called
                `detector.set_target_classes(briefed_target_class_ids)`.
            tracker: Multi-object tracker (see tracker.Tracker).
            geolocator: Per-track geolocator, e.g. `MedianGeolocatorAdapter`.
            telemetry: Monotonic-time -> pose lookup (see telemetry.TelemetryBuffer).
            frame_queue: Frame source, e.g. `capture.LatestFrameQueue`.
            briefed_target_class_ids: The 4 briefed class ids (ASSUMPTION
                MS8) - required explicitly; no default.  Used here only to
                validate `cfg.mission.target_flag_class` membership - it is
                NOT applied to `detector` (see above).
            judge_log_path: Optional destination for the judge log, written
                at handoff via `VisionSystem.finalize`.
            coverage_complete_check: Optional zero-arg predicate autonomy can
                inject to signal search-pattern-complete (ASSUMPTION MS10).
                Defaults to "the frame source is exhausted".

        Raises:
            ValueError: if `cfg.mission.target_flag_class` is not a member of
                `briefed_target_class_ids` (ASSUMPTION MS8).
        """
        briefed_ids = frozenset(briefed_target_class_ids)
        if cfg.mission.target_flag_class not in briefed_ids:
            raise ValueError(
                f"cfg.mission.target_flag_class={cfg.mission.target_flag_class!r} "
                f"is not among briefed_target_class_ids={sorted(briefed_ids)!r}"
            )

        # Soft, best-effort check only (ASSUMPTION MS9): DetectorLike stays
        # minimal (detect(frame) only, ioutils.output.DetectorLike) so test
        # fakes never need to implement masking - getattr skips them
        # silently.  A real Detector that IS masked but to the wrong set is
        # almost certainly a wiring bug (set_target_classes forgotten/wrong
        # set before constructing Mission), so it's worth a loud warning,
        # never a raise.
        detector_target_ids = getattr(detector, "target_class_ids", None)
        if detector_target_ids is not None and frozenset(detector_target_ids) != briefed_ids:
            log.warning(
                "detector.target_class_ids=%s does not match briefed_target_class_ids=%s; "
                "the detector may not be masked to the briefed set (was set_target_classes "
                "called before constructing Mission?)",
                sorted(detector_target_ids), sorted(briefed_ids),
            )

        self._cfg = cfg
        self._briefed_target_class_ids = briefed_ids
        self._vision = VisionSystem(
            cfg,
            detector=detector,
            tracker=tracker,
            geolocator=geolocator,
            telemetry=telemetry,
        )
        self._frame_queue = frame_queue
        self._judge_log_path = judge_log_path
        self._coverage_complete_check = coverage_complete_check or self._frames_exhausted
        self._frame_idx = 0
        self._degraded_count = 0
        self._final_result: MissionResult | None = None

        self._state = MissionState.INIT
        self._state = transition(self._state, MissionState.SEARCHING)
        log.info(
            "mission initialized: target_flag_class=%s briefed_target_class_ids=%s",
            cfg.mission.target_flag_class, sorted(briefed_ids),
        )

    def _frames_exhausted(self) -> bool:
        """Default coverage-complete predicate.  # TODO: autonomy coverage-complete interface"""
        return self._frame_queue.exhausted

    @property
    def state(self) -> MissionState:
        """Current mission state."""
        return self._state

    @property
    def degraded_count(self) -> int:
        """Total number of DEGRADED vision results seen so far."""
        return self._degraded_count

    def final_result(self) -> MissionResult | None:
        """The final `MissionResult` handed to autonomy, or None before handoff."""
        return self._final_result

    def step(self) -> MissionResult:
        """Process one frame (if available), update state, return the result.

        Never raises: `VisionSystem.process_frame` already never raises
        (output.py ASSUMPTION O1), and a missing frame just returns the
        last-known result instead of blocking indefinitely.
        """
        item = self._frame_queue.read()
        if item is not None:
            frame, _capture_t = item
            result = self._vision.process_frame(frame, self._frame_idx)
            self._frame_idx += 1
            self._apply_result(result)
        else:
            result = self._vision.get_result()
            if not self._frame_queue.exhausted:
                time.sleep(_FRAME_POLL_INTERVAL_S)

        self._maybe_handoff()
        return result

    def run(self) -> MissionResult | None:
        """Run `step()` until the mission reaches a terminal state."""
        while self._state not in _TERMINAL_STATES:
            self.step()
        return self._final_result

    def acknowledge_handoff(self) -> None:
        """Autonomy's acknowledgement that it has taken over.  HANDOFF_COMPLETE -> DONE."""
        self._state = transition(self._state, MissionState.DONE)

    def abort(self, reason: str) -> None:
        """Abort the mission from any non-terminal state."""
        log.error("mission abort: %s", reason)
        self._state = transition(self._state, MissionState.ABORT)

    def _apply_result(self, result: MissionResult) -> None:
        """Drive SEARCHING/TRACKING from a fresh `MissionResult` (ASSUMPTION MS12)."""
        if result.status is MissionStatus.DEGRADED:
            self._degraded_count += 1
            log.warning(
                "DEGRADED vision result at frame_idx=%d (total degraded=%d)",
                self._frame_idx - 1, self._degraded_count,
            )
            return
        target_state = _STATUS_TO_STATE[result.status]
        if self._state in (MissionState.SEARCHING, MissionState.TRACKING):
            self._state = transition(self._state, target_state)

    def _maybe_handoff(self) -> None:
        """Check the coverage-complete trigger and hand off if it fires."""
        if self._state not in (
            MissionState.SEARCHING, MissionState.TRACKING, MissionState.COVERAGE_COMPLETE,
        ):
            return
        if not self._coverage_complete_check():
            return

        if self._state in (MissionState.SEARCHING, MissionState.TRACKING):
            self._state = transition(self._state, MissionState.COVERAGE_COMPLETE)
        self._state = transition(self._state, MissionState.HANDOFF_COMPLETE)

        self._final_result = self._vision.finalize(judge_log_path=self._judge_log_path)
        log.info(
            "mission handoff complete: final status=%s", self._final_result.status.value
        )


# ---------------------------------------------------------------------------
# Real hardware wiring - NOT part of the testable core above.

# GAP-1 (ASSUMPTION MS8): placeholder for the 4 briefed class ids from the
# mission brief document. config.py has no field for these yet; replace with
# the real briefed ids (or a config.py extension) before a live flight.
_BRIEFED_TARGET_CLASS_IDS_PLACEHOLDER: tuple[int, ...] = (0, 1, 2, 3)


def main(config_dir: str = "config") -> None:
    """Real-hardware startup wiring.  Load-bearing order, do not reorder.

    load_config -> configure_logging -> telemetry (buffer fills before frames
    are consumed) -> detector -> tracker -> MedianGeolocatorAdapter ->
    VisionSystem -> capture source + LatestFrameQueue -> Mission -> run.
    """
    from geo.telemetry import TelemetryBuffer
    from vision.capture import FileFrameSource, LatestFrameQueue  # ASSUMPTION: Mac/dev-only import
    from vision.detector import build_detector
    from vision.tracker import Tracker

    cfg = load_config(config_dir)
    configure_logging(level=cfg.log_level, log_dir=cfg.log_dir)

    telemetry_buffer = TelemetryBuffer()
    telemetry_buffer.start("udpin:0.0.0.0:14550")
    time.sleep(2.0)  # let the buffer fill before frames are consumed

    detector = build_detector(cfg, backend="ncnn", param_path="model.param", bin_path="model.bin")
    # GAP-2, resolved (ASSUMPTION MS9): mask the live decode to the briefed
    # set directly on the detector, before argmax - no post-hoc filter.
    detector.set_target_classes(_BRIEFED_TARGET_CLASS_IDS_PLACEHOLDER)
    tracker = Tracker(
        iou_threshold=cfg.mission.iou_threshold,
        min_hits=cfg.mission.min_hits,
        max_age=cfg.mission.max_age,
    )
    geolocator = MedianGeolocatorAdapter(cfg)

    # TODO: swap to RtspFrameSource once SIYI A8 + Raspberry Pi 5 hardware
    # exists (capture.py ASSUMPTION CAP10); FileFrameSource is the honest
    # dev/test stand-in until then.
    capture_source = FileFrameSource("recording.mp4")
    frame_queue = LatestFrameQueue(capture_source)

    mission = Mission(
        cfg,
        detector=detector,
        tracker=tracker,
        geolocator=geolocator,
        telemetry=telemetry_buffer,
        frame_queue=frame_queue,
        briefed_target_class_ids=_BRIEFED_TARGET_CLASS_IDS_PLACEHOLDER,
        judge_log_path="judge_log.json",
    )
    try:
        mission.run()
    finally:
        frame_queue.close()
        telemetry_buffer.stop()


if __name__ == "__main__":
    main()
