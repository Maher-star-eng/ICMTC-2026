# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   19/06/2026

"""test_mission.py - hardware-free pytest suite for mission.py.

Every test uses in-memory fakes for detector/tracker/geolocator/telemetry/
frame-source - no camera, model runtime, MAVLink link, or real background
threading (see `SynchronousFrameQueue` below, ASSUMPTION MS5 in mission.py).
The eight required cases map to:

  1  median aggregation (the ~2 km bug regression test)
  2  telemetry gap -> locate() returns None
  3  ground-miss frame -> prior median preserved
  4  body vs earth routing (plus a None-defaults-to-body sub-case)
  5  state machine: legal / illegal transitions, scope excludes autonomy states
  6  full loop on a fake source -> TARGET_FOUND + judge log
  7  degraded tolerance -> no crash
  8  coverage-complete -> handoff, final result exposed

Run:
    pytest test_mission.py -v
"""

from __future__ import annotations

import json
import logging
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import geo.geolocate as geo
from core import mission
from core.config import from_dicts
from geo.telemetry import Pose
from ioutils.output import MissionStatus
from vision.tracker import Detection, Tracker

# ---------------------------------------------------------------------------
# Shared fixtures / fakes

_CAMERA = {
    "fx": 1000.0, "fy": 1000.0, "cx": 640.0, "cy": 360.0,
    "image_width": 1280, "image_height": 720,
}
_K = geo.build_intrinsics(_CAMERA["fx"], _CAMERA["fy"], _CAMERA["cx"], _CAMERA["cy"])


def _make_cfg(*, target_flag_class=3, min_hits=2, max_age=5, iou_threshold=0.3):
    mission_section = {
        "search_altitude_m": 50.0,
        "ground_speed_mps": 5.0,
        "target_flag_class": target_flag_class,
        "confidence_threshold": 0.5,
        "iou_threshold": iou_threshold,
        "min_hits": min_hits,
        "max_age": max_age,
        "lane_overlap": 0.2,
    }
    frames = {"gimbal_reference_frame": "body"}
    return from_dicts(_CAMERA, mission_section, frames)


def _quat_from_euler(pitch: float, roll: float, yaw: float) -> tuple[float, float, float, float]:
    """(pitch, roll, yaw) radians -> (w, x, y, z), the exact inverse of mission._quat_to_euler."""
    xyzw = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_quat()
    return (float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2]))


NADIR_QUAT = _quat_from_euler(pitch=-math.pi / 2, roll=0.0, yaw=0.0)
LEVEL_QUAT = _quat_from_euler(pitch=0.0, roll=0.0, yaw=0.0)


def make_pose(
    *,
    gimbal_quat=NADIR_QUAT,
    gimbal_frame="body",
    lat=30.0,
    lon=31.0,
    alt_agl=50.0,
    roll=0.0,
    pitch=0.0,
    yaw=0.0,
) -> Pose:
    return Pose(
        t=0.0, lat=lat, lon=lon, alt_agl=alt_agl,
        roll=roll, pitch=pitch, yaw=yaw,
        gimbal_quat=gimbal_quat, gimbal_frame=gimbal_frame,
        fix_type=3, h_acc_m=1.0,
    )


class _FakeBoxedTrack:
    """Structural fake fully satisfying output.TrackLike, plus `.box` (as a real
    Track also carries).  MedianGeolocatorAdapter still reads `.box`
    defensively (mission.py ASSUMPTION MS13) rather than assuming it - see
    FakeTrackLikeNoBox below for a TrackLike fake that lacks it.
    """

    def __init__(self, track_id: int, box: tuple[float, float, float, float]):
        self.id = track_id
        self.box = box
        self.hits = 3
        self.class_votes = {3: 3}

    def committed_class(self) -> int | None:
        return 3


def make_track(track_id: int, box: tuple[float, float, float, float]) -> _FakeBoxedTrack:
    return _FakeBoxedTrack(track_id, box)


class FakeTrackLikeNoBox:
    """Minimal TrackLike-shaped fake: id/hits/class_votes/committed_class(), no `.box`.

    Mirrors ioutils.output's own test fake of the same shape (test_output.py's
    FakeTrack) - injecting this into MedianGeolocatorAdapter.locate is exactly
    the substitution the TrackLike/GeolocatorLike pyright check exists to
    guard: the adapter must degrade gracefully, not raise AttributeError
    (mission.py ASSUMPTION MS13).
    """

    def __init__(self, track_id: int):
        self.id = track_id
        self.hits = 3
        self.class_votes = {3: 3}

    def committed_class(self) -> int | None:
        return 3


def _box_for_bottom_center(u: float, v: float, *, half: float = 5.0) -> tuple[float, float, float, float]:
    """A box whose bottom-center pixel is exactly (u, v) (geolocate.py convention #13)."""
    return (u - half, v - 2 * half, u + half, v)


# ---------------------------------------------------------------------------
# 1. Median aggregation - the regression test for the ~2 km bug

def test_median_geolocator_returns_median_not_the_outlier_last_frame():
    cfg = _make_cfg()
    adapter = mission.MedianGeolocatorAdapter(cfg)
    pose = make_pose(gimbal_quat=NADIR_QUAT)
    track = make_track(1, box=_box_for_bottom_center(640.0, 400.0))

    cluster_pixels = [(620.0, 400.0), (630.0, 405.0), (640.0, 400.0), (650.0, 395.0), (660.0, 400.0)]
    cluster_coords = []
    for u, v in cluster_pixels:
        track.box = _box_for_bottom_center(u, v)
        adapter.locate(track, pose)
        gimbal_pitch, gimbal_roll, gimbal_yaw = mission._quat_to_euler(pose.gimbal_quat)
        cluster_coords.append(
            geo.pixel_to_gps(
                u, v, pose.lat, pose.lon, pose.alt_agl,
                gimbal_pitch, gimbal_roll, gimbal_yaw,
                pose.roll, pose.pitch, pose.yaw, _K,
            )
        )
    expected_median = geo.aggregate_coords(cluster_coords)

    # A single wild-outlier frame (e.g. a bad gimbal reading far off nadir).
    outlier_u, outlier_v = 50.0, 700.0
    track.box = _box_for_bottom_center(outlier_u, outlier_v)
    result = adapter.locate(track, pose)

    gimbal_pitch, gimbal_roll, gimbal_yaw = mission._quat_to_euler(pose.gimbal_quat)
    outlier_coord = geo.pixel_to_gps(
        outlier_u, outlier_v, pose.lat, pose.lon, pose.alt_agl,
        gimbal_pitch, gimbal_roll, gimbal_yaw,
        pose.roll, pose.pitch, pose.yaw, _K,
    )
    outlier_distance_deg = math.hypot(
        outlier_coord[0] - expected_median[0], outlier_coord[1] - expected_median[1]
    )

    assert result == pytest.approx(expected_median, abs=1e-9)
    assert outlier_distance_deg > 1e-4  # the outlier really is far away in GPS-degree terms
    assert result != pytest.approx(outlier_coord, abs=1e-6)


def test_median_geolocator_bounds_history_to_the_configured_window():
    cfg = _make_cfg()
    adapter = mission.MedianGeolocatorAdapter(cfg, history_window=3)
    pose = make_pose(gimbal_quat=NADIR_QUAT)
    track = make_track(1, box=_box_for_bottom_center(640.0, 400.0))

    for _ in range(10):
        adapter.locate(track, pose)

    assert len(adapter._history[1]) == 3


# ---------------------------------------------------------------------------
# 2. Telemetry gap

def test_locate_returns_none_on_telemetry_gap():
    cfg = _make_cfg()
    adapter = mission.MedianGeolocatorAdapter(cfg)
    track = make_track(1, box=_box_for_bottom_center(640.0, 400.0))

    assert adapter.locate(track, None) is None
    assert adapter._history.get(1, []) == []


# ---------------------------------------------------------------------------
# 3. Ground-miss frame

def test_ground_miss_frame_preserves_prior_median_and_never_raises():
    cfg = _make_cfg()
    adapter = mission.MedianGeolocatorAdapter(cfg)
    track = make_track(1, box=_box_for_bottom_center(640.0, 410.0))

    good_pose = make_pose(gimbal_quat=NADIR_QUAT)
    first = adapter.locate(track, good_pose)
    assert first is not None

    # A pixel well above the principal point, combined with a level gimbal and
    # a level drone, points the ray above the horizon -> ray_to_ground raises
    # ValueError internally (geolocate.py convention #8).
    bad_pose = make_pose(gimbal_quat=LEVEL_QUAT)
    track.box = _box_for_bottom_center(640.0, 100.0)
    second = adapter.locate(track, bad_pose)

    assert second == pytest.approx(first, abs=1e-9)


# ---------------------------------------------------------------------------
# 3b. Protocol-narrowness fix: box-less TrackLike fake / non-Pose pose

def test_locate_degrades_gracefully_for_a_box_less_tracklike_fake():
    cfg = _make_cfg()
    adapter = mission.MedianGeolocatorAdapter(cfg)
    track = FakeTrackLikeNoBox(track_id=1)
    pose = make_pose()

    result = adapter.locate(track, pose)

    assert result is None
    assert adapter._history.get(1, []) == []


def test_locate_falls_back_to_prior_median_when_track_loses_its_box():
    cfg = _make_cfg()
    adapter = mission.MedianGeolocatorAdapter(cfg)
    pose = make_pose()

    boxed_track = make_track(1, box=_box_for_bottom_center(640.0, 410.0))
    first = adapter.locate(boxed_track, pose)
    assert first is not None

    boxless_track = FakeTrackLikeNoBox(track_id=1)
    second = adapter.locate(boxless_track, pose)

    assert second == pytest.approx(first, abs=1e-9)


def test_missing_box_logs_a_warning_once(caplog):
    cfg = _make_cfg()
    adapter = mission.MedianGeolocatorAdapter(cfg)
    track = FakeTrackLikeNoBox(track_id=1)
    pose = make_pose()

    with caplog.at_level(logging.WARNING, logger="mission"):
        adapter.locate(track, pose)
        adapter.locate(track, pose)

    warnings = [r for r in caplog.records if "needs a .box-bearing track" in r.message]
    assert len(warnings) == 1


def test_locate_degrades_gracefully_for_a_non_pose_object():
    cfg = _make_cfg()
    adapter = mission.MedianGeolocatorAdapter(cfg)
    track = make_track(1, box=_box_for_bottom_center(640.0, 410.0))

    result = adapter.locate(track, "fake-pose")

    assert result is None
    assert adapter._history.get(1, []) == []


# ---------------------------------------------------------------------------
# 4. Body vs earth routing

def test_body_gimbal_frame_routes_through_pixel_to_gps(monkeypatch):
    cfg = _make_cfg()
    adapter = mission.MedianGeolocatorAdapter(cfg)
    track = make_track(1, box=_box_for_bottom_center(640.0, 410.0))
    pose = make_pose(gimbal_frame="body")

    calls = {"body": 0, "earth": 0}
    real_body = mission.geo.pixel_to_gps
    real_earth = mission.geo.ray_cam_to_world_from_earth_attitude
    monkeypatch.setattr(
        mission.geo, "pixel_to_gps",
        lambda *a, **kw: (calls.__setitem__("body", calls["body"] + 1), real_body(*a, **kw))[1],
    )
    monkeypatch.setattr(
        mission.geo, "ray_cam_to_world_from_earth_attitude",
        lambda *a, **kw: (calls.__setitem__("earth", calls["earth"] + 1), real_earth(*a, **kw))[1],
    )

    result = adapter.locate(track, pose)

    assert result is not None
    assert calls == {"body": 1, "earth": 0}


def test_earth_gimbal_frame_routes_through_earth_relative_chain(monkeypatch):
    cfg = _make_cfg()
    adapter = mission.MedianGeolocatorAdapter(cfg)
    track = make_track(1, box=_box_for_bottom_center(640.0, 410.0))
    pose = make_pose(gimbal_frame="earth", gimbal_quat=NADIR_QUAT)

    calls = {"body": 0, "earth": 0}
    real_body = mission.geo.pixel_to_gps
    real_earth = mission.geo.ray_cam_to_world_from_earth_attitude
    monkeypatch.setattr(
        mission.geo, "pixel_to_gps",
        lambda *a, **kw: (calls.__setitem__("body", calls["body"] + 1), real_body(*a, **kw))[1],
    )
    monkeypatch.setattr(
        mission.geo, "ray_cam_to_world_from_earth_attitude",
        lambda *a, **kw: (calls.__setitem__("earth", calls["earth"] + 1), real_earth(*a, **kw))[1],
    )

    result = adapter.locate(track, pose)

    assert result is not None
    assert calls == {"body": 0, "earth": 1}


def test_none_gimbal_frame_defaults_to_body_and_warns_once(caplog):
    cfg = _make_cfg()
    adapter = mission.MedianGeolocatorAdapter(cfg)
    track = make_track(1, box=_box_for_bottom_center(640.0, 410.0))
    pose = make_pose(gimbal_frame=None)

    with caplog.at_level(logging.WARNING, logger="mission"):
        adapter.locate(track, pose)
        adapter.locate(track, pose)

    warnings = [r for r in caplog.records if "defaulting to body-relative" in r.message]
    assert len(warnings) == 1


# ---------------------------------------------------------------------------
# 5. State machine

def test_state_machine_legal_transitions_succeed():
    S = mission.MissionState
    assert mission.transition(S.INIT, S.SEARCHING) is S.SEARCHING
    assert mission.transition(S.SEARCHING, S.TRACKING) is S.TRACKING
    assert mission.transition(S.TRACKING, S.SEARCHING) is S.SEARCHING
    assert mission.transition(S.TRACKING, S.TRACKING) is S.TRACKING
    assert mission.transition(S.SEARCHING, S.COVERAGE_COMPLETE) is S.COVERAGE_COMPLETE
    assert mission.transition(S.COVERAGE_COMPLETE, S.HANDOFF_COMPLETE) is S.HANDOFF_COMPLETE
    assert mission.transition(S.HANDOFF_COMPLETE, S.DONE) is S.DONE
    assert mission.transition(S.SEARCHING, S.ABORT) is S.ABORT


def test_state_machine_illegal_transitions_are_rejected():
    S = mission.MissionState
    with pytest.raises(mission.IllegalTransitionError):
        mission.transition(S.INIT, S.TRACKING)
    with pytest.raises(mission.IllegalTransitionError):
        mission.transition(S.HANDOFF_COMPLETE, S.SEARCHING)
    with pytest.raises(mission.IllegalTransitionError):
        mission.transition(S.DONE, S.SEARCHING)
    with pytest.raises(mission.IllegalTransitionError):
        mission.transition(S.ABORT, S.SEARCHING)


def test_state_machine_scope_excludes_autonomy_states():
    names = {member.name for member in mission.MissionState}
    assert names == {
        "INIT", "SEARCHING", "TRACKING", "COVERAGE_COMPLETE",
        "HANDOFF_COMPLETE", "DONE", "ABORT",
    }
    assert not names & {"TARGETING", "DROPPING", "RETURNING"}


# ---------------------------------------------------------------------------
# Fakes for the full Mission loop (tests 6-8)

class ListFrameSource:
    """A FileFrameSource-shaped fake: read() -> (frame, t) | None, close()."""

    def __init__(self, n_frames: int):
        self._remaining = n_frames
        self.closed = False

    def read(self):
        if self.closed or self._remaining <= 0:
            return None
        self._remaining -= 1
        return np.zeros((4, 4, 3), dtype=np.uint8), 0.0

    def close(self):
        self.closed = True


class SynchronousFrameQueue:
    """Deterministic, non-threaded stand-in for capture.LatestFrameQueue.

    Satisfies mission.FrameQueueLike without a background thread, so tests
    never race the real LatestFrameQueue's single-slot "latest wins" drop
    behavior (capture.py ASSUMPTION CAP8).
    """

    def __init__(self, source):
        self._source = source
        self._exhausted = False

    def read(self):
        if self._exhausted:
            return None
        result = self._source.read()
        if result is None:
            self._exhausted = True
            return None
        return result

    @property
    def exhausted(self) -> bool:
        return self._exhausted


class FakeDetector:
    def __init__(self, detections):
        self._detections = list(detections)

    def detect(self, frame):
        return list(self._detections)


class FakeMaskedDetector(FakeDetector):
    """A FakeDetector that also exposes target_class_ids, like a real
    vision.detector.Detector - used only to exercise Mission's soft
    defensive wiring check (ASSUMPTION MS9); plain FakeDetector has no
    such attribute and is skipped via getattr, by design.
    """

    def __init__(self, detections, target_class_ids):
        super().__init__(detections)
        self.target_class_ids = frozenset(target_class_ids)


class FlakyFakeDetector:
    def __init__(self, detections, fail_on_calls: set[int]):
        self._detections = list(detections)
        self._fail_on_calls = set(fail_on_calls)
        self._calls = 0

    def detect(self, frame):
        call_index = self._calls
        self._calls += 1
        if call_index in self._fail_on_calls:
            raise RuntimeError("simulated detector failure")
        return list(self._detections)


class FakeGeolocator:
    def __init__(self, coord=(30.0, 31.0)):
        self._coord = coord

    def locate(self, track, pose):
        return self._coord


class FakeTelemetry:
    def pose_at(self, t):
        return "fake-pose"


def _make_mission(
    *,
    detector,
    frame_count=6,
    min_hits=2,
    target_flag_class=3,
    briefed_target_class_ids=(1, 2, 3, 4),
    judge_log_path=None,
    coverage_complete_check=None,
) -> mission.Mission:
    cfg = _make_cfg(target_flag_class=target_flag_class, min_hits=min_hits)
    tracker = Tracker(
        iou_threshold=cfg.mission.iou_threshold,
        min_hits=cfg.mission.min_hits,
        max_age=cfg.mission.max_age,
    )
    frame_queue = SynchronousFrameQueue(ListFrameSource(frame_count))
    return mission.Mission(
        cfg,
        detector=detector,
        tracker=tracker,
        geolocator=FakeGeolocator(),
        telemetry=FakeTelemetry(),
        frame_queue=frame_queue,
        briefed_target_class_ids=briefed_target_class_ids,
        judge_log_path=judge_log_path,
        coverage_complete_check=coverage_complete_check,
    )


# ---------------------------------------------------------------------------
# 6. Full loop on a fake source

def test_full_loop_confirms_target_and_writes_judge_log(tmp_path):
    detection = Detection(box=(10.0, 10.0, 50.0, 50.0), class_id=3, score=0.9, coord=None)
    judge_log_path = tmp_path / "judge_log.json"

    m = _make_mission(detector=FakeDetector([detection]), frame_count=5, judge_log_path=judge_log_path)
    m.run()

    assert m.state is mission.MissionState.HANDOFF_COMPLETE
    final = m.final_result()
    assert final is not None
    assert final.status is MissionStatus.TARGET_FOUND
    assert final.target_flag_class == 3

    with judge_log_path.open() as handle:
        payload = json.load(handle)
    assert payload["flags"][0]["flag_class"] == 3


# ---------------------------------------------------------------------------
# 7. Degraded tolerance

def test_degraded_frames_do_not_crash_the_loop_and_recover():
    detection = Detection(box=(10.0, 10.0, 50.0, 50.0), class_id=3, score=0.9, coord=None)
    detector = FlakyFakeDetector([detection], fail_on_calls={0, 1})
    m = _make_mission(detector=detector, frame_count=6)

    first = m.step()
    second = m.step()
    assert first.status is MissionStatus.DEGRADED
    assert second.status is MissionStatus.DEGRADED
    assert m.state is mission.MissionState.SEARCHING  # DEGRADED never moves state

    m.run()

    assert m.state is mission.MissionState.HANDOFF_COMPLETE
    final = m.final_result()
    assert final is not None
    assert final.status is MissionStatus.TARGET_FOUND


# ---------------------------------------------------------------------------
# 8. Coverage-complete -> handoff

def test_coverage_complete_triggers_handoff_and_exposes_final_result():
    m = _make_mission(detector=FakeDetector([]), frame_count=3)

    assert m.state is mission.MissionState.SEARCHING

    m.run()

    assert m.state is mission.MissionState.HANDOFF_COMPLETE
    final = m.final_result()
    assert final is not None
    assert final.status is MissionStatus.SEARCHING

    m.acknowledge_handoff()
    assert m.state is mission.MissionState.DONE


# ---------------------------------------------------------------------------
# Bonus: GAP-1 / GAP-2 wiring, directly exercising the documented behavior

def test_target_flag_class_must_be_in_briefed_ids():
    with pytest.raises(ValueError):
        _make_mission(
            detector=FakeDetector([]),
            target_flag_class=99,
            briefed_target_class_ids=(1, 2, 3, 4),
        )


def test_mission_passes_detector_through_unwrapped():
    """GAP-2 resolved (ASSUMPTION MS9): Mission no longer post-hoc filters.

    Class-masking now lives inside the detector itself (see
    vision.detector.Detector.target_class_ids / set_target_classes,
    ASSUMPTION D13) - `_ClassMaskedDetector` is gone.  `Mission` hands its
    injected `detector` straight to `VisionSystem`, unwrapped and
    untouched: no filtering object sits between them any more.
    """
    detector = FakeDetector([])

    m = _make_mission(detector=detector, frame_count=1)

    assert m._vision._detector is detector


def test_mismatched_detector_target_classes_logs_a_warning(caplog):
    """ASSUMPTION MS9 defensive check: a detector masked to the WRONG set
    logs a loud warning at Mission construction time - a real Detector
    that's masked but to the wrong ids is almost certainly a wiring bug
    (set_target_classes forgotten/wrong before constructing Mission).  This
    is soft and best-effort: it must warn, never raise, and must not
    require DetectorLike fakes in general to grow this attribute.
    """
    detector = FakeMaskedDetector([], target_class_ids=frozenset({9, 10}))

    with caplog.at_level(logging.WARNING):
        _make_mission(detector=detector, briefed_target_class_ids=(1, 2, 3, 4))

    assert "does not match briefed_target_class_ids" in caplog.text
    assert "[9, 10]" in caplog.text
    assert "[1, 2, 3, 4]" in caplog.text


def test_matching_detector_target_classes_does_not_warn(caplog):
    """The mirror case: target_class_ids already equal to the briefed set
    -> no warning, confirming the check compares content, not just presence.
    """
    detector = FakeMaskedDetector([], target_class_ids=frozenset({1, 2, 3, 4}))

    with caplog.at_level(logging.WARNING):
        _make_mission(detector=detector, briefed_target_class_ids=(1, 2, 3, 4))

    assert "does not match briefed_target_class_ids" not in caplog.text


def test_detector_without_target_class_ids_does_not_warn(caplog):
    """A plain FakeDetector (no target_class_ids attribute at all, like
    every other fake in this test suite) is silently skipped via getattr -
    no warning, no AttributeError, no Protocol change required.
    """
    detector = FakeDetector([])
    assert not hasattr(detector, "target_class_ids")

    with caplog.at_level(logging.WARNING):
        _make_mission(detector=detector, briefed_target_class_ids=(1, 2, 3, 4))

    assert "does not match briefed_target_class_ids" not in caplog.text
