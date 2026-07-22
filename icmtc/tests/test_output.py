# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   15/06/2026

"""test_output.py - hardware-free pytest suite for output.py.

Every test uses in-memory fakes for detector/tracker/geolocator/telemetry -
no camera, model runtime, MAVLink link, or filesystem beyond pytest's
`tmp_path` for the judge-log tests.  The nine required cases map to:

  1  target found              6  immutability (frozen)
  2  no target                 7  to_dict is JSON-safe (no numpy types)
  3  searching                 8  judge log: valid file + atomic on failure
  4  degraded / never raises   9  multiple matching targets tolerated
  5  GPS dedup (merge)

Plus a few edge cases the deliverable explicitly calls out as load-bearing:
last-known-coord persistence through a telemetry gap, a track omitted until
it is ever geolocated, and finalize()'s optional judge-log flush.

Run:
    pytest test_output.py -v
"""

from __future__ import annotations

import dataclasses
import json
import os

import numpy as np
import pytest

from core.config import from_dicts
from ioutils.output import (
    FlagReport,
    MissionResult,
    MissionStatus,
    VisionSystem,
    _merge_duplicate_flags,
    write_judge_log,
)


# ---------------------------------------------------------------------------
# Fixtures / fakes

def _make_cfg(*, target_flag_class: int = 3, dedup_merge_distance_m: float = 5.0):
    """A minimal valid AppConfig, built through the real config.py loader."""
    camera = {
        "fx": 1200.0,
        "fy": 1200.0,
        "cx": 640.0,
        "cy": 360.0,
        "image_width": 1280,
        "image_height": 720,
    }
    mission = {
        "search_altitude_m": 80.0,
        "ground_speed_mps": 6.0,
        "target_flag_class": target_flag_class,
        "confidence_threshold": 0.6,
        "dedup_merge_distance_m": dedup_merge_distance_m,
        "iou_threshold": 0.3,
        "min_hits": 3,
        "max_age": 5,
        "lane_overlap": 0.25,
    }
    frames = {"gimbal_reference_frame": "body"}
    return from_dicts(camera, mission, frames)


class FakeTrack:
    """Minimal stand-in for tracker.Track: id, hits, class_votes, committed_class()."""

    def __init__(self, track_id: int, hits: int, class_votes: dict[int, int]):
        self.id = track_id
        self.hits = hits
        self.class_votes = class_votes

    def committed_class(self) -> int | None:
        if not self.class_votes:
            return None
        best = max(self.class_votes.values())
        return min(c for c, v in self.class_votes.items() if v == best)


class FakeDetector:
    """output.py never inspects detections; any placeholder return is fine."""

    def detect(self, frame):
        return []


class FlakyDetector:
    """A detector that can be switched to raise, to exercise the DEGRADED path."""

    def __init__(self):
        self.should_raise = False

    def detect(self, frame):
        if self.should_raise:
            raise RuntimeError("simulated detector failure")
        return []


class FakeTracker:
    """Returns a scripted, already-confirmed track list regardless of input."""

    def __init__(self, confirmed_tracks):
        self._confirmed_tracks = confirmed_tracks

    def update(self, detections, frame_idx):
        return self._confirmed_tracks


class FakeGeolocator:
    """Resolves a coordinate per track id from a scripted mapping, else None."""

    def __init__(self, coords_by_track_id: dict[int, tuple[float, float]]):
        self._coords_by_track_id = coords_by_track_id

    def locate(self, track, pose):
        return self._coords_by_track_id.get(track.id)


class FakeTelemetry:
    """Returns a settable, opaque pose (or None to simulate a telemetry gap)."""

    def __init__(self, pose: object = "fake-pose"):
        self.pose = pose

    def pose_at(self, t):
        return self.pose


def _make_vision(
    *,
    cfg=None,
    detector=None,
    tracker_tracks=(),
    coords_by_track_id=None,
    pose: object = "fake-pose",
) -> VisionSystem:
    return VisionSystem(
        cfg if cfg is not None else _make_cfg(),
        detector=detector if detector is not None else FakeDetector(),
        tracker=FakeTracker(list(tracker_tracks)),
        geolocator=FakeGeolocator(coords_by_track_id or {}),
        telemetry=FakeTelemetry(pose),
    )


# ---------------------------------------------------------------------------
# 1. Target found

def test_target_found_selects_matching_flag():
    vision = _make_vision(
        cfg=_make_cfg(target_flag_class=3),
        tracker_tracks=[FakeTrack(track_id=1, hits=5, class_votes={3: 5})],
        coords_by_track_id={1: (30.0, 31.0)},
    )

    result = vision.process_frame(frame=None, frame_idx=0)

    assert result.status is MissionStatus.TARGET_FOUND
    assert result.target_flag_class == 3
    assert result.target_coord == pytest.approx((30.0, 31.0))
    assert result.target_confidence == pytest.approx(1.0)
    assert len(result.all_flags) == 1


# ---------------------------------------------------------------------------
# 2. No target

def test_no_target_when_confirmed_flags_exist_but_none_match():
    vision = _make_vision(
        cfg=_make_cfg(target_flag_class=3),
        tracker_tracks=[FakeTrack(track_id=1, hits=4, class_votes={7: 4})],
        coords_by_track_id={1: (30.0, 31.0)},
    )

    result = vision.process_frame(frame=None, frame_idx=0)

    assert result.status is MissionStatus.NO_TARGET
    assert result.target_flag_class is None
    assert result.target_coord is None
    assert result.target_confidence is None
    assert len(result.all_flags) == 1
    assert result.all_flags[0].flag_class == 7


# ---------------------------------------------------------------------------
# 3. Searching

def test_searching_when_no_confirmed_flags_yet():
    vision = _make_vision(tracker_tracks=[])

    result = vision.process_frame(frame=None, frame_idx=0)

    assert result.status is MissionStatus.SEARCHING
    assert result.all_flags == ()


def test_get_result_before_any_frame_is_searching():
    vision = _make_vision()

    result = vision.get_result()

    assert result.status is MissionStatus.SEARCHING
    assert result.all_flags == ()


# ---------------------------------------------------------------------------
# 4. Degraded / never raises

def test_degraded_on_internal_failure_never_raises():
    vision = _make_vision(detector=FlakyDetector())
    vision._detector.should_raise = True

    result = vision.process_frame(frame=None, frame_idx=0)

    assert result.status is MissionStatus.DEGRADED
    assert result.target_flag_class is None
    assert result.target_coord is None
    assert result.target_confidence is None


def test_degraded_preserves_last_known_all_flags():
    detector = FlakyDetector()
    vision = _make_vision(
        cfg=_make_cfg(target_flag_class=3),
        detector=detector,
        tracker_tracks=[FakeTrack(track_id=1, hits=5, class_votes={3: 5})],
        coords_by_track_id={1: (30.0, 31.0)},
    )
    first = vision.process_frame(frame=None, frame_idx=0)
    assert first.status is MissionStatus.TARGET_FOUND

    detector.should_raise = True
    second = vision.process_frame(frame=None, frame_idx=1)

    assert second.status is MissionStatus.DEGRADED
    assert second.all_flags == first.all_flags


# ---------------------------------------------------------------------------
# 5. GPS dedup

def test_merge_duplicate_flags_collapses_close_pairs_and_keeps_far_ones_separate():
    close_low_confidence = FlagReport(
        flag_class=3, coord=(30.00000, 31.00000), track_id=1, confidence=0.6
    )
    close_high_confidence = FlagReport(
        flag_class=3, coord=(30.00003, 31.00000), track_id=2, confidence=0.9
    )
    far = FlagReport(flag_class=3, coord=(30.01000, 31.00000), track_id=3, confidence=0.5)

    merged = _merge_duplicate_flags(
        [close_low_confidence, close_high_confidence, far], dist_m=5.0
    )

    assert len(merged) == 2
    assert {flag.track_id for flag in merged} == {2, 3}


def test_merge_duplicate_flags_handles_empty_and_singleton():
    assert _merge_duplicate_flags([], dist_m=5.0) == ()

    single = FlagReport(flag_class=1, coord=(1.0, 1.0), track_id=9, confidence=0.5)
    assert _merge_duplicate_flags([single], dist_m=5.0) == (single,)


def test_merge_duplicate_flags_is_transitive_across_a_chain():
    a = FlagReport(flag_class=1, coord=(30.00000, 31.00000), track_id=1, confidence=0.5)
    b = FlagReport(flag_class=1, coord=(30.00003, 31.00000), track_id=2, confidence=0.9)
    c = FlagReport(flag_class=1, coord=(30.00006, 31.00000), track_id=3, confidence=0.4)

    merged = _merge_duplicate_flags([a, b, c], dist_m=5.0)

    assert len(merged) == 1
    assert merged[0].track_id == 2  # highest confidence in the chain survives


# ---------------------------------------------------------------------------
# 6. Immutability

def test_flag_report_is_frozen():
    flag = FlagReport(flag_class=3, coord=(30.0, 31.0), track_id=1, confidence=0.9)
    with pytest.raises(dataclasses.FrozenInstanceError):
        flag.confidence = 0.5


def test_mission_result_is_frozen():
    result = MissionResult(
        status=MissionStatus.SEARCHING,
        target_flag_class=None,
        target_coord=None,
        target_confidence=None,
        all_flags=(),
        timestamp=0.0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = MissionStatus.DEGRADED


# ---------------------------------------------------------------------------
# 7. to_dict serializable (no numpy types)

def test_to_dict_round_trips_through_json_with_no_numpy_types():
    flag = FlagReport(
        flag_class=3,
        coord=(np.float64(30.0), np.float64(31.0)),
        track_id=1,
        confidence=np.float64(0.9),
    )
    result = MissionResult(
        status=MissionStatus.TARGET_FOUND,
        target_flag_class=3,
        target_coord=(np.float64(30.0), np.float64(31.0)),
        target_confidence=np.float64(0.9),
        all_flags=(flag,),
        timestamp=123.456,
    )

    assert not isinstance(flag.coord[0], np.floating)
    assert not isinstance(flag.confidence, np.floating)

    encoded = json.dumps(result.to_dict())
    decoded = json.loads(encoded)

    assert decoded == {
        "status": "TARGET_FOUND",
        "target_flag_class": 3,
        "target_coord": [30.0, 31.0],
        "target_confidence": 0.9,
        "all_flags": [
            {"flag_class": 3, "coord": [30.0, 31.0], "track_id": 1, "confidence": 0.9}
        ],
        "timestamp": 123.456,
    }


# ---------------------------------------------------------------------------
# 8. Judge log: valid file + atomic on failure

def test_write_judge_log_produces_a_valid_file(tmp_path):
    path = tmp_path / "results" / "judge_log.json"
    result = MissionResult(
        status=MissionStatus.TARGET_FOUND,
        target_flag_class=3,
        target_coord=(30.0, 31.0),
        target_confidence=0.9,
        all_flags=(
            FlagReport(flag_class=3, coord=(30.0, 31.0), track_id=1, confidence=0.9),
            FlagReport(flag_class=5, coord=(30.1, 31.1), track_id=2, confidence=0.7),
        ),
        timestamp=123.0,
    )

    write_judge_log(path, result)

    with path.open() as handle:
        loaded = json.load(handle)
    assert loaded == {
        "flags": [
            {"flag_class": 3, "coord": [30.0, 31.0], "track_id": 1, "confidence": 0.9},
            {"flag_class": 5, "coord": [30.1, 31.1], "track_id": 2, "confidence": 0.7},
        ]
    }


def test_write_judge_log_atomic_failure_leaves_original_untouched(tmp_path, monkeypatch):
    path = tmp_path / "judge_log.json"
    original = MissionResult(
        status=MissionStatus.TARGET_FOUND,
        target_flag_class=3,
        target_coord=(30.0, 31.0),
        target_confidence=0.9,
        all_flags=(FlagReport(flag_class=3, coord=(30.0, 31.0), track_id=1, confidence=0.9),),
        timestamp=1.0,
    )
    write_judge_log(path, original)
    original_bytes = path.read_bytes()

    def boom(_src, _dst):
        raise OSError("simulated crash before rename")

    monkeypatch.setattr(os, "replace", boom)

    updated = dataclasses.replace(
        original,
        all_flags=(FlagReport(flag_class=3, coord=(99.0, 99.0), track_id=2, confidence=0.5),),
    )
    with pytest.raises(OSError):
        write_judge_log(path, updated)

    assert path.read_bytes() == original_bytes
    assert list(tmp_path.glob(".*.tmp")) == []


# ---------------------------------------------------------------------------
# 9. Multiple targets tolerated

def test_multiple_matching_flags_tolerated_without_crashing():
    vision = _make_vision(
        cfg=_make_cfg(target_flag_class=3),
        tracker_tracks=[
            FakeTrack(track_id=1, hits=5, class_votes={3: 5}),
            FakeTrack(track_id=2, hits=4, class_votes={3: 3, 5: 1}),
        ],
        coords_by_track_id={1: (30.0, 31.0), 2: (40.0, 41.0)},  # far apart, not merged
    )

    result = vision.process_frame(frame=None, frame_idx=0)

    assert result.status is MissionStatus.TARGET_FOUND
    assert len(result.all_flags) == 2
    # track 1: confidence 1.0 (5/5) beats track 2: confidence 0.75 (3/4)
    assert result.target_coord == pytest.approx((30.0, 31.0))
    assert result.target_confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Extra: last-known-coord cache and finalize()

def test_last_known_coord_persists_through_a_telemetry_gap():
    vision = _make_vision(
        cfg=_make_cfg(target_flag_class=3),
        tracker_tracks=[FakeTrack(track_id=1, hits=5, class_votes={3: 5})],
        coords_by_track_id={1: (30.0, 31.0)},
        pose="ok",
    )
    first = vision.process_frame(frame=None, frame_idx=0)
    assert first.target_coord == pytest.approx((30.0, 31.0))

    vision._telemetry.pose = None  # ASSUMPTION A8 (telemetry.py): out of range -> None
    second = vision.process_frame(frame=None, frame_idx=1)

    assert second.status is MissionStatus.TARGET_FOUND
    assert second.target_coord == pytest.approx((30.0, 31.0))


def test_track_never_geolocated_is_omitted_from_all_flags():
    vision = _make_vision(
        cfg=_make_cfg(target_flag_class=3),
        tracker_tracks=[FakeTrack(track_id=1, hits=5, class_votes={3: 5})],
        coords_by_track_id={},  # never resolves a coordinate
    )

    result = vision.process_frame(frame=None, frame_idx=0)

    assert result.status is MissionStatus.SEARCHING
    assert result.all_flags == ()


def test_finalize_writes_judge_log_and_returns_final_result(tmp_path):
    vision = _make_vision(
        cfg=_make_cfg(target_flag_class=3),
        tracker_tracks=[FakeTrack(track_id=1, hits=5, class_votes={3: 5})],
        coords_by_track_id={1: (30.0, 31.0)},
    )
    vision.process_frame(frame=None, frame_idx=0)

    path = tmp_path / "judge_log.json"
    result = vision.finalize(judge_log_path=path)

    assert result.status is MissionStatus.TARGET_FOUND
    with path.open() as handle:
        loaded = json.load(handle)
    assert loaded["flags"][0]["flag_class"] == 3


def test_finalize_without_path_does_not_write_anything(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    vision = _make_vision()

    result = vision.finalize()

    assert result.status is MissionStatus.SEARCHING
    assert list(tmp_path.iterdir()) == []
