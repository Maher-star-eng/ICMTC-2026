# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   08/06/2026

"""Tests for tracker.py.

All inputs are synthetic per-frame detections.  No camera, model, telemetry, or
geo-location hardware is involved.
"""

from __future__ import annotations

import pytest

from vision.tracker import Detection, Tracker


def make_detection(
    x: float,
    y: float,
    *,
    class_id: int = 1,
    score: float = 0.9,
    coord: tuple[float, float] | None = None,
    size: float = 20.0,
) -> Detection:
    """Create a square synthetic detection box."""
    return Detection(
        box=(x, y, x + size, y + size),
        class_id=class_id,
        score=score,
        coord=coord,
    )


def test_single_flag_many_frames_commits_one_track_class_and_median_coord():
    """A drifting single flag confirms once and aggregates class + coordinate."""
    tracker = Tracker(iou_threshold=0.3, min_hits=3, max_age=5)
    coords = [
        (30.00000, 31.00000),
        (30.00010, 31.00010),
        (30.00020, 31.00020),
        (30.00030, 31.00030),
        (30.00040, 31.00040),
    ]

    for frame_idx, coord in enumerate(coords):
        tracks = tracker.update(
            [
                make_detection(
                    100.0 + frame_idx,
                    200.0 + frame_idx,
                    class_id=4,
                    coord=coord,
                )
            ],
            frame_idx,
        )

    assert len(tracks) == 1
    track = tracks[0]
    assert track.id == 1
    assert track.hits == len(coords)
    assert track.confirmed
    assert track.committed_class() == 4
    assert track.committed_coord() == pytest.approx((30.00020, 31.00020))


def test_two_non_overlapping_flags_keep_distinct_stable_ids_when_order_changes():
    """Two separated flags do not cross-assign, even when detector order flips."""
    tracker = Tracker(iou_threshold=0.3, min_hits=1, max_age=5)

    first_frame_tracks = tracker.update(
        [
            make_detection(10.0, 10.0, class_id=10),
            make_detection(200.0, 10.0, class_id=20),
        ],
        0,
    )
    initial_ids_by_class = {
        track.committed_class(): track.id
        for track in first_frame_tracks
    }

    second_frame_tracks = tracker.update(
        [
            make_detection(202.0, 12.0, class_id=20),
            make_detection(12.0, 11.0, class_id=10),
        ],
        1,
    )
    ids_by_class = {
        track.committed_class(): track.id
        for track in second_frame_tracks
    }

    assert ids_by_class == initial_ids_by_class
    assert set(ids_by_class) == {10, 20}
    assert len(set(ids_by_class.values())) == 2


def test_one_frame_false_positive_never_confirms_or_reports():
    """A one-frame stray detection is retired unconfirmed and never reported."""
    tracker = Tracker(iou_threshold=0.3, min_hits=2, max_age=1)

    assert tracker.update([make_detection(50.0, 50.0, class_id=99)], 0) == []
    assert tracker.update([], 1) == []
    assert tracker.update([], 2) == []
    assert tracker.confirmed_flags() == []
    assert tracker.active_tracks() == []


def test_occlusion_shorter_than_max_age_reuses_same_track_id():
    """A confirmed flag can miss a few frames, reappear, and keep its id."""
    tracker = Tracker(iou_threshold=0.3, min_hits=2, max_age=5)

    tracker.update([make_detection(100.0, 100.0, class_id=3)], 0)
    tracks = tracker.update([make_detection(101.0, 101.0, class_id=3)], 1)
    original_id = tracks[0].id

    tracker.update([], 2)
    tracker.update([], 3)
    tracks = tracker.update([make_detection(102.0, 102.0, class_id=3)], 4)

    assert len(tracks) == 1
    assert tracks[0].id == original_id
    assert tracks[0].hits == 3
    assert not tracks[0].retired


def test_class_flicker_commits_majority_vote():
    """A few class misfires are outvoted by the dominant class."""
    tracker = Tracker(iou_threshold=0.3, min_hits=1, max_age=5)

    for frame_idx, class_id in enumerate([7, 7, 8, 7, 8]):
        tracks = tracker.update(
            [make_detection(300.0 + frame_idx, 100.0, class_id=class_id)],
            frame_idx,
        )

    assert len(tracks) == 1
    assert tracks[0].class_votes == {7: 3, 8: 2}
    assert tracks[0].committed_class() == 7


def test_coordinate_outlier_does_not_move_median_commit():
    """The per-axis median stays near the cluster despite one wild coordinate."""
    tracker = Tracker(iou_threshold=0.3, min_hits=1, max_age=5)
    coords = [
        (30.00000, 31.00000),
        (30.00001, 31.00002),
        (35.00000, 40.00000),
        (30.00002, 31.00001),
        (30.00003, 31.00003),
    ]

    for frame_idx, coord in enumerate(coords):
        tracks = tracker.update(
            [
                make_detection(
                    400.0 + frame_idx,
                    150.0,
                    class_id=5,
                    coord=coord,
                )
            ],
            frame_idx,
        )

    assert len(tracks) == 1
    assert tracks[0].committed_coord() == pytest.approx((30.00002, 31.00002))


def test_age_out_retires_track_after_more_than_max_age_frames():
    """A track is retired only after the documented > max_age boundary."""
    tracker = Tracker(iou_threshold=0.3, min_hits=1, max_age=1)

    first_tracks = tracker.update([make_detection(80.0, 80.0, class_id=2)], 0)
    assert len(first_tracks) == 1
    assert not first_tracks[0].retired

    tracker.update([], 1)
    assert tracker.active_tracks()[0].id == first_tracks[0].id

    tracker.update([], 2)
    confirmed = tracker.confirmed_flags()
    assert confirmed[0].id == first_tracks[0].id
    assert confirmed[0].retired
    assert tracker.active_tracks() == []


def test_retired_ids_are_never_reused_for_later_detection_in_same_spot():
    """After age-out, even an identical later box starts a new stable id."""
    tracker = Tracker(iou_threshold=0.3, min_hits=1, max_age=0)

    tracker.update([make_detection(500.0, 500.0, class_id=1)], 0)
    tracker.update([], 1)
    tracks = tracker.update([make_detection(500.0, 500.0, class_id=1)], 2)

    assert [track.id for track in tracks] == [1, 2]
    assert tracks[0].retired
    assert not tracks[1].retired


def test_same_input_sequence_is_deterministic():
    """Running the same detections through fresh trackers gives identical output."""
    frames = [
        [
            make_detection(10.0, 10.0, class_id=1, coord=(30.0, 31.0)),
            make_detection(100.0, 10.0, class_id=2, coord=(30.1, 31.1)),
        ],
        [
            make_detection(12.0, 11.0, class_id=1, coord=(30.0, 31.0001)),
            make_detection(102.0, 11.0, class_id=3, coord=(30.1, 31.1001)),
        ],
        [],
        [
            make_detection(13.0, 12.0, class_id=1, coord=(30.0001, 31.0)),
            make_detection(103.0, 12.0, class_id=2, coord=(30.1001, 31.1)),
        ],
    ]

    def run_once() -> tuple[tuple[int, int | None, tuple[float, float] | None, int, bool], ...]:
        tracker = Tracker(iou_threshold=0.3, min_hits=2, max_age=3)
        for frame_idx, detections in enumerate(frames):
            tracker.update(detections, frame_idx)
        return tuple(
            (
                track.id,
                track.committed_class(),
                track.committed_coord(),
                track.hits,
                track.retired,
            )
            for track in tracker.confirmed_flags()
        )

    assert run_once() == run_once()
