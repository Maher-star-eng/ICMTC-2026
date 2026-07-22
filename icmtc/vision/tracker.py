# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   08/06/2026

"""tracker.py - multi-object flag tracker for ICMTC 2026 Mission 2.

Pure computation module: detector boxes + class labels + optional GPS
coordinates are associated across frames into stable per-flag tracks.  No
hardware access, no model calls, no I/O, and no threading.

================================================================================
ASSUMPTIONS (collected here; also marked inline with # ASSUMPTION: <tag>)
================================================================================

T1  PREDICTION MODEL
    Tracks use a constant-position prediction: the predicted box for
    association is simply the last observed box.  Flags are static in the
    world, and any apparent image motion is handled by the IoU gate tolerating
    modest frame-to-frame drift.  The `_predicted_box` method is intentionally
    isolated so a Kalman or camera-motion model can be added later.

T2  ASSOCIATION METRIC
    Association uses bounding-box IoU only.  A detection may match a track when
    IoU >= iou_threshold.  Default iou_threshold is 0.3.

T3  ASSIGNMENT METHOD
    Assignment is greedy, not Hungarian: all gated track/detection pairs are
    sorted by descending IoU, then by ascending track id and detection index for
    deterministic tie-breaking.  The mission expects four sparse, separated
    flags, so this is auditable and sufficient.

T4  DETECTION SCORE
    Detection.score is validated as finite but is not used by this tracker.
    Detector confidence thresholding is assumed to happen upstream; class
    identity is aggregated by per-frame votes here.

T5  CLASS COMMIT
    The committed class is the mode of per-frame class_id votes.  Ties are
    broken by the lowest class_id for deterministic behavior.

T6  COORDINATE COMMIT
    The committed GPS coordinate is the per-axis median of all supplied
    (lat, lon) observations for that track.  A track with no supplied
    coordinates returns None.

T7  CONFIRMATION
    A track becomes confirmed once hits >= min_hits.  Default min_hits is 3.
    Confirmed status is sticky and is not removed when a track later retires.

T8  AGE-OUT BOUNDARY
    A track is retired before association when frame_idx - last_seen_frame >
    max_age.  Therefore a gap exactly equal to max_age is still eligible for
    matching; a larger gap forces a new id.  Default max_age is 5.

T9  FINAL FLAG SET
    confirmed_flags() returns every confirmed track ever created, including
    retired confirmed tracks, sorted by id.  This matches the mission need to
    keep a counted flag after it leaves the current view.  Retired tracks are
    never considered for future association.

T10 ID POLICY
    Track ids start at 1, are assigned once, and are never reused.

T11 FRAME ORDER
    update() is called once per frame with strictly increasing integer
    frame_idx values.  Non-increasing frame indices raise ValueError.

T12 CALLER-SIDE PROTOCOL WIDTH
    `ioutils.output.TrackerLike.update()` declares `detections:
    Sequence[object]` (VisionSystem never inspects detection contents - see
    output.py's DetectorLike/TrackerLike docstrings), so `update()`'s
    parameter here is typed the same way, to structurally satisfy that
    Protocol rather than silently narrowing what it claims to accept.  The
    runtime requirement is unchanged: every item must actually be a
    `Detection` instance, enforced by the same `isinstance` check as before
    (now run once, up front, into a locally-typed `list[Detection]`); a
    non-`Detection` item still raises `TypeError`.

PIPELINE NOTE
-------------
The detector produces Detection objects from each frame.  The geolocation layer
fills Detection.coord when a per-frame GPS estimate is available.  The mission
loop calls Tracker.update(detections, frame_idx) once per frame and reads
Tracker.confirmed_flags() at the end for one committed class and coordinate per
confirmed physical flag.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

Box = tuple[float, float, float, float]
GPSCoord = tuple[float, float]


def _validate_finite_float(name: str, value: float) -> float:
    """Return `value` as float after checking it is finite."""
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric value") from exc

    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return scalar


def _validate_box(box: Sequence[float]) -> Box:
    """Validate and normalize a detector box in x1, y1, x2, y2 format."""
    if len(box) != 4:
        raise ValueError(f"box must contain exactly four values; got {box!r}")

    x1, y1, x2, y2 = (
        _validate_finite_float("box coordinate", value)
        for value in box
    )
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            "box must have positive width and height in x1, y1, x2, y2 order; "
            f"got {(x1, y1, x2, y2)!r}"
        )
    return x1, y1, x2, y2


def _validate_class_id(class_id: int) -> int:
    """Validate and return a detector class id."""
    if isinstance(class_id, bool) or not isinstance(class_id, int):
        raise ValueError(f"class_id must be an integer; got {class_id!r}")
    return class_id


def _validate_coord(coord: Sequence[float] | None) -> GPSCoord | None:
    """Validate and normalize an optional finite coordinate pair."""
    if coord is None:
        return None
    if len(coord) != 2:
        raise ValueError(f"coord must contain exactly two values; got {coord!r}")

    lat = _validate_finite_float("coord latitude", coord[0])
    lon = _validate_finite_float("coord longitude", coord[1])
    return lat, lon


def _iou(box_a: Box, box_b: Box) -> float:
    """Return the intersection-over-union of two positive-area boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection == 0.0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union


@dataclass(frozen=True, slots=True)
class Detection:
    """One detector output for a single frame.

    Attributes:
        box: Detector box as (x1, y1, x2, y2) pixels.
        class_id: Detector class label.  This is treated as the noisy per-frame
            flag identity and is aggregated by the tracker.
        score: Detector confidence.  Validated but not used for matching or
            class voting.  # ASSUMPTION T4
        coord: Optional (lat, lon) GPS estimate for this detection.
    """

    box: Box
    class_id: int
    score: float
    coord: GPSCoord | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "box", _validate_box(self.box))
        object.__setattr__(self, "class_id", _validate_class_id(self.class_id))
        object.__setattr__(self, "score", _validate_finite_float("score", self.score))
        object.__setattr__(self, "coord", _validate_coord(self.coord))


@dataclass(slots=True)
class Track:
    """State and committed views for one physical flag hypothesis.

    The public fields make the tracker easy to inspect during mission testing.
    `box` is the last observed detector box.  `class_votes` and `coords` retain
    the evidence used by committed_class() and committed_coord().
    """

    id: int
    box: Box
    hits: int
    age: int
    first_seen_frame: int
    last_seen_frame: int
    confirmed: bool
    class_votes: dict[int, int] = field(default_factory=dict)
    coords: list[GPSCoord] = field(default_factory=list)
    retired: bool = False

    def committed_class(self) -> int | None:
        """Return the deterministic majority-vote class for this track."""
        if not self.class_votes:
            return None

        best_count = max(self.class_votes.values())
        # ASSUMPTION T5: lowest class_id wins ties.
        return min(
            class_id
            for class_id, count in self.class_votes.items()
            if count == best_count
        )

    def committed_coord(self) -> GPSCoord | None:
        """Return the per-axis median GPS coordinate for this track."""
        if not self.coords:
            return None

        # ASSUMPTION T6: per-axis median gives robust aggregation.
        coord_array = np.asarray(self.coords, dtype=np.float64)
        lat, lon = np.median(coord_array, axis=0)
        return float(lat), float(lon)

    def add_detection(
        self,
        detection: Detection,
        frame_idx: int,
        min_hits: int,
    ) -> None:
        """Update this track with a matched detection."""
        self.box = detection.box
        self.hits += 1
        self.age = frame_idx - self.first_seen_frame + 1
        self.last_seen_frame = frame_idx
        self.class_votes[detection.class_id] = (
            self.class_votes.get(detection.class_id, 0) + 1
        )
        if detection.coord is not None:
            self.coords.append(detection.coord)
        if self.hits >= min_hits:
            # ASSUMPTION T7: confirmation is sticky.
            self.confirmed = True


class Tracker:
    """Simplified ByteTrack-style multi-object tracker for static flags.

    The tracker maintains active tracks for association and archives retired
    confirmed tracks for mission-level reporting.  Call update() exactly once
    per frame with detections for that frame.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        min_hits: int = 3,
        max_age: int = 5,
    ) -> None:
        """Create a tracker.

        Args:
            iou_threshold: Minimum IoU required to match a detection to a track.
                # ASSUMPTION T2
            min_hits: Number of matched observations needed to confirm a track.
                # ASSUMPTION T7
            max_age: Maximum tolerated frame-index gap before retirement.
                # ASSUMPTION T8
        """
        threshold = _validate_finite_float("iou_threshold", iou_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in [0, 1]; got {threshold!r}")
        if isinstance(min_hits, bool) or not isinstance(min_hits, int) or min_hits <= 0:
            raise ValueError(f"min_hits must be a positive integer; got {min_hits!r}")
        if isinstance(max_age, bool) or not isinstance(max_age, int) or max_age < 0:
            raise ValueError(f"max_age must be a non-negative integer; got {max_age!r}")

        self.iou_threshold = threshold
        self.min_hits = min_hits
        self.max_age = max_age

        self._active_tracks: list[Track] = []
        self._retired_tracks: list[Track] = []
        self._next_id = 1
        self._last_frame_idx: int | None = None

    def update(self, detections: Sequence[object], frame_idx: int) -> list[Track]:
        """Associate detections for one frame and return confirmed flag tracks.

        Args:
            detections: Detector outputs for the frame, as a sequence of
                opaque objects (matches `ioutils.output.TrackerLike.update`,
                which promises nothing about their shape - ASSUMPTION T12).
                Each item must actually be a `Detection` instance; the order
                is allowed to change between frames, association is based on
                IoU.
            frame_idx: Strictly increasing integer frame index.  # ASSUMPTION T11

        Returns:
            The current mission-level confirmed flag set: every confirmed track
            ever created, including retired confirmed tracks.  # ASSUMPTION T9

        Raises:
            TypeError: if any item in `detections` is not a `Detection`
                instance.  # ASSUMPTION T12
        """
        frame_idx = self._validate_frame_idx(frame_idx)
        self._retire_stale_tracks(frame_idx)

        validated_detections: list[Detection] = []
        for detection in detections:
            if not isinstance(detection, Detection):
                raise TypeError(
                    "detections must contain Detection instances; "
                    f"got {type(detection).__name__}"
                )
            validated_detections.append(detection)

        matches, unmatched_detection_indices = self._associate(validated_detections)

        for track_index, detection_index in matches:
            self._active_tracks[track_index].add_detection(
                validated_detections[detection_index],
                frame_idx,
                self.min_hits,
            )

        for detection_index in unmatched_detection_indices:
            self._active_tracks.append(
                self._new_track(validated_detections[detection_index], frame_idx)
            )

        return self.confirmed_flags()

    def confirmed_flags(self) -> list[Track]:
        """Return all confirmed tracks ever created, sorted by stable id."""
        # ASSUMPTION T9: keep retired confirmed flags in the final answer.
        confirmed = [
            track
            for track in (*self._retired_tracks, *self._active_tracks)
            if track.confirmed
        ]
        return sorted(confirmed, key=lambda track: track.id)

    def active_tracks(self) -> list[Track]:
        """Return active tracks, including unconfirmed hypotheses, sorted by id."""
        return sorted(self._active_tracks, key=lambda track: track.id)

    def _validate_frame_idx(self, frame_idx: int) -> int:
        """Validate monotonic frame order and record the latest frame index."""
        if isinstance(frame_idx, bool) or not isinstance(frame_idx, int):
            raise ValueError(f"frame_idx must be an integer; got {frame_idx!r}")
        if frame_idx < 0:
            raise ValueError(f"frame_idx must be non-negative; got {frame_idx!r}")
        if self._last_frame_idx is not None and frame_idx <= self._last_frame_idx:
            raise ValueError(
                "frame_idx values must be strictly increasing; "
                f"got {frame_idx!r} after {self._last_frame_idx!r}"
            )
        self._last_frame_idx = frame_idx
        return frame_idx

    def _predicted_box(self, track: Track) -> Box:
        """Return the predicted box used for association."""
        # ASSUMPTION T1: constant-position prediction.
        return track.box

    def _associate(self, detections: list[Detection]) -> tuple[list[tuple[int, int]], list[int]]:
        """Greedily match active tracks to detections using gated IoU."""
        if not self._active_tracks:
            return [], list(range(len(detections)))
        if not detections:
            return [], []

        candidates: list[tuple[float, int, int, int]] = []
        for track_index, track in enumerate(self._active_tracks):
            predicted_box = self._predicted_box(track)
            for detection_index, detection in enumerate(detections):
                overlap = _iou(predicted_box, detection.box)
                if overlap >= self.iou_threshold:
                    candidates.append((-overlap, track.id, track_index, detection_index))

        # ASSUMPTION T3: greedy assignment with deterministic tie-breaking.
        candidates.sort()

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        matches: list[tuple[int, int]] = []
        for _negative_iou, _track_id, track_index, detection_index in candidates:
            if track_index in matched_tracks or detection_index in matched_detections:
                continue
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)
            matches.append((track_index, detection_index))

        unmatched_detection_indices = [
            index
            for index in range(len(detections))
            if index not in matched_detections
        ]
        return matches, unmatched_detection_indices

    def _new_track(self, detection: Detection, frame_idx: int) -> Track:
        """Create a new active track from an unmatched detection."""
        track = Track(
            id=self._next_id,
            box=detection.box,
            hits=0,
            age=1,
            first_seen_frame=frame_idx,
            last_seen_frame=frame_idx,
            confirmed=False,
        )
        # ASSUMPTION T10: ids are monotonically assigned and never reused.
        self._next_id += 1
        track.add_detection(detection, frame_idx, self.min_hits)
        return track

    def _retire_stale_tracks(self, frame_idx: int) -> None:
        """Move tracks past the max_age boundary out of active association."""
        still_active: list[Track] = []
        for track in self._active_tracks:
            track.age = frame_idx - track.first_seen_frame + 1
            if frame_idx - track.last_seen_frame > self.max_age:
                # ASSUMPTION T8: age-out happens before association.
                track.retired = True
                self._retired_tracks.append(track)
            else:
                still_active.append(track)
        self._active_tracks = still_active
