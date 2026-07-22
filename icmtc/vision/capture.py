# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   12/06/2026

"""capture.py - frame source for the vision pipeline (ICMTC 2026, Mission 2).

    [ capture ] -> detector -> tracker -> geolocate -> output -> mission

This file owns exactly one job: hand the rest of the pipeline a timestamped
frame.  It does not detect, decode telemetry, or make mission decisions -
those live downstream.

================================================================================
SCOPE - read first
================================================================================
No SIYI A8 gimbal camera and no Raspberry Pi 5 are available yet, so the real
RTSP/H.265 hardware-decode path cannot be built OR verified today.  This file
therefore ships only the dev/test frame source - `FileFrameSource`, which
reads a recorded video file or a directory of sequential images - on the
exact `FrameSource` interface the future RTSP source must satisfy.  The RTSP
class itself is a documented, clearly-marked placeholder (`RtspFrameSource`)
that raises `NotImplementedError`; it deliberately fakes nothing, because an
unverifiable "working" RTSP implementation is worse than an honest gap.  See
ASSUMPTION CAP10.

================================================================================
ASSUMPTIONS (collected here; also marked inline with # ASSUMPTION: <tag>)
================================================================================

CAP1  FRAME FORMAT
    Every frame is a numpy.ndarray, shape (H, W, 3), BGR, uint8 - identical
    to detector.py's convention and export.py's `load_test_image` (both via
    cv2, ASSUMPTION E8 in export.py).  No RGB conversion happens here.

CAP2  TIMESTAMP CONVENTION
    The timestamp returned alongside each frame is `time.monotonic()`,
    sampled at the instant the frame is handed to the caller - matching
    telemetry.py's ASSUMPTION A1 (buffer timestamps are Pi-receive-time
    `time.monotonic()`).  It is NOT derived from the video container's
    embedded PTS or frame index; a recorded file's internal timing is
    ignored entirely.  This keeps a `TelemetryBuffer.pose_at(t)` lookup
    against frames from this source mechanically identical in code to a
    lookup against real camera frames later, even though a recorded video
    obviously has no telemetry correlated to its content - the point is
    interface parity for `mission.py`, not physical meaning during file
    playback.

CAP3  PACING MODES
    `realtime=True` (default) paces `read()` to the source's FPS using
    `time.sleep()` on a fixed-rate schedule (each frame's due time is the
    previous due time plus one interval, not "now + interval", so brief
    scheduling jitter does not compound into long-run drift).  If the caller
    stops calling `read()` for a long pause, the schedule is not reset and
    will "catch up" by returning subsequent frames with little or no sleep
    until it does - there is no burst-suppression, since this is a dev/test
    tool, not a real-time guarantee.  `realtime=False` returns frames as
    fast as asked, no sleep at all, for batch/regression runs.

CAP4  FPS FALLBACK (ASSUMPTION - flagged, not silently guessed)
    Video mode: FPS is `cv2.VideoCapture`'s `CAP_PROP_FPS` unless the
    caller passes an explicit `fps=`.  If the container reports FPS <= 0
    (missing/corrupt metadata), this falls back to `_DEFAULT_FALLBACK_FPS`
    (30.0) with a logged warning.  Image-directory mode has no inherent FPS
    at all; it always uses `fps=` if given, else the same 30.0 default.
    Pick a real `fps=` for accurate pacing against a specific test rig.

CAP5  EXHAUSTION VS ERROR
    `read()` returns `None` exactly once the source has no more frames
    (video: `cap.read()` reports `ok=False`; images: every path has been
    tried).  This is the *only* "done" signal and is distinct from a
    transient per-frame problem.  A construction-time failure (missing
    path, an unopenable/corrupt video file, or an image directory with zero
    readable images) instead raises `CaptureError` immediately - fail loud
    at startup, matching config.py's ASSUMPTION G1 - because a silently
    empty frame source is worse than a startup crash naming the bad path.

CAP6  MID-STREAM BAD FRAME (video)
    `cv2.VideoCapture` conflates "this frame failed to decode" and "this is
    the end of the stream" into the same `ok=False` return; there is no
    portable way to tell them apart from outside OpenCV, so an `ok=False`
    is always treated as exhaustion here (ASSUMPTION CAP5).  The one
    mid-stream failure this module CAN reliably detect and skip is
    `ok=True` paired with a malformed/undecodable frame (wrong shape,
    `None`, wrong dtype) - `_is_valid_frame` gates every video and image
    frame on this before it is ever returned, and a bad one is logged and
    skipped, never surfaced as a crash or as `None`.  Bounded by
    `_MAX_CONSECUTIVE_BAD_FRAMES` so a pathological stream that never
    recovers cannot spin forever.

CAP7  IMAGE FILE ORDERING
    Directory-of-images mode reads files sorted by filename (plain string
    sort) filtered to `_IMAGE_EXTENSIONS`.  The caller's naming scheme
    (e.g. `frame_0001.png`, `frame_0002.png`, ...) is what determines
    playback order; this module does not inspect file contents or mtimes.

CAP8  LATESTFRAMEQUEUE SEMANTICS
    `LatestFrameQueue` holds a single slot, not a FIFO queue: each new
    frame from the background thread overwrites the previous one, so a
    caller that reads slower than the source produces always gets the
    newest frame and silently drops the backlog - never a growing queue,
    never a replay of stale frames.  This is exactly the behaviour the
    future RTSP source will also need (never let inference fall behind a
    live feed), so it is written once here, against the file source, and
    is meant to wrap `RtspFrameSource` unchanged later.

CAP9  STALL / EXHAUSTION SIGNAL
    `LatestFrameQueue.exhausted` is set the moment the wrapped source's
    `read()` returns `None` (or raises - see below).  `LatestFrameQueue.age`
    returns seconds since the newest frame arrived (`None` before the
    first), mirroring telemetry.TelemetryBuffer.latest_age, so a caller can
    detect a stalled-but-not-yet-exhausted source (e.g. a live feed that
    has stopped delivering) the same way it already watches telemetry
    staleness.

CAP10  RTSP DEFERRED
    `RtspFrameSource` is an intentionally unimplemented placeholder: it
    raises `NotImplementedError` from `__init__` rather than shipping a
    fabricated "working" GStreamer/H.265 pipeline that has never touched
    the actual SIYI A8 + Raspberry Pi 5 hardware.  Once that hardware
    exists, implement it to satisfy the same `FrameSource` Protocol
    (`read()` / `close()`, ASSUMPTION CAP1/CAP2 conventions) so
    `mission.py`'s dependency on `FrameSource` never has to change - only
    the one constructor call at startup does.

CAP11  CLOSE IDEMPOTENCY
    `close()` on both `FileFrameSource` and `LatestFrameQueue` is safe to
    call more than once (an internal flag short-circuits repeats),
    mirroring logging_cfg.configure_logging's idempotency precedent.

================================================================================
USAGE
-----
    source = FileFrameSource("recording.mp4")            # or a directory of images
    # once RtspFrameSource exists:  source = RtspFrameSource(...)
    frames = LatestFrameQueue(source)
    ...
    result = frames.read()             # -> (frame, capture_t) | None
mission.py constructs whichever `FrameSource` is available - `FileFrameSource`
today, `RtspFrameSource` once hardware exists - and depends only on the
`FrameSource` / `LatestFrameQueue` interface; the rest of the pipeline never
changes.
================================================================================
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Protocol

import cv2  # ASSUMPTION: Mac/dev-only dependency of this file (mirrors export.py ASSUMPTION E8)
import numpy as np

from ioutils.logging_cfg import get_logger

log = get_logger(__name__)

# ASSUMPTION CAP4: fallback FPS when neither an explicit fps= nor a usable
# native FPS is available.
_DEFAULT_FALLBACK_FPS = 30.0

# ASSUMPTION CAP7: recognized still-image extensions for directory mode.
_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})

# ASSUMPTION CAP6: bound on consecutive undecodable video frames before
# giving up (treated as exhaustion) instead of spinning forever.
_MAX_CONSECUTIVE_BAD_FRAMES = 50


class CaptureError(Exception):
    """Raised when a frame source cannot be opened or is unreadable at construction."""


# ---------------------------------------------------------------------------
# Shared interface - both FileFrameSource (today) and RtspFrameSource
# (once hardware exists) satisfy this.  mission.py depends only on this.
# ---------------------------------------------------------------------------
class FrameSource(Protocol):
    """A source of timestamped BGR frames.  # ASSUMPTION CAP1 / CAP2"""

    def read(self) -> tuple[np.ndarray, float] | None:
        """Return the next (frame, capture_timestamp_monotonic_seconds).

        Returns:
            `(frame, t)` where `frame` is (H, W, 3) BGR uint8 and `t` is
            `time.monotonic()` at hand-out time, or `None` when the source
            is exhausted (or has no frame yet, for a live/queued source).
        """
        ...

    def close(self) -> None:
        """Release any underlying resources.  Must be safe to call twice."""
        ...


class _CvCapture(Protocol):
    """The slice of `cv2.VideoCapture`'s surface this module depends on.

    Kept as a Protocol (mirroring webcam_infer.py's `VideoCapture` seam) so
    `FileFrameSource`'s video path is testable with a scripted fake instead
    of a real video file, e.g. to inject a mid-stream corrupt frame.
    """

    def isOpened(self) -> bool: ...  # noqa: N802 - matches cv2's camelCase API

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def get(self, propId: int) -> float: ...  # noqa: N803 - matches cv2's camelCase API

    def release(self) -> None: ...


def _is_valid_frame(frame: object) -> bool:
    """True only for a non-empty (H, W, 3) BGR uint8 array.  # ASSUMPTION CAP1"""
    return (
        isinstance(frame, np.ndarray)
        and frame.ndim == 3
        and frame.shape[2] == 3
        and frame.shape[0] > 0
        and frame.shape[1] > 0
        and frame.dtype == np.uint8
    )


# ---------------------------------------------------------------------------
# The dev/test fallback source.
# ---------------------------------------------------------------------------
class FileFrameSource:
    """Reads a recorded video file or a directory of sequential images.

    Satisfies `FrameSource`.  Frame content comes from `cv2` (video decode
    or `cv2.imread`); every timestamp is assigned locally at hand-out time
    (ASSUMPTION CAP2), never read from the file.
    """

    def __init__(
        self,
        source: str | Path,
        *,
        realtime: bool = True,
        fps: float | None = None,
        capture_factory: Callable[[str], _CvCapture] | None = None,
    ) -> None:
        """Open a video file or image directory as a frame source.

        Args:
            source: Path to a video file, or a directory of sequentially
                named images (ASSUMPTION CAP7).
            realtime: Pace `read()` to the source's FPS with real
                wall-clock sleeps (default), or return frames as fast as
                asked when False.  # ASSUMPTION CAP3
            fps: Overrides the pacing rate.  Required to have a meaningful
                rate for image-directory mode unless the 30.0 default is
                acceptable.  # ASSUMPTION CAP4
            capture_factory: Overrides `cv2.VideoCapture` construction for
                video mode (tests only), e.g. to inject a scripted mid-
                stream bad frame.

        Raises:
            ValueError: if `fps` is given and not positive.
            CaptureError: if `source` does not exist, a video file cannot
                be opened, or an image directory contains no readable
                image files.  # ASSUMPTION CAP5
        """
        if fps is not None and fps <= 0:
            raise ValueError(f"fps must be positive; got {fps!r}")

        self._path = Path(source).expanduser()
        self._realtime = realtime
        self._closed = False
        self._next_due: float | None = None
        self._cap: _CvCapture | None = None
        self._image_paths: list[Path] = []
        self._index = 0

        if self._path.is_dir():
            self._mode = "images"
            self._image_paths = sorted(
                p
                for p in self._path.iterdir()
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
            )
            if not self._image_paths:
                raise CaptureError(
                    f"no image files found in directory: {self._path} "
                    f"(looked for extensions {sorted(_IMAGE_EXTENSIONS)})"
                )
            self._fps = float(fps) if fps is not None else _DEFAULT_FALLBACK_FPS
            log.info(
                "FileFrameSource: image-sequence mode, %d frame(s) in %s, "
                "fps=%.2f, realtime=%s",
                len(self._image_paths), self._path, self._fps, realtime,
            )
        elif self._path.is_file():
            self._mode = "video"
            factory = capture_factory or cv2.VideoCapture
            cap = factory(str(self._path))
            if cap is None or not cap.isOpened():
                raise CaptureError(
                    f"could not open video file (missing, unsupported, or "
                    f"corrupt): {self._path}"
                )
            self._cap = cap
            native_fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps is not None:
                self._fps = float(fps)
            elif native_fps > 0:
                self._fps = native_fps
            else:
                log.warning(
                    "video %s reports no usable FPS (got %.3f); falling "
                    "back to %.1f",  # ASSUMPTION CAP4
                    self._path, native_fps, _DEFAULT_FALLBACK_FPS,
                )
                self._fps = _DEFAULT_FALLBACK_FPS
            log.info(
                "FileFrameSource: video mode, %s, fps=%.2f, realtime=%s",
                self._path, self._fps, realtime,
            )
        else:
            raise CaptureError(f"frame source not found: {self._path}")

        self._interval = 1.0 / self._fps

    def read(self) -> tuple[np.ndarray, float] | None:
        """Return the next (frame, capture_timestamp), or None when exhausted.

        Never raises for a recoverable per-frame problem; only construction
        can raise (ASSUMPTION CAP5).
        """
        if self._closed:
            return None

        frame = (
            self._next_video_frame() if self._mode == "video" else self._next_image_frame()
        )
        if frame is None:
            return None

        if self._realtime:
            self._pace()

        timestamp = time.monotonic()  # ASSUMPTION CAP2: assigned at hand-out time
        return frame, timestamp

    def close(self) -> None:
        """Release the underlying video capture / file handles.  Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    # Internal per-mode frame retrieval.

    def _next_image_frame(self) -> np.ndarray | None:
        """Advance through the image list, skipping unreadable files."""
        while self._index < len(self._image_paths):
            path = self._image_paths[self._index]
            self._index += 1
            try:
                frame = cv2.imread(str(path))
            except Exception:  # ASSUMPTION: never raise out of the read loop
                log.warning("error reading image frame %s; skipping", path, exc_info=True)
                continue
            if not _is_valid_frame(frame):
                log.warning("skipping unreadable/corrupt image frame: %s", path)
                continue
            return frame
        return None

    def _next_video_frame(self) -> np.ndarray | None:
        """Read the next valid video frame, skipping corrupt ones.  # ASSUMPTION CAP6"""
        assert self._cap is not None
        consecutive_bad = 0
        while True:
            try:
                ok, frame = self._cap.read()
            except Exception:  # ASSUMPTION: never raise out of the read loop
                log.warning(
                    "error reading video frame from %s; treating as end of stream",
                    self._path, exc_info=True,
                )
                return None
            if not ok:
                return None  # true exhaustion, or an unrecoverable decode
                # failure cv2 cannot distinguish from EOF.  # ASSUMPTION CAP6
            if _is_valid_frame(frame):
                return frame

            consecutive_bad += 1
            log.warning("skipping corrupt/malformed video frame in %s", self._path)
            if consecutive_bad >= _MAX_CONSECUTIVE_BAD_FRAMES:
                log.error(
                    "too many consecutive corrupt frames (%d) in %s; giving up",
                    consecutive_bad, self._path,
                )
                return None

    def _pace(self) -> None:
        """Sleep, if needed, to hold the fixed-rate schedule.  # ASSUMPTION CAP3"""
        now = time.monotonic()
        if self._next_due is None:
            self._next_due = now + self._interval
            return
        delay = self._next_due - now
        if delay > 0:
            time.sleep(delay)
        self._next_due += self._interval


# ---------------------------------------------------------------------------
# Bounded "latest-wins" wrapper - shared by the file source today and the
# RTSP source later (ASSUMPTION CAP8).
# ---------------------------------------------------------------------------
class LatestFrameQueue:
    """Runs a `FrameSource` on a background thread; always serves the newest frame.

    A single-slot design (ASSUMPTION CAP8): unread frames are overwritten,
    never queued, so a slow consumer never falls behind on stale frames.
    Structurally also satisfies `FrameSource` (`read()` / `close()`), so it
    can itself be wrapped or passed anywhere a `FrameSource` is expected.
    """

    def __init__(self, source: FrameSource) -> None:
        """Wrap `source` and immediately start consuming it on a daemon thread."""
        self._source = source
        self._lock = threading.Lock()
        self._latest: tuple[np.ndarray, float] | None = None
        self._exhausted = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="latest-frame-queue", daemon=True
        )
        self._thread.start()

    def read(self) -> tuple[np.ndarray, float] | None:
        """Return the most recently produced (frame, t), or None if none has arrived yet."""
        with self._lock:
            return self._latest

    def close(self) -> None:
        """Stop the background thread and close the wrapped source.  Idempotent."""
        if self._stop.is_set():
            return
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._source.close()

    @property
    def exhausted(self) -> bool:
        """True once the wrapped source is exhausted (or its read() raised)."""
        return self._exhausted.is_set()

    @property
    def age(self) -> float | None:
        """Seconds since the newest frame arrived; None before the first frame.

        Mirrors telemetry.TelemetryBuffer.latest_age (ASSUMPTION CAP9); a
        caller can compare this against a staleness threshold to detect a
        stalled-but-not-yet-exhausted source.
        """
        with self._lock:
            if self._latest is None:
                return None
            return time.monotonic() - self._latest[1]

    def _run(self) -> None:
        """Background loop: pull from the source, overwrite the latest slot."""
        while not self._stop.is_set():
            try:
                result = self._source.read()
            except Exception:  # ASSUMPTION: a reader thread never dies silently
                log.exception(
                    "LatestFrameQueue: source.read() raised; stopping background thread"
                )
                self._exhausted.set()
                return
            if result is None:
                self._exhausted.set()
                return
            with self._lock:
                self._latest = result


# ---------------------------------------------------------------------------
# TODO: RTSP implementation, requires A8 + Pi hardware.
# ---------------------------------------------------------------------------
class RtspFrameSource:
    """SIYI A8 RTSP frame source for the Raspberry Pi 5 - NOT IMPLEMENTED.

    # TODO: RTSP implementation, requires A8 + Pi hardware.
    No SIYI A8 gimbal camera or Raspberry Pi 5 is available yet, so the real
    hardware H.265/RTSP decode pipeline (GStreamer or similar) cannot be
    built or verified against real hardware.  This class deliberately
    raises instead of shipping a fabricated "working" implementation that
    has never touched actual camera hardware.  # ASSUMPTION CAP10

    When the hardware exists, implement this to satisfy the same
    `FrameSource` Protocol as `FileFrameSource` - `read() -> (frame, t) |
    None` with `t` a `time.monotonic()` timestamp at frame hand-out time
    (ASSUMPTION CAP2), and `close() -> None` - so `mission.py` and
    `LatestFrameQueue` need no changes to adopt it; only the constructor
    call in mission.py's startup wiring changes from `FileFrameSource(...)`
    to `RtspFrameSource(...)`.
    """

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise NotImplementedError(
            "RtspFrameSource is not implemented: requires the SIYI A8 gimbal "
            "camera and Raspberry Pi 5 to build and verify a real RTSP/H.265 "
            "hardware-decode pipeline (see capture.py ASSUMPTION CAP10). Use "
            "FileFrameSource for development and testing until hardware "
            "arrives."
        )
