# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   12/06/2026

"""Tests for capture.py.

All tests are hardware-free: they use synthetic video/image fixtures built
with cv2.VideoWriter / cv2.imwrite, never real camera footage or an RTSP
connection.  No network, no MAVLink, no camera.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from vision.capture import CaptureError, FileFrameSource, LatestFrameQueue, _is_valid_frame


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------
def make_frame(width: int, height: int, marker: int) -> np.ndarray:
    """A solid-color BGR uint8 frame; `marker` identifies it for assertions."""
    return np.full((height, width, 3), fill_value=marker % 256, dtype=np.uint8)


def make_test_video(path, num_frames: int = 5, fps: float = 10.0, size=(64, 48)) -> None:
    """Write a small synthetic video with `num_frames` distinguishable frames."""
    width, height = size
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
    )
    assert writer.isOpened(), "test setup: VideoWriter failed to open"
    for i in range(num_frames):
        writer.write(make_frame(width, height, marker=i * 20 + 10))
    writer.release()


def make_test_image_dir(dir_path, num_frames: int = 4, size=(64, 48)) -> None:
    """Write `num_frames` sequentially-named PNGs into `dir_path`."""
    dir_path.mkdir(parents=True, exist_ok=True)
    width, height = size
    for i in range(num_frames):
        frame = make_frame(width, height, marker=i * 20 + 10)
        ok = cv2.imwrite(str(dir_path / f"frame_{i:04d}.png"), frame)
        assert ok, "test setup: cv2.imwrite failed"


class _ScriptedCapture:
    """A `_CvCapture`-shaped fake with a scripted (ok, frame) read sequence."""

    def __init__(self, script: list[tuple[bool, np.ndarray | None]], fps: float = 20.0) -> None:
        self._script = list(script)
        self._fps = fps
        self._released = False

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._script:
            return False, None
        return self._script.pop(0)

    def get(self, propId: int) -> float:  # noqa: N803 - matches cv2's camelCase API
        return self._fps if propId == cv2.CAP_PROP_FPS else 0.0

    def release(self) -> None:
        self._released = True


class _FakeFrameSource:
    """A minimal FrameSource producing N frames with no pacing, then None."""

    def __init__(self, num_frames: int) -> None:
        self._remaining = list(range(num_frames))
        self.closed = False

    def read(self) -> tuple[np.ndarray, float] | None:
        if not self._remaining:
            return None
        marker = self._remaining.pop(0)
        return make_frame(8, 8, marker=marker), time.monotonic()

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# 1. Correct frame format
# ---------------------------------------------------------------------------
def test_read_returns_correct_frame_format(tmp_path):
    """read() returns (ndarray, float); array is (H, W, 3) BGR uint8."""
    video_path = tmp_path / "clip.avi"
    make_test_video(video_path, num_frames=3, size=(64, 48))

    source = FileFrameSource(video_path, realtime=False)
    try:
        result = source.read()
        assert result is not None
        frame, timestamp = result
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (48, 64, 3)
        assert frame.dtype == np.uint8
        assert isinstance(timestamp, float)
    finally:
        source.close()


# ---------------------------------------------------------------------------
# 2. Timestamps monotonic increasing
# ---------------------------------------------------------------------------
def test_timestamps_monotonic_increasing(tmp_path):
    """Successive reads produce strictly increasing timestamps."""
    video_path = tmp_path / "clip.avi"
    make_test_video(video_path, num_frames=6)

    source = FileFrameSource(video_path, realtime=False)
    try:
        timestamps = []
        while True:
            result = source.read()
            if result is None:
                break
            timestamps.append(result[1])

        assert len(timestamps) == 6
        assert all(b > a for a, b in zip(timestamps, timestamps[1:]))
    finally:
        source.close()


# ---------------------------------------------------------------------------
# 3. Exhaustion returns None
# ---------------------------------------------------------------------------
def test_exhaustion_returns_none(tmp_path):
    """Reading past the last frame doesn't raise; it returns None cleanly."""
    video_path = tmp_path / "clip.avi"
    make_test_video(video_path, num_frames=2)

    source = FileFrameSource(video_path, realtime=False)
    try:
        assert source.read() is not None
        assert source.read() is not None
        assert source.read() is None
        assert source.read() is None  # remains exhausted, never raises
    finally:
        source.close()


# ---------------------------------------------------------------------------
# 4. Corrupt file at construction raises immediately
# ---------------------------------------------------------------------------
def test_corrupt_video_file_raises_at_construction(tmp_path):
    """A garbage file (not a real video) raises CaptureError immediately."""
    corrupt_path = tmp_path / "corrupt.avi"
    corrupt_path.write_bytes(b"not a real video file" * 20)

    with pytest.raises(CaptureError):
        FileFrameSource(corrupt_path)


def test_missing_source_raises_at_construction(tmp_path):
    """A path that does not exist at all also raises CaptureError."""
    with pytest.raises(CaptureError):
        FileFrameSource(tmp_path / "does_not_exist.avi")


# ---------------------------------------------------------------------------
# 5. A single bad mid-stream frame is skipped, not fatal
# ---------------------------------------------------------------------------
def test_bad_midstream_frame_is_skipped(tmp_path):
    """A malformed frame mid-stream is skipped; good frames before/after still come through."""
    good_a = make_frame(16, 16, marker=1)
    good_b = make_frame(16, 16, marker=2)
    script = [
        (True, good_a),
        (True, None),  # injected corrupt/undecodable frame
        (True, good_b),
        (False, None),  # true end of stream
    ]

    dummy_path = tmp_path / "scripted.avi"
    dummy_path.write_bytes(b"placeholder")  # never actually opened by cv2

    source = FileFrameSource(
        dummy_path,
        realtime=False,
        capture_factory=lambda _path: _ScriptedCapture(script),
    )
    try:
        frames = []
        while True:
            result = source.read()
            if result is None:
                break
            frames.append(result[0])

        assert len(frames) == 2  # the bad frame never surfaced
        assert np.array_equal(frames[0], good_a)
        assert np.array_equal(frames[1], good_b)
    finally:
        source.close()


# ---------------------------------------------------------------------------
# 6. realtime=False is not paced
# ---------------------------------------------------------------------------
def test_realtime_false_is_not_paced(tmp_path):
    """Fast batch mode (realtime=False) reads N frames in ~0 wall-clock time."""
    video_path = tmp_path / "clip.avi"
    num_frames = 8
    fps = 5.0  # paced playback of this clip would take >= (N-1)/fps ~= 1.4s
    make_test_video(video_path, num_frames=num_frames, fps=fps)

    source = FileFrameSource(video_path, realtime=False)
    try:
        start = time.monotonic()
        count = 0
        while source.read() is not None:
            count += 1
        elapsed = time.monotonic() - start

        assert count == num_frames
        assert elapsed < 0.5, f"batch read took {elapsed:.3f}s; should be near-instant"
    finally:
        source.close()


# ---------------------------------------------------------------------------
# 7. LatestFrameQueue always serves the newest frame
# ---------------------------------------------------------------------------
def test_latest_frame_queue_drops_stale_frames():
    """A fast, unpaced source overtakes a slow consumer; only the newest frame survives.

    The fake source produces frames with no pacing at all, so by the time
    the queue reports `exhausted`, many frames have been produced without
    ever being read.  A single read() afterward must return the LAST frame
    produced, not the first - if frames were queued (FIFO) instead of
    overwritten, the first read() would return frame 0, not the newest.
    """
    num_frames = 200
    fake_source = _FakeFrameSource(num_frames)
    queue = LatestFrameQueue(fake_source)
    try:
        deadline = time.monotonic() + 5.0
        while not queue.exhausted and time.monotonic() < deadline:
            time.sleep(0.01)
        assert queue.exhausted, "producer did not finish within the test deadline"

        result = queue.read()
        assert result is not None
        frame, _timestamp = result
        expected_last = make_frame(8, 8, marker=num_frames - 1)
        assert np.array_equal(frame, expected_last), (
            "expected the newest frame to survive, not an earlier queued one"
        )
    finally:
        queue.close()


def test_latest_frame_queue_returns_none_before_first_frame():
    """read() returns None if no frame has arrived yet."""
    slow_source = _FakeFrameSource(0)  # exhausted immediately, never produces
    queue = LatestFrameQueue(slow_source)
    try:
        deadline = time.monotonic() + 5.0
        while not queue.exhausted and time.monotonic() < deadline:
            time.sleep(0.01)
        assert queue.read() is None
        assert queue.age is None
    finally:
        queue.close()


# ---------------------------------------------------------------------------
# 8. close() is idempotent
# ---------------------------------------------------------------------------
def test_file_frame_source_close_is_idempotent(tmp_path):
    """Calling close() twice on FileFrameSource does not raise."""
    video_path = tmp_path / "clip.avi"
    make_test_video(video_path, num_frames=2)

    source = FileFrameSource(video_path, realtime=False)
    source.close()
    source.close()  # must not raise


def test_latest_frame_queue_close_is_idempotent():
    """Calling close() twice on LatestFrameQueue does not raise."""
    queue = LatestFrameQueue(_FakeFrameSource(3))
    queue.close()
    queue.close()  # must not raise


# ---------------------------------------------------------------------------
# 9. Directory-of-images mode works identically to video-file mode
# ---------------------------------------------------------------------------
def test_directory_of_images_matches_video_interface(tmp_path):
    """Reading a folder of sequential images works via the same FrameSource interface."""
    image_dir = tmp_path / "frames"
    make_test_image_dir(image_dir, num_frames=4, size=(64, 48))

    source = FileFrameSource(image_dir, realtime=False, fps=15.0)
    try:
        frames = []
        while True:
            result = source.read()
            if result is None:
                break
            frame, timestamp = result
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (48, 64, 3)
            assert frame.dtype == np.uint8
            assert isinstance(timestamp, float)
            frames.append(frame)

        assert len(frames) == 4
        # Sequential naming -> ascending marker order (0*20+10, 1*20+10, ...).
        for i, frame in enumerate(frames):
            assert np.array_equal(frame, make_frame(64, 48, marker=i * 20 + 10))
    finally:
        source.close()


def test_image_directory_skips_unreadable_file(tmp_path):
    """A corrupt image file mid-directory is skipped, not fatal."""
    image_dir = tmp_path / "frames"
    make_test_image_dir(image_dir, num_frames=3, size=(32, 32))
    (image_dir / "frame_0001_bad.png").write_bytes(b"not a real png")

    source = FileFrameSource(image_dir, realtime=False)
    try:
        count = 0
        while source.read() is not None:
            count += 1
        assert count == 3  # the corrupt file never surfaced as a frame
    finally:
        source.close()


def test_empty_image_directory_raises_at_construction(tmp_path):
    """A directory with no image files raises CaptureError immediately."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(CaptureError):
        FileFrameSource(empty_dir)


# ---------------------------------------------------------------------------
# Helper sanity: _is_valid_frame
# ---------------------------------------------------------------------------
def test_is_valid_frame_rejects_bad_shapes_and_types():
    assert _is_valid_frame(np.zeros((10, 10, 3), dtype=np.uint8)) is True
    assert _is_valid_frame(None) is False
    assert _is_valid_frame(np.zeros((10, 10), dtype=np.uint8)) is False  # missing channel dim
    assert _is_valid_frame(np.zeros((10, 10, 3), dtype=np.float32)) is False  # wrong dtype
    assert _is_valid_frame(np.zeros((0, 10, 3), dtype=np.uint8)) is False  # empty
