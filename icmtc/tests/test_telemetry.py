# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   10/06/2026

"""test_telemetry.py — hardware-free pytest suite for telemetry.py.

All tests inject synthetic Pose samples via TelemetryBuffer.add_sample;
no MAVLink connection is ever opened.

Run:
    pytest test_telemetry.py -v
"""

from __future__ import annotations

import math
import threading
import time as _time

import pytest

from geo.telemetry import (
    Pose,
    TelemetryBuffer,
    _gimbal_frame_from_flags,
    _is_csv_replay_connection,
    interpolate,
)


# ---------------------------------------------------------------------------
# Helpers

def make_pose(
    t: float = 0.0,
    lat: float = 30.0,
    lon: float = 31.0,
    alt_agl: float = 50.0,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    gimbal_quat: tuple[float, float, float, float] | None = (1.0, 0.0, 0.0, 0.0),
    gimbal_frame: str | None = "body",
    fix_type: int | None = 3,
    h_acc_m: float | None = 1.5,
) -> Pose:
    return Pose(
        t=t,
        lat=lat,
        lon=lon,
        alt_agl=alt_agl,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        gimbal_quat=gimbal_quat,
        gimbal_frame=gimbal_frame,
        fix_type=fix_type,
        h_acc_m=h_acc_m,
    )


# ---------------------------------------------------------------------------
# Test 0 — MAVLink "vehicle" flag translates to internal "body" vocabulary

def test_gimbal_frame_from_flags_translates_vehicle_to_body():
    """MAVLink's VEHICLE_FRAME/default bits decode to "body", never "vehicle"."""
    assert _gimbal_frame_from_flags(0) == "body"                # default, no flags set
    assert _gimbal_frame_from_flags(32) == "body"                # YAW_IN_VEHICLE_FRAME
    assert _gimbal_frame_from_flags(64) == "earth"               # YAW_IN_EARTH_FRAME
    assert _gimbal_frame_from_flags(16) == "earth"               # YAW_LOCK (older)
    assert _gimbal_frame_from_flags(32 | 64) == "earth"          # newer bits: earth wins


# ---------------------------------------------------------------------------
# Test 1 — exact timestamp hit

def test_exact_hit():
    """Query at a timestamp equal to a stored sample returns that sample."""
    buf = TelemetryBuffer()
    p1 = make_pose(t=1.0, yaw=0.1)
    p2 = make_pose(t=2.0, yaw=0.2)
    p3 = make_pose(t=3.0, yaw=0.3)
    buf.add_sample(p1)
    buf.add_sample(p2)
    buf.add_sample(p3)

    result = buf.pose_at(2.0)
    assert result is p2, "Expected the exact sample object at t=2.0"


# ---------------------------------------------------------------------------
# Test 2 — midpoint linear interpolation

def test_midpoint_interpolation():
    """Midpoint between two samples has the arithmetic mean of each linear field."""
    p0 = make_pose(t=0.0, lat=10.0, lon=20.0, alt_agl=50.0,
                   roll=0.0, pitch=0.0, yaw=0.0, h_acc_m=2.0)
    p1 = make_pose(t=1.0, lat=11.0, lon=21.0, alt_agl=60.0,
                   roll=0.2, pitch=0.1, yaw=0.4, h_acc_m=3.0)
    buf = TelemetryBuffer()
    buf.add_sample(p0)
    buf.add_sample(p1)

    result = buf.pose_at(0.5)
    assert result is not None

    assert result.lat     == pytest.approx(10.5)
    assert result.lon     == pytest.approx(20.5)
    assert result.alt_agl == pytest.approx(55.0)
    assert result.roll    == pytest.approx(0.1,   abs=1e-9)
    assert result.pitch   == pytest.approx(0.05,  abs=1e-9)
    assert result.yaw     == pytest.approx(0.2,   abs=1e-9)
    assert result.h_acc_m == pytest.approx(2.5,   abs=1e-9)
    assert result.t       == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Test 3 — yaw wrap across ±π

def test_yaw_wrap():
    """Yaw 179° → −179° midpoint is ≈ 180° (shortest arc, NOT 0°)."""
    yaw_a = math.radians(179.0)
    yaw_b = math.radians(-179.0)
    p0 = make_pose(t=0.0, yaw=yaw_a)
    p1 = make_pose(t=1.0, yaw=yaw_b)
    buf = TelemetryBuffer()
    buf.add_sample(p0)
    buf.add_sample(p1)

    result = buf.pose_at(0.5)
    assert result is not None

    # Shortest arc through ±π: step is +2°, midpoint is 180° = ±π
    assert abs(result.yaw) == pytest.approx(math.pi, abs=1e-9), (
        f"Expected |yaw| ≈ π (180°), got {math.degrees(result.yaw):.6f}°  "
        f"(incorrect 0° would mean arithmetic average was used)"
    )


# ---------------------------------------------------------------------------
# Test 4 — gimbal quaternion SLERP

def test_quaternion_slerp():
    """Gimbal quaternion midpoint is the correct spherical linear interpolation."""
    # q0 = identity (0° rotation)
    # q1 = 90° rotation around Z axis in wxyz: (cos 45°, 0, 0, sin 45°)
    q0: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    q1: tuple[float, float, float, float] = (
        math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)
    )
    p0 = make_pose(t=0.0, gimbal_quat=q0)
    p1 = make_pose(t=1.0, gimbal_quat=q1)
    buf = TelemetryBuffer()
    buf.add_sample(p0)
    buf.add_sample(p1)

    result = buf.pose_at(0.5)
    assert result is not None
    assert result.gimbal_quat is not None

    # Expected: 45° rotation around Z → (cos 22.5°, 0, 0, sin 22.5°) in wxyz
    w, x, y, z = result.gimbal_quat
    assert w == pytest.approx(math.cos(math.pi / 8), abs=1e-6)
    assert x == pytest.approx(0.0,                   abs=1e-6)
    assert y == pytest.approx(0.0,                   abs=1e-6)
    assert z == pytest.approx(math.sin(math.pi / 8), abs=1e-6)

    # Result must remain a unit quaternion
    norm = math.sqrt(w**2 + x**2 + y**2 + z**2)
    assert norm == pytest.approx(1.0, abs=1e-9)


def test_quaternion_not_slerped_across_gimbal_frame_change():
    """Body-frame and Earth-frame gimbal quaternions must not be SLERPed."""
    q_body: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    q_earth: tuple[float, float, float, float] = (
        math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)
    )
    p0 = make_pose(t=0.0, gimbal_quat=q_body, gimbal_frame="body")
    p1 = make_pose(t=1.0, gimbal_quat=q_earth, gimbal_frame="earth")
    buf = TelemetryBuffer()
    buf.add_sample(p0)
    buf.add_sample(p1)

    result = buf.pose_at(0.5)
    assert result is not None

    assert result.gimbal_frame == "earth"
    assert result.gimbal_quat == q_earth


# ---------------------------------------------------------------------------
# Test 5 — out-of-range queries return None

def test_out_of_range():
    """Queries before the first or after the last sample return None."""
    buf = TelemetryBuffer()
    buf.add_sample(make_pose(t=1.0))
    buf.add_sample(make_pose(t=2.0))

    assert buf.pose_at(0.999) is None, "Before first sample should be None"
    assert buf.pose_at(2.001) is None, "After last sample should be None"

    # Boundary values themselves are valid
    assert buf.pose_at(1.0) is not None
    assert buf.pose_at(2.0) is not None


# ---------------------------------------------------------------------------
# Test 6 — ring buffer eviction

def test_ring_eviction():
    """Adding samples beyond capacity evicts the oldest ones."""
    buf = TelemetryBuffer(capacity_seconds=1.0)

    # Add samples at t = 0.0, 0.1, 0.2, …, 1.9 (20 samples over 2 s)
    for i in range(20):
        buf.add_sample(make_pose(t=round(i * 0.1, 9)))

    # Newest sample is at t=1.9; cutoff is 1.9 − 1.0 = 0.9
    # Samples with t < 0.9 must have been evicted.
    assert buf.pose_at(0.8) is None, "t=0.8 should have been evicted"
    assert buf.pose_at(0.9) is not None, "t=0.9 is exactly at the boundary, must survive"
    assert buf.pose_at(1.5) is not None, "t=1.5 is within the retained window"

    # Memory is bounded: buffer length ≤ samples in [0.9, 1.9] = 11
    with buf._lock:
        assert len(buf._buf) <= 11, (
            f"Buffer holds {len(buf._buf)} samples; expected ≤11 (1 s window at 10 Hz)"
        )


# ---------------------------------------------------------------------------
# Test 7 — concurrent add_sample and pose_at are race-free

def test_thread_safety():
    """Concurrent readers and writers do not crash or corrupt the buffer."""
    buf = TelemetryBuffer(capacity_seconds=0.5)
    errors: list[str] = []
    done = threading.Event()

    def writer(start: float, count: int) -> None:
        for i in range(count):
            try:
                buf.add_sample(make_pose(t=start + i * 0.001))
            except Exception as exc:
                errors.append(f"writer: {exc}")

    def reader() -> None:
        while not done.is_set():
            try:
                result = buf.pose_at(0.1)
                if result is not None and not isinstance(result, Pose):
                    errors.append(f"reader: wrong type {type(result)}")
            except Exception as exc:
                errors.append(f"reader: {exc}")

    reader_threads = [threading.Thread(target=reader) for _ in range(3)]
    writer_threads = [
        threading.Thread(target=writer, args=(j * 0.05, 300))
        for j in range(4)
    ]

    for t in reader_threads:
        t.start()
    for t in writer_threads:
        t.start()

    for t in writer_threads:
        t.join(timeout=5.0)
    done.set()
    for t in reader_threads:
        t.join(timeout=2.0)

    assert not errors, f"Thread-safety errors: {errors}"


# ---------------------------------------------------------------------------
# Test 8 — clock_offset shifts the effective query time

def test_clock_offset():
    """A non-zero clock_offset shifts the effective lookup time as documented."""
    p0 = make_pose(t=1.0, alt_agl=40.0)
    p1 = make_pose(t=2.0, alt_agl=60.0)

    # With offset=1.0: pose_at(0.5) → effective_t = 1.5 → midpoint of [1.0, 2.0]
    buf_offset = TelemetryBuffer(clock_offset=1.0)
    buf_offset.add_sample(p0)
    buf_offset.add_sample(p1)

    result = buf_offset.pose_at(0.5)
    assert result is not None, "With offset=1.0, t=0.5 maps to effective_t=1.5 (in range)"
    assert result.t       == pytest.approx(1.5)
    assert result.alt_agl == pytest.approx(50.0)  # midpoint of 40 and 60

    # Without offset: the same raw t=0.5 is out of range
    buf_no_offset = TelemetryBuffer(clock_offset=0.0)
    buf_no_offset.add_sample(p0)
    buf_no_offset.add_sample(p1)

    assert buf_no_offset.pose_at(0.5) is None, (
        "Without offset, t=0.5 is before the first sample at t=1.0"
    )


# ---------------------------------------------------------------------------
# Bonus: standalone interpolate() is independently testable

def test_interpolate_pure():
    """interpolate() is a pure function that does not require a buffer."""
    a = make_pose(t=0.0, lat=0.0, lon=0.0, alt_agl=10.0)
    b = make_pose(t=2.0, lat=2.0, lon=4.0, alt_agl=20.0)

    mid = interpolate(a, b, 1.0)
    assert mid.t       == pytest.approx(1.0)
    assert mid.lat     == pytest.approx(1.0)
    assert mid.lon     == pytest.approx(2.0)
    assert mid.alt_agl == pytest.approx(15.0)


def test_interpolate_degenerate_equal_timestamps():
    """interpolate() with a.t == b.t returns a without a ZeroDivisionError."""
    a = make_pose(t=1.0, lat=5.0)
    b = make_pose(t=1.0, lat=9.0)
    result = interpolate(a, b, 1.0)
    assert result is a


# ---------------------------------------------------------------------------
# Test 9 — CSV replay connection strings are rejected loudly, not silently
# swallowed deep inside the reader loop (RB-3: pymavlink.CSVReader.CSVReader
# has no close() method and its recv_match() has no timeout kwarg).

def test_is_csv_replay_connection_matches_pymavlinks_own_dispatch_rule():
    """The prefix check mirrors mavutil.mavlink_connection()'s own dispatch."""
    assert _is_csv_replay_connection("csv:replay.csv")
    assert _is_csv_replay_connection("CSV:replay.csv")  # case-insensitive
    assert _is_csv_replay_connection("Csv:replay.csv:separator=,")
    assert not _is_csv_replay_connection("udpin:0.0.0.0:14550")
    assert not _is_csv_replay_connection("tcp:127.0.0.1:5760")
    assert not _is_csv_replay_connection("serial:/dev/ttyUSB0:57600")
    assert not _is_csv_replay_connection("")


def test_start_rejects_csv_connection_string_immediately():
    """start() raises before ever spawning the reader thread or touching a file.

    Before this fix, a "csv:..." connection string silently ingested zero
    pose samples forever while leaking one file descriptor per reconnect
    attempt: CSVReader.recv_match() has no `timeout` kwarg (TypeError,
    swallowed by the reader loop's broad exception handling) and CSVReader
    has no close() method at all (AttributeError, swallowed by a blind
    contextlib.suppress(Exception)).
    """
    buf = TelemetryBuffer()

    with pytest.raises(ValueError, match="CSV replay"):
        buf.start("csv:nonexistent_replay_log.csv")

    assert buf._thread is None, (
        "Reader thread must never be spawned for a rejected connection string"
    )


def test_reader_loop_closes_connection_via_narrow_exception_handling(monkeypatch):
    """The finally-block calls conn.close() on the real connection object.

    Replaces the old blind `contextlib.suppress(Exception)` around
    conn.close() with a narrow `except OSError`; this test uses a fake
    connection object (never a real serial/UDP MAVLink link) to confirm
    close() is actually invoked once the inner read loop exits.
    """
    import pymavlink.mavutil as mavutil_module

    class _FakeConn:
        def __init__(self):
            self.closed = False

        def recv_match(self, type=None, blocking=False, timeout=None):
            return None  # nothing to ingest; loop just polls until stopped

        def close(self):
            self.closed = True

    fake = _FakeConn()
    monkeypatch.setattr(mavutil_module, "mavlink_connection", lambda device: fake)

    buf = TelemetryBuffer()
    buf._connection_str = "udpin:0.0.0.0:14550"

    def stop_soon():
        _time.sleep(0.05)
        buf._stop.set()

    stopper = threading.Thread(target=stop_soon)
    stopper.start()
    buf._reader_loop()
    stopper.join()

    assert fake.closed, "conn.close() must be called once the inner loop exits"
