# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   10/06/2026

"""telemetry.py — MAVLink pose ingestion and timestamped lookup for UAV geo-location.

Raspberry Pi 5 companion-computer module for ICMTC 2026, Mission 2.
Feeds the :mod:`geolocate` pipeline with pose samples interpolated to the
exact monotonic timestamp of each camera frame.

SITL QUICKSTART
---------------
    pip install pymavlink scipy

    python - <<'EOF'
    import time
    from geo.telemetry import TelemetryBuffer

    buf = TelemetryBuffer()
    buf.start("udpin:0.0.0.0:14550")   # listen for ArduPilot SITL telemetry
    time.sleep(5)
    pose = buf.pose_at(time.monotonic())
    print(pose)
    buf.stop()
    EOF

Run ArduCopter SITL with:
    sim_vehicle.py -v ArduCopter --out=udpin:127.0.0.1:14550 --console

PYMAVLINK CALLS USED
--------------------
    mavutil.mavlink_connection(connection_str)
        Open any MAVLink transport: "serial:/dev/ttyUSB0:57600",
        "udpin:0.0.0.0:14550", "tcp:127.0.0.1:5760", etc.

    conn.recv_match(type=[...], blocking=True, timeout=1.0)
        Block until one of the listed message types arrives, or timeout.
        Returns None on timeout; raises on connection error.

    msg.get_type()            – message-type string, e.g. "ATTITUDE"
    msg.time_boot_ms          – autopilot boot-relative ms (ATTITUDE, GLOBAL_POSITION_INT)
    msg.lat, msg.lon          – degE7 signed integers (GLOBAL_POSITION_INT)
    msg.relative_alt          – mm signed integer, above Home (GLOBAL_POSITION_INT)
    msg.roll, msg.pitch, msg.yaw  – radians floats (ATTITUDE)
    msg.q                     – float[4] wxyz quaternion (GIMBAL_DEVICE_ATTITUDE_STATUS)
    msg.flags                 – uint16 GIMBAL_DEVICE_FLAGS bitmask
    msg.time_usec             – epoch-usec (GIMBAL_DEVICE_ATTITUDE_STATUS, GPS_RAW_INT)
    msg.fix_type              – GPS fix 0-8 (GPS_RAW_INT)
    msg.h_acc                 – horizontal accuracy, mm (GPS_RAW_INT)
    msg.roll, msg.pitch, msg.yaw  – degrees (MOUNT_ORIENTATION fallback)

================================================================================
ASSUMPTIONS (collected here; also marked inline with # ASSUMPTION: <tag>)
================================================================================

A1  TIME BASE — AUTHORITATIVE TIMESTAMP
    Buffer timestamps are time.monotonic() on the Raspberry Pi, recorded at
    the instant each MAVLink message is received by the reader thread.
    Rationale: monotonic() never goes backward (immune to NTP jumps),
    starts at an arbitrary epoch that is stable within a flight, and is
    available on the same Python process that drives the camera pipeline.
    The autopilot's time_boot_ms / time_usec fields are stored in the Pose
    for diagnostics and cross-validation but are NOT used for buffer indexing,
    because the offset between autopilot boot and Pi boot is unknown without
    a calibration step (e.g. GPS PPS comparison).

A2  CLOCK OFFSET (camera ↔ MAVLink receive clock)
    If the camera pipeline timestamps frames from a clock other than
    time.monotonic() — e.g. a hardware trigger referenced to CLOCK_REALTIME
    or a GStreamer pipeline clock — pass:
        clock_offset = time.monotonic() - camera_clock_reference
    measured at the same instant (e.g. at startup against a GPIO pulse or
    GPS PPS edge).  pose_at(camera_t) applies:
        effective_t = camera_t + clock_offset
    so effective_t lands in the buffer's time.monotonic() receive-time base.
    When both clocks are time.monotonic() (the default Pi camera setup),
    leave clock_offset = 0.0.

A3  ALTITUDE REFERENCE
    Despite the legacy field name Pose.alt_agl, telemetry stores
    GLOBAL_POSITION_INT.relative_alt: metres above the ArduPilot Home
    waypoint.  It is NOT a terrain-AGL measurement unless Home is on the
    same ground plane as the target.  Before passing altitude to
    geolocate.pixel_to_gps, the integration layer must convert this value to
    height above the local projection ground plane when terrain/Home
    elevation differs.

A4  GIMBAL FRAME FLAGS
    GIMBAL_DEVICE_ATTITUDE_STATUS.flags is interpreted as follows, translating
    MAVLink's frame terminology to this project's internal "body"/"earth"
    vocabulary exactly once, at this module boundary (see A11):
        bit 64  GIMBAL_DEVICE_FLAGS_YAW_IN_EARTH_FRAME   → "earth" (newer MAVLink)
        bit 32  GIMBAL_DEVICE_FLAGS_YAW_IN_VEHICLE_FRAME → "body"  (newer MAVLink)
        bit 16  GIMBAL_DEVICE_FLAGS_YAW_LOCK              → "earth" (older convention)
        default (none of the above)                       → "body"
    Matches ArduPilot 2024 Copter behaviour.  Verify against actual gimbal
    hardware during integration (see geolocate.py header, section
    "BODY-RELATIVE VS EARTH-RELATIVE GIMBAL TELEMETRY").  Everything past this
    translation - Pose.gimbal_frame, tests, and every downstream module - only
    ever sees "body" or "earth"; MAVLink's own "vehicle" spelling never leaks
    past this function.

A5  GIMBAL QUATERNION ORDER
    GIMBAL_DEVICE_ATTITUDE_STATUS.q is wxyz (MAVLink spec §11.4).
    Pose.gimbal_quat stores the same (w, x, y, z) tuple unchanged.
    Conversion to xyzw only happens internally inside _slerp_quat for scipy.

A6  GIMBAL FALLBACK — MOUNT_ORIENTATION
    If GIMBAL_DEVICE_ATTITUDE_STATUS is absent from the stream, the reader
    falls back to MOUNT_ORIENTATION (roll/pitch/yaw, degrees, body-
    relative by convention, no frame flags).  A ZYX Euler quaternion is
    synthesised and gimbal_frame = "body".
    LIMITATION: MOUNT_ORIENTATION does not carry frame-flag metadata.
    If the gimbal is in earth-locked mode, the downstream geo-location will
    double-count vehicle attitude.  This matches geolocate.py ASSUMPTION #4;
    verify during hardware integration.

A7  OUT-OF-ORDER SAMPLES
    Because the reader uses Pi receive-time (monotonic), samples it produces
    are naturally ordered.  add_sample uses bisect.insort for sorted
    insertion so that synthetic samples injected by tests in arbitrary order
    are handled correctly.  Duplicate timestamps insert after any existing
    sample with the same t (right-side bisect).

A8  OUT-OF-RANGE POLICY
    pose_at returns None when t (after clock_offset) falls outside the
    buffered time range.  No extrapolation is performed.  Callers must drop
    the corresponding frame.

A9  GPS_RAW_INT UNITS
    GPS_RAW_INT.h_acc is uint32 in millimetres (0 or UINT32_MAX = unknown).
    Pose.h_acc_m stores the value converted to metres, or None if unknown.

A10 ATTITUDE SAMPLE TRIGGER
    A new Pose sample is written to the buffer on every ATTITUDE message,
    incorporating the most-recently-received values for position, gimbal,
    and GPS quality.  GLOBAL_POSITION_INT and GPS_RAW_INT only update
    internal state; they do not trigger a sample by themselves.

A11 TELEMETRY / GEOLOCATION BOUNDARY
    This module only produces timestamped Pose samples.  It does not route
    BODY_RELATIVE vs EARTH_RELATIVE gimbal data into a geolocation path and
    does not convert gimbal quaternions to Euler angles.  That conversion
    and routing belongs in the integration layer (for example mission.py),
    where hardware-validated frame semantics are available.

A12 CSV REPLAY CONNECTIONS ARE NOT SUPPORTED
    TelemetryBuffer.start() rejects any connection string starting with
    "csv:" (case-insensitive, matching pymavlink's own dispatch rule in
    mavutil.mavlink_connection()) by raising ValueError immediately and
    synchronously in start() -- before the reader thread is even spawned.
    Rationale: pymavlink.CSVReader.CSVReader (one of mavlink_connection's
    possible return types) has no close() method at all, and its
    recv_match() has no timeout keyword argument, so _mavlink_loop's
    conn.recv_match(..., timeout=1.0) call raises TypeError on the very
    first read.  Before this guard existed, that TypeError was swallowed by
    the outer reconnect loop's broad exception handling, and the resulting
    close() AttributeError was swallowed by a blind
    contextlib.suppress(Exception) -- so a CSV connection string silently
    ingested zero pose samples forever while leaking one open file
    descriptor per reconnect attempt (CSVReader opens a real file handle in
    its constructor).  If offline CSV/log replay ever becomes a real
    requirement, do not just widen this guard: give CSVReader a real close
    path (closing its underlying `.f` file handle directly, since it
    exposes no public close()) and make the `timeout` kwarg conditional on
    the connection type, since CSVReader.recv_match() doesn't accept it
    either.
    RELATED, NOT GUARDED: pymavlink.DFReader.DFReader_binary (returned for
    ".bin"/".px4log" dataflash-log connection strings) has the same
    recv_match()-has-no-timeout incompatibility.  It does implement
    close(), so it doesn't leak a file descriptor, but it would still
    silently ingest zero samples.  Out of scope here since nothing in this
    project passes a dataflash-log connection string, but worth guarding
    the same way if that ever changes.

================================================================================
"""

from __future__ import annotations

import bisect
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol, cast

from scipy.spatial.transform import Rotation, Slerp

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GIMBAL_DEVICE_FLAGS bitmask constants (MAVLink common.xml, 2024)
# ASSUMPTION A4: these bit positions are used to decode the gimbal frame.
# Names mirror the MAVLink spec exactly (external protocol); the values they
# decode to are translated to this project's "body"/"earth" vocabulary by
# _gimbal_frame_from_flags, immediately below.
_GIMBAL_FLAG_YAW_LOCK = 16           # bit 16 — yaw locked to earth (older)
_GIMBAL_FLAG_YAW_IN_VEHICLE_FRAME = 32  # bit 32 — explicit vehicle/body frame (newer)
_GIMBAL_FLAG_YAW_IN_EARTH_FRAME = 64   # bit 64 — explicit earth frame (newer)

_GPS_UNKNOWN_HACC = (0, 0xFFFFFFFF)  # values that mean "h_acc unknown"


# ---------------------------------------------------------------------------
# Data model

@dataclass(frozen=True, slots=True)
class Pose:
    """One timestamped snapshot of drone + gimbal pose.

    All angles are in radians; NED conventions match geolocate.py.

    Attributes:
        t:            Buffer time base — time.monotonic() on the Pi at
                      MAVLink message-receive time (seconds).  # ASSUMPTION A1
        lat:          Drone latitude, decimal degrees.
        lon:          Drone longitude, decimal degrees.
        alt_agl:      Drone altitude above ArduPilot Home, metres, from
                      GLOBAL_POSITION_INT.relative_alt.  Despite the field
                      name, this is not true terrain AGL unless Home is on
                      the local projection ground plane.  # ASSUMPTION A3
        roll:         Drone roll, radians, NED body frame.
        pitch:        Drone pitch, radians, NED body frame.
        yaw:          Drone heading, radians, 0 = North, clockwise positive.
        gimbal_quat:  Gimbal attitude quaternion (w, x, y, z) or None.
                      # ASSUMPTION A5 — wxyz order (MAVLink spec).
        gimbal_frame: "body" or "earth", determined from GIMBAL_DEVICE_FLAGS.
                      None when gimbal data is unavailable.  # ASSUMPTION A4
        fix_type:     GPS fix type (0-8, GPS_RAW_INT.fix_type) or None.
        h_acc_m:      Horizontal GPS accuracy, metres, or None.  # ASSUMPTION A9
    """

    t: float
    lat: float
    lon: float
    alt_agl: float
    roll: float
    pitch: float
    yaw: float
    gimbal_quat: tuple[float, float, float, float] | None
    gimbal_frame: str | None
    fix_type: int | None
    h_acc_m: float | None


# ---------------------------------------------------------------------------
# Internal helpers

def _gimbal_frame_from_flags(flags: int) -> str:
    """Return "earth" or "body" from a GIMBAL_DEVICE_FLAGS bitmask.

    This is the module boundary (ASSUMPTION A4/A11) where MAVLink's
    "vehicle"-frame terminology is translated, exactly once, to this
    project's internal "body" vocabulary.  Priority: newer explicit-frame
    bits (32/64) override the older YAW_LOCK bit (16).
    """
    if flags & _GIMBAL_FLAG_YAW_IN_EARTH_FRAME:
        return "earth"
    if flags & _GIMBAL_FLAG_YAW_IN_VEHICLE_FRAME:
        return "body"
    if flags & _GIMBAL_FLAG_YAW_LOCK:
        return "earth"
    return "body"


def _slerp_quat(
    q0: tuple[float, float, float, float],
    q1: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float]:
    """Spherical-linear interpolation between two unit quaternions.

    Args:
        q0: Start quaternion in (w, x, y, z) order.  # ASSUMPTION A5
        q1: End quaternion in (w, x, y, z) order.
        alpha: Interpolation parameter in [0, 1].

    Returns:
        Interpolated unit quaternion in (w, x, y, z) order.
    """
    # scipy Rotation.from_quat expects (x, y, z, w)
    rotations = Rotation.from_quat([
        [q0[1], q0[2], q0[3], q0[0]],
        [q1[1], q1[2], q1[3], q1[0]],
    ])
    slerp_fn = Slerp([0.0, 1.0], rotations)
    xyzw = slerp_fn([alpha]).as_quat()[0]
    return (float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2]))


def _interp_angle(ang_a: float, ang_b: float, alpha: float) -> float:
    """Interpolate between two angles using the shortest arc, result in [-π, π].

    Handles the ±π wrap correctly: yaw 179° → −179° interpolates through 180°,
    not through 0°.
    """
    diff = (ang_b - ang_a + math.pi) % (2.0 * math.pi) - math.pi
    raw = ang_a + alpha * diff
    return (raw + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Public interpolation function

def interpolate(a: Pose, b: Pose, t: float) -> Pose:
    """Interpolate between two bracketing pose samples at time *t*.

    Linear interpolation for position (lat, lon, alt_agl) and angles
    (roll, pitch, yaw) with shortest-arc wrap for all three angles.
    Quaternion gimbal attitude is interpolated via SLERP only when adjacent
    samples have the same gimbal reference frame.  If the frame tag changes
    across the interval, the newer sample's quaternion is used instead of
    performing an invalid SLERP between incompatible frames.  Discrete fields
    (fix_type, gimbal_frame) are taken from the newer sample *b*.

    Args:
        a: Earlier sample; must satisfy ``a.t <= t``.
        b: Later sample; must satisfy ``b.t >= t``.
        t: Desired time in the buffer's time base.

    Returns:
        A new :class:`Pose` at time *t*.
    """
    if b.t <= a.t:
        return a
    alpha = (t - a.t) / (b.t - a.t)

    # Linear fields
    lat = a.lat + alpha * (b.lat - a.lat)
    lon = a.lon + alpha * (b.lon - a.lon)
    alt_agl = a.alt_agl + alpha * (b.alt_agl - a.alt_agl)

    # Shortest-arc angle interpolation
    roll = _interp_angle(a.roll, b.roll, alpha)
    pitch = _interp_angle(a.pitch, b.pitch, alpha)
    yaw = _interp_angle(a.yaw, b.yaw, alpha)

    # Gimbal quaternion SLERP.  Reference-frame changes are discontinuities:
    # BODY_RELATIVE and EARTH_RELATIVE quaternions do not live in the same
    # frame, so interpolating between them would be physically invalid.
    if a.gimbal_quat is not None and b.gimbal_quat is not None:
        if a.gimbal_frame == b.gimbal_frame:
            gimbal_quat: tuple[float, float, float, float] | None = _slerp_quat(
                a.gimbal_quat, b.gimbal_quat, alpha
            )
        else:
            gimbal_quat = b.gimbal_quat
    else:
        gimbal_quat = b.gimbal_quat if b.gimbal_quat is not None else a.gimbal_quat

    # Discrete fields: prefer newer sample
    gimbal_frame = b.gimbal_frame if b.gimbal_frame is not None else a.gimbal_frame
    fix_type = b.fix_type if b.fix_type is not None else a.fix_type

    # h_acc: interpolate when both are known; otherwise fall back to available
    if a.h_acc_m is not None and b.h_acc_m is not None:
        h_acc_m: float | None = a.h_acc_m + alpha * (b.h_acc_m - a.h_acc_m)
    else:
        h_acc_m = b.h_acc_m if b.h_acc_m is not None else a.h_acc_m

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
# MAVLink connection typing
#
# pymavlink.mavutil.mavlink_connection() is untyped; pyright infers its return
# as the union of every concrete connection class it can construct (mavudp,
# mavtcp, mavserial, mavmcast, mavwebsocket, ..., plus the log-replay readers
# pymavlink.CSVReader.CSVReader and pymavlink.DFReader.DFReader_binary). Every
# live-stream class subclasses pymavlink's own `mavfile` and satisfies the
# Protocol below; CSVReader does not (no close(), and recv_match() has no
# timeout kwarg) -- TelemetryBuffer.start() rejects "csv:" connection strings
# before one is ever constructed (ASSUMPTION A12), so the cast() in
# _reader_loop is sound for every connection string this class actually
# accepts. pymavlink ships no py.typed marker, so recv_match()'s message
# return is typed Any: each MAVLink message type carries different dynamic
# fields (.lat, .roll, .q, ...) that no static stub covers.

class MavlinkConnectionLike(Protocol):
    """Structural surface `_mavlink_loop`/`_reader_loop` actually call."""

    def recv_match(
        self,
        type: list[str] | None = None,
        blocking: bool = False,
        timeout: float | None = None,
    ) -> Any: ...

    def close(self) -> None: ...


def _is_csv_replay_connection(connection_str: str) -> bool:
    """True if *connection_str* would resolve to pymavlink's CSVReader.

    Mirrors pymavlink's own dispatch rule, verified against the installed
    package (mavutil.py, mavlink_connection()): ``device.lower().startswith
    ("csv:")``.  # ASSUMPTION A12
    """
    return connection_str.lower().startswith("csv:")


# ---------------------------------------------------------------------------
# Ring buffer and MAVLink reader

class TelemetryBuffer:
    """Thread-safe ring buffer of timestamped drone pose samples.

    A background reader thread ingests MAVLink messages and calls
    :meth:`add_sample`.  The vision thread calls :meth:`pose_at` to retrieve
    an interpolated pose for an arbitrary frame-capture timestamp.

    Time base: ``time.monotonic()`` on the Raspberry Pi recorded at
    MAVLink message-receive time.  # ASSUMPTION A1

    Args:
        capacity_seconds: How many seconds of history to retain.  Older
            samples are evicted automatically on each insert.
        clock_offset: Added to every ``t`` argument in :meth:`pose_at`
            before lookup.  Use when camera frame timestamps come from a
            different clock than ``time.monotonic()``; compute as
            ``time.monotonic() - camera_clock`` at a shared sync instant.
            # ASSUMPTION A2
    """

    def __init__(
        self,
        capacity_seconds: float = 5.0,
        clock_offset: float = 0.0,
    ) -> None:
        self._capacity = float(capacity_seconds)
        self.clock_offset = float(clock_offset)

        # Buffer storage: parallel sorted lists (sorted by Pose.t ascending).
        # Protected by _lock for thread safety.
        self._buf: list[Pose] = []
        self._times: list[float] = []
        self._lock = threading.Lock()

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._connection_str: str = ""

    # ------------------------------------------------------------------
    # Public interface

    def add_sample(self, pose: Pose) -> None:
        """Insert a pose sample into the buffer.

        Thread-safe; called by the reader thread and also by tests directly.
        Maintains buffer sorted by ``pose.t``.  Evicts samples older than
        ``capacity_seconds`` relative to the newest sample.

        Out-of-order inserts (e.g. from test code) are handled correctly via
        bisect; from the reader thread they are rare because Pi receive-time
        is monotonic.  # ASSUMPTION A7
        """
        with self._lock:
            idx = bisect.bisect_right(self._times, pose.t)
            self._buf.insert(idx, pose)
            self._times.insert(idx, pose.t)

            # Evict samples that fall outside the capacity window.
            newest_t = self._times[-1]
            cutoff = newest_t - self._capacity
            while self._times and self._times[0] < cutoff:
                self._buf.pop(0)
                self._times.pop(0)

    def pose_at(self, t: float) -> Pose | None:
        """Return the pose interpolated to time *t*, or ``None`` if out of range.

        Args:
            t: Frame-capture timestamp.  With the default ``clock_offset=0``,
               this is already in the buffer time base (``time.monotonic()``
               on the Pi).  Otherwise, ``clock_offset`` converts the camera
               clock into that buffer time base before lookup.  # ASSUMPTION A2

        Returns:
            Interpolated :class:`Pose`, or ``None`` when *t* falls outside
            the buffered range.  # ASSUMPTION A8 — no extrapolation.
        """
        effective_t = t + self.clock_offset

        with self._lock:
            if not self._buf:
                return None
            if effective_t < self._times[0] or effective_t > self._times[-1]:
                return None

            idx = bisect.bisect_right(self._times, effective_t)

            # Exact match: the sample just left of the bisect position.
            if self._times[idx - 1] == effective_t:
                return self._buf[idx - 1]

            # Bracket: a = self._buf[idx-1], b = self._buf[idx].
            # idx > 0 is guaranteed (effective_t >= times[0] checked above).
            # idx < len is guaranteed (effective_t < times[-1] checked above,
            # so it cannot equal the last element after a non-exact bisect).
            a = self._buf[idx - 1]
            b = self._buf[idx]

        # Interpolate outside the lock; Pose is frozen so a/b are safe.
        return interpolate(a, b, effective_t)

    @property
    def latest_age(self) -> float | None:
        """Seconds since the most recent sample; ``None`` if buffer is empty.

        A watchdog can use this to detect a stalled link: if ``latest_age``
        exceeds a threshold (e.g. 2 s), the reader thread or MAVLink source
        has likely failed.
        """
        with self._lock:
            if not self._times:
                return None
            return time.monotonic() - self._times[-1]

    def start(self, connection_str: str) -> None:
        """Spawn the background MAVLink reader thread.

        Args:
            connection_str: Any live-stream pymavlink connection string, e.g.
                ``"udpin:0.0.0.0:14550"``, ``"serial:/dev/ttyAMA0:57600"``,
                ``"tcp:127.0.0.1:5760"``.  CSV replay strings (``"csv:..."``)
                are rejected -- see ASSUMPTION A12.

        Raises:
            RuntimeError: If the buffer is already running.
            ValueError: If *connection_str* is a CSV replay connection string.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("TelemetryBuffer.start() called while already running")
        if _is_csv_replay_connection(connection_str):
            raise ValueError(
                "CSV replay connections are not supported by this reader "
                f"loop (got {connection_str!r}); use a UDP/TCP MAVLink "
                'source, e.g. "udpin:0.0.0.0:14550". '
                "pymavlink.CSVReader.CSVReader has no close() method and its "
                "recv_match() has no timeout keyword argument, so "
                "_mavlink_loop cannot service it.  See ASSUMPTION A12."
            )
        self._connection_str = connection_str
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="mavlink-telemetry",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the reader thread to exit and wait for it to finish."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    # ------------------------------------------------------------------
    # Internal reader

    def _reader_loop(self) -> None:
        """Outer reconnect loop; runs in the background thread."""
        from pymavlink import mavutil  # lazy import: not needed by tests

        backoff = 1.0
        while not self._stop.is_set():
            conn: MavlinkConnectionLike | None = None
            try:
                # mavlink_connection()'s return type is a broad union (see
                # MavlinkConnectionLike above); start() has already rejected
                # the one branch (CSVReader) that doesn't satisfy it, so this
                # cast reflects what every remaining connection string
                # actually returns.
                conn = cast(
                    MavlinkConnectionLike,
                    mavutil.mavlink_connection(self._connection_str),
                )
                _log.info("MAVLink connected: %s", self._connection_str)
                backoff = 1.0
                self._mavlink_loop(conn)
            except Exception:
                _log.exception(
                    "MAVLink reader error (backoff=%.1fs); reconnecting", backoff
                )
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except OSError:
                        # Best-effort cleanup: the socket/port may already be
                        # torn down (e.g. after the connection-reset that
                        # triggered this cleanup in the first place). Any
                        # other exception type here is unexpected for a
                        # MavlinkConnectionLike connection and should surface
                        # loudly rather than be swallowed.
                        _log.debug(
                            "MAVLink conn.close() failed; ignoring", exc_info=True
                        )

            if not self._stop.is_set():
                self._stop.wait(backoff)
                backoff = min(backoff * 2.0, 30.0)

    def _mavlink_loop(self, conn: MavlinkConnectionLike) -> None:
        """Inner message-receive loop; runs until connection error or stop."""
        _WANTED = [
            "GLOBAL_POSITION_INT",
            "ATTITUDE",
            "GIMBAL_DEVICE_ATTITUDE_STATUS",
            "MOUNT_ORIENTATION",
            "GPS_RAW_INT",
        ]

        # Fused per-connection state; updated on each relevant message.
        # None means "not yet received".  # ASSUMPTION A10
        lat: float | None = None
        lon: float | None = None
        alt_agl: float | None = None
        gimbal_quat: tuple[float, float, float, float] | None = None
        gimbal_frame: str | None = None
        fix_type: int | None = None
        h_acc_m: float | None = None
        has_gimbal_device_msg = False
        warned_mount = False

        while not self._stop.is_set():
            msg = conn.recv_match(type=_WANTED, blocking=True, timeout=1.0)
            if msg is None:
                continue

            mtype: str = msg.get_type()
            t_recv = time.monotonic()  # ASSUMPTION A1: receive time is authoritative

            try:
                if mtype == "GLOBAL_POSITION_INT":
                    # degE7 -> degrees; relative_alt mm above Home -> metres.
                    # Despite the field name, alt_agl is not terrain AGL.
                    # ASSUMPTION A1, A3
                    lat = msg.lat * 1e-7
                    lon = msg.lon * 1e-7
                    alt_agl = msg.relative_alt * 1e-3

                elif mtype == "ATTITUDE":
                    # Snapshot the full fused state as one Pose sample.  # ASSUMPTION A10
                    if lat is None or lon is None or alt_agl is None:
                        continue  # haven't received a position yet; skip
                    pose = Pose(
                        t=t_recv,
                        lat=lat,
                        lon=lon,
                        alt_agl=alt_agl,
                        roll=msg.roll,    # radians, ArduPilot body frame
                        pitch=msg.pitch,
                        yaw=msg.yaw,
                        gimbal_quat=gimbal_quat,
                        gimbal_frame=gimbal_frame,
                        fix_type=fix_type,
                        h_acc_m=h_acc_m,
                    )
                    self.add_sample(pose)

                elif mtype == "GIMBAL_DEVICE_ATTITUDE_STATUS":
                    has_gimbal_device_msg = True
                    q = msg.q  # float[4], wxyz — # ASSUMPTION A5
                    gimbal_quat = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
                    gimbal_frame = _gimbal_frame_from_flags(int(msg.flags))  # ASSUMPTION A4

                elif mtype == "MOUNT_ORIENTATION":
                    # Fallback: only use when GIMBAL_DEVICE_ATTITUDE_STATUS has not
                    # been seen on this connection.  # ASSUMPTION A6
                    if has_gimbal_device_msg:
                        continue
                    if not warned_mount:
                        _log.warning(
                            "GIMBAL_DEVICE_ATTITUDE_STATUS not seen; using "
                            "MOUNT_ORIENTATION fallback (body-relative assumed). "
                            "See telemetry.py ASSUMPTION A6."
                        )
                        warned_mount = True
                    # MOUNT_ORIENTATION angles are in degrees, body-relative
                    r = Rotation.from_euler(
                        "ZYX",
                        [math.radians(msg.yaw), math.radians(msg.pitch), math.radians(msg.roll)],
                    )
                    xyzw = r.as_quat()
                    gimbal_quat = (float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2]))
                    gimbal_frame = "body"

                elif mtype == "GPS_RAW_INT":
                    fix_type = int(msg.fix_type)
                    raw_hacc = int(msg.h_acc)
                    # ASSUMPTION A9: 0 and UINT32_MAX both mean "unknown"
                    h_acc_m = None if raw_hacc in _GPS_UNKNOWN_HACC else raw_hacc * 1e-3

            except Exception:
                _log.debug(
                    "Malformed or partial %s message; skipping", mtype, exc_info=True
                )
