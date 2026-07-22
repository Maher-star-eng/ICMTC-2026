# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   06/06/2026

"""geolocate.py - UAV flag pixel -> GPS geo-location (ICMTC 2026, Mission 2).

Pure computation module: pixel coordinates + drone/gimbal pose + camera
intrinsics -> a single flag GPS coordinate. No hardware access, no I/O, no
network calls.

================================================================================
CONVENTIONS AND ASSUMPTIONS (read this before changing any sign or axis order)
================================================================================

This module follows the ArduPilot/aerospace NED convention throughout. Every
place where the convention was ambiguous, the choice below was made
deliberately and is marked with a matching `# ASSUMPTION:` comment at its
point of use so a reviewer can check it against ArduPilot.

1. WORLD FRAME (NED): North = +X, East = +Y, Down = +Z. Down is positive, so
   a ray pointing *into the ground* has a positive Down component. This is
   the literal definition of NED and matches ArduPilot's local-position /
   EKF output frame. A drone altitude above ground is still a positive
   height, but the ground plane below the drone has local NED coordinate
   Down = +altitude_agl.

2. BODY FRAME: X = forward (nose), Y = right (starboard), Z = down (belly).
   Standard ArduPilot/PX4 body-fixed frame.

3. ATTITUDE ROTATION ORDER (body -> world): standard aerospace 3-2-1 Euler
   sequence -- yaw about world Z, then pitch about the new Y, then roll
   about the newest X -- i.e. R = Rz(yaw) @ Ry(pitch) @ Rx(roll). Positive
   roll = right wing down, positive pitch = nose up, positive yaw =
   clockwise when viewed from above (compass heading, 0 = North,
   +90 deg = East). Implemented with
   `Rotation.from_euler('ZYX', [yaw, pitch, roll])`, which scipy composes as
   Rz(yaw) @ Ry(pitch) @ Rx(roll) for this intrinsic axis order (verified
   against a hand-built Rz@Ry@Rx matrix).

4. GIMBAL ROTATION (camera -> body): the gimbal is treated as a small
   "vehicle" bolted to the body, using the *same* ZYX convention as #3:
   R_gimbal = Rz(gimbal_yaw) @ Ry(gimbal_pitch) @ Rx(gimbal_roll).
   # ASSUMPTION: gimbal angles are defined relative to the body frame (not
   the world frame), matching how SIYI/typical 3-axis gimbals report angles
   in "follow" mode. The rotation chain is therefore:

       Camera optical frame
         -> body-aligned camera frame (fixed axis relabelling)
         -> gimbal rotation (body-relative)
         -> vehicle body frame
         -> world NED frame

   If future hardware reports Earth-frame/camera-world attitude instead
   (for example MAVLink GIMBAL_DEVICE_ATTITUDE_STATUS with Earth-frame
   flags), this assumption is no longer valid: applying vehicle body
   attitude again would double-count aircraft attitude. Verify this during
   hardware integration before using live telemetry.

5. CAMERA-ZERO ORIENTATION: # ASSUMPTION: when all three gimbal angles are
   zero, the camera boresight points along body +X (forward, level with the
   horizon) -- the SIYI A8 mini convention (gimbal pitch 0 deg = forward-
   level). Consequently:
   # ASSUMPTION: gimbal_pitch = -pi/2 (-90 deg) is straight down (nadir).
   This is exercised directly by the nadir test case.

6. CAMERA OPTICAL FRAME: standard OpenCV/pinhole convention -- X = image
   right, Y = image down, Z = boresight (forward, out of the lens). This is
   the frame `ray_cam = inv(K) @ [u, v, 1]` naturally lives in.

7. CAMERA-FRAME -> BODY-ALIGNED-FRAME fixed relabelling (applied before the
   gimbal rotation; it defines the gimbal's own "zero" frame in terms of
   body axes): # ASSUMPTION: body_forward = cam_Z, body_right = cam_X,
   body_down = cam_Y (i.e. "up" in the image is "up" relative to the
   vehicle when the gimbal is level).

8. GROUND-INTERSECTION SIGN -- CORRECTED FROM THE LITERAL BRIEF TEXT:
   # ASSUMPTION: the brief's literal formula `t = altitude_agl /
   (-ray_world_down)` (guard: invalid when `ray_world_down >= 0`) is
   inconsistent with the brief's own explicit NED statement that "a
   downward-looking ray has a positive Down component" (see #1). A ray
   that actually points at the ground must have ray_world_down > 0 by that
   definition, so dividing by `-ray_world_down` would yield a negative
   (behind-the-camera) intersection distance for every physically valid
   downward ray. This module implements the sign consistent with the
   explicitly-stated NED convention: `t = down_distance_to_ground /
   ray_world_down`, where `down_distance_to_ground` is the positive
   `altitude_agl` magnitude represented along NED +Z. It guards
   `ray_world_down <= 0` (horizontal or upward ray) as invalid.
   This is also the formula that reproduces the brief's own worked example
   (10 deg gimbal tilt at 50 m AGL -> ~50*tan(10 deg) ~= 8.82 m).

9. METERS-PER-DEGREE LATITUDE: a fixed 111320.0 m/deg, used for both the
   north->latitude and east->longitude (scaled by cos(drone_lat))
   conversions. This matches the brief's `degrees(north_m / 6378137.0)`
   formula to within ~0.0007% (6378137 * pi/180 ~= 111319.49) and exactly
   matches the brief's later explicit "~111320 m/deg lat" instruction, so a
   single constant is used everywhere for self-consistency.

10. FLAT-EARTH / LOCAL-TANGENT-PLANE: valid at the <=~100 m displacement
    scale relevant to this mission (scoring tolerance is 20-60 m); Earth
    curvature error at this scale is negligible.

11. DISTORTION: undistorted via `cv2.undistortPoints` when `dist_coeffs` is
    given AND OpenCV is importable. # ASSUMPTION: if OpenCV is unavailable,
    distortion is silently ignored (a `RuntimeWarning` is emitted) rather
    than raising, since the SIYI A8 mini's residual distortion is small
    relative to the 20 m scoring tolerance at 50-60 m AGL.

12. RAY NORMALIZATION: `pixel_to_ray` and `ray_cam_to_world` return unit
    vectors. This is purely for numerical hygiene -- the ground-intersection
    math in `ray_to_ground` is scale-invariant -- but keeps intermediate
    values well-conditioned.

13. PIXEL CHOICE (caller responsibility, not enforced here): because the
    flag lies flat on the ground, the correct (u, v) to pass is the
    bottom-center of the YOLO bounding box (the true ground-contact point),
    not the box center. See `pixel_to_gps` docstring.

BODY-RELATIVE VS EARTH-RELATIVE GIMBAL TELEMETRY
------------------------------------------------

Body-relative telemetry reports the camera/gimbal attitude as an offset from
the aircraft body, so the world ray must compose both rotations:

    Camera -> Body-aligned -> Gimbal(body-relative) -> Body -> World(NED)

    ray_world = R_body_to_world @ R_gimbal_body @ R_cam_fixed @ ray_cam

Earth-relative telemetry already reports the camera/gimbal attitude in the
world frame, so the body rotation must not be applied again:

    Camera -> Body-aligned -> Gimbal/camera attitude in World(NED)

    ray_world = R_camera_to_world @ R_cam_fixed @ ray_cam

The current public pipeline uses body-relative telemetry by default. The
Earth-relative helper in this module is intentionally opt-in and not selected
automatically.

ALTITUDE REFERENCE
------------------

`altitude_agl` must be the drone height above the local ground plane used for
projection, i.e. the plane the flag lies on. Supplying altitude above Home,
MSL, ellipsoid height, or any other reference can introduce projection error
equal to the reference mismatch scaled by the ray angle.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

#: Meters per degree of latitude (and, scaled by cos(lat), of longitude).
#: See ASSUMPTION #9 above.
METERS_PER_DEG_LAT: float = 111320.0

#: Decimal-degree GPS coordinate, ``(latitude, longitude)``.
GPSCoord = tuple[float, float]

#: Local horizontal NED offset, ``(north_m, east_m)``. The Down component is
#: represented separately by altitude/down-distance values.
LocalNED = tuple[float, float]

#: Image pixel coordinate, ``(u, v)`` = ``(column, row)``.
Pixel = tuple[float, float]

# TODO(hardware-integration): Verify SIYI A8 telemetry reference frame.
# TODO(hardware-integration): Verify MAVLink frame flags before using gimbal data.
# TODO(hardware-integration): Bench-test by tilting the aircraft while holding
# the camera at nadir.
# TODO(hardware-integration): Enable Earth-frame implementation only if telemetry
# is Earth-relative.


class GimbalReferenceFrame(Enum):
    """Reference frame used by gimbal attitude telemetry."""

    BODY_RELATIVE = "body_relative"
    EARTH_RELATIVE = "earth_relative"


@dataclass(frozen=True, slots=True)
class GeolocationConfig:
    """Explicit geolocation assumptions for future hardware integration.

    The current public API uses `BODY_RELATIVE` and does not auto-switch based
    on telemetry. Use `ray_cam_to_world_from_earth_attitude` explicitly if
    hardware validation proves the gimbal reports Earth-relative attitude.
    """

    gimbal_reference_frame: GimbalReferenceFrame = GimbalReferenceFrame.BODY_RELATIVE

    def __post_init__(self) -> None:
        if not isinstance(self.gimbal_reference_frame, GimbalReferenceFrame):
            raise ValueError(
                "gimbal_reference_frame must be a GimbalReferenceFrame value"
            )


#: Active module assumption. Kept explicit so hardware integration has one
#: obvious place to inspect; current behavior remains body-relative.
DEFAULT_GEOLOCATION_CONFIG = GeolocationConfig()


def _validate_finite_scalar(name: str, value: float) -> float:
    """Return `value` as float after checking it is finite."""
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite numeric value; got {value!r}"
        ) from exc

    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return scalar


def _validate_positive_scalar(name: str, value: float) -> float:
    """Return `value` as float after checking it is finite and positive."""
    scalar = _validate_finite_scalar(name, value)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive; got {value!r}")
    return scalar


def _as_finite_float_array(name: str, values: object) -> np.ndarray:
    """Return `values` as a float64 array after lightweight numeric validation."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numeric values")
    return array


def _validate_camera_matrix(K: np.ndarray) -> np.ndarray:
    """Validate and return a camera intrinsics matrix as a numeric array."""
    matrix = _as_finite_float_array("K", K)
    if matrix.shape != (3, 3):
        raise ValueError(
            f"K must be a 3x3 camera intrinsics matrix; got shape {matrix.shape}"
        )

    fx = matrix[0, 0]
    fy = matrix[1, 1]
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(
            f"K focal lengths must be positive; got fx={fx!r}, fy={fy!r}"
        )
    return matrix


def _validate_vector3(name: str, vector: np.ndarray) -> np.ndarray:
    """Validate and return a finite 3-vector."""
    array = _as_finite_float_array(name, vector)
    if array.shape != (3,):
        raise ValueError(
            f"{name} must be a length-3 vector; got shape {array.shape}"
        )

    if np.linalg.norm(array) == 0.0:
        raise ValueError(f"{name} must be non-zero")
    return array


def _validate_dist_coeffs(
    dist_coeffs: Sequence[float] | np.ndarray | None,
) -> np.ndarray | None:
    """Validate optional OpenCV-style distortion coefficients."""
    if dist_coeffs is None:
        return None
    return _as_finite_float_array("dist_coeffs", dist_coeffs)


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Validated camera intrinsics used internally without changing the old API."""

    matrix: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix", _validate_camera_matrix(self.matrix))

    @property
    def fx(self) -> float:
        """Focal length in pixels along image x/columns."""
        return float(self.matrix[0, 0])

    @property
    def fy(self) -> float:
        """Focal length in pixels along image y/rows."""
        return float(self.matrix[1, 1])


@dataclass(frozen=True, slots=True)
class PixelCoordinate:
    """Validated image coordinate, ``u`` column and ``v`` row."""

    u: float
    v: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "u", _validate_finite_scalar("u", self.u))
        object.__setattr__(self, "v", _validate_finite_scalar("v", self.v))


@dataclass(frozen=True, slots=True)
class GimbalPose:
    """Validated gimbal attitude relative to the drone body, radians."""

    pitch: float
    roll: float
    yaw: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "pitch", _validate_finite_scalar("gimbal_pitch", self.pitch)
        )
        object.__setattr__(
            self, "roll", _validate_finite_scalar("gimbal_roll", self.roll)
        )
        object.__setattr__(
            self, "yaw", _validate_finite_scalar("gimbal_yaw", self.yaw)
        )


@dataclass(frozen=True, slots=True)
class DronePose:
    """Validated drone position and attitude at capture time.

    `altitude_agl` remains a positive height above the ground. In the local
    NED intersection math, the ground plane is represented as a positive Down
    distance from the drone because NED defines +Z as Down.
    """

    lat: float
    lon: float
    altitude_agl: float
    roll: float
    pitch: float
    yaw: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "lat", _validate_finite_scalar("drone_lat", self.lat)
        )
        object.__setattr__(
            self, "lon", _validate_finite_scalar("drone_lon", self.lon)
        )
        object.__setattr__(
            self,
            "altitude_agl",
            _validate_positive_scalar("altitude_agl", self.altitude_agl),
        )
        object.__setattr__(self, "roll", _validate_finite_scalar("drone_roll", self.roll))
        object.__setattr__(
            self, "pitch", _validate_finite_scalar("drone_pitch", self.pitch)
        )
        object.__setattr__(self, "yaw", _validate_finite_scalar("drone_yaw", self.yaw))


def build_intrinsics(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Build a pinhole camera intrinsics matrix.

    Args:
        fx: Focal length in pixels along the image x-axis (columns).
        fy: Focal length in pixels along the image y-axis (rows).
        cx: Principal point x-coordinate, pixels (image column).
        cy: Principal point y-coordinate, pixels (image row).

    Returns:
        3x3 ``np.ndarray`` (float64): ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``.

    Raises:
        ValueError: if any input is non-finite, or if `fx`/`fy` is not
            positive.
    """
    fx = _validate_positive_scalar("fx", fx)
    fy = _validate_positive_scalar("fy", fy)
    cx = _validate_finite_scalar("cx", cx)
    cy = _validate_finite_scalar("cy", cy)

    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def pixel_to_ray(
    u: float,
    v: float,
    K: np.ndarray,
    dist_coeffs: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Back-project a pixel to a unit direction ray in the camera optical frame.

    Camera optical frame (module header, convention #6): X = image right,
    Y = image down, Z = boresight (forward, out of the lens). The returned
    ray therefore has a positive Z component for any pixel inside the image.

    Args:
        u: Pixel column (x), 0 at the image's left edge.
        v: Pixel row (y), 0 at the image's top edge.
        K: 3x3 camera intrinsics matrix (see `build_intrinsics`).
        dist_coeffs: Optional OpenCV-style distortion coefficients
            (k1, k2, p1, p2[, k3[, ...]]). If given and OpenCV (`cv2`) is
            importable, the pixel is undistorted via `cv2.undistortPoints`
            before back-projection. If OpenCV is not importable,
            # ASSUMPTION: distortion is ignored and a `RuntimeWarning` is
            raised (module header, convention #11).

    Returns:
        Unit-length ``np.ndarray`` of shape (3,): the ray direction in the
        camera optical frame.

    Raises:
        ValueError: if `K` is not a finite 3x3 camera matrix with positive
            focal lengths, if `K` is singular, or if any supplied numeric
            input is non-finite.
    """
    pixel = PixelCoordinate(u, v)
    intrinsics = CameraIntrinsics(K)
    dist_coeffs = _validate_dist_coeffs(dist_coeffs)

    if dist_coeffs is not None:
        try:
            import cv2  # type: ignore[import-not-found]

            pts = np.array([[[pixel.u, pixel.v]]], dtype=np.float64)
            undistorted = cv2.undistortPoints(pts, intrinsics.matrix, dist_coeffs)
            x, y = undistorted[0, 0]
        except ImportError:
            warnings.warn(
                "dist_coeffs given but OpenCV (cv2) is not installed; "
                "distortion is being ignored (see geolocate.py header, "
                "convention #11).",
                RuntimeWarning,
                stacklevel=2,
            )
            K_inv = _invert_camera_matrix(intrinsics.matrix)
            x, y, _ = K_inv @ np.array([pixel.u, pixel.v, 1.0])
    else:
        K_inv = _invert_camera_matrix(intrinsics.matrix)
        x, y, _ = K_inv @ np.array([pixel.u, pixel.v, 1.0])

    ray_cam = np.array([x, y, 1.0], dtype=np.float64)
    return ray_cam / np.linalg.norm(ray_cam)


def _invert_camera_matrix(K: np.ndarray) -> np.ndarray:
    """Invert a validated camera matrix and raise `ValueError` if singular."""
    try:
        return np.linalg.inv(K)
    except np.linalg.LinAlgError as exc:
        raise ValueError("K must be invertible for pixel back-projection") from exc


def _attitude_rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Build a standard aerospace 3-2-1 (yaw, pitch, roll) rotation matrix.

    R = Rz(yaw) @ Ry(pitch) @ Rx(roll), built via
    `Rotation.from_euler('ZYX', [yaw, pitch, roll])` (module header,
    convention #3). Used for both the drone body->world attitude and the
    gimbal->body-aligned-frame rotation (convention #4), since both follow
    the same ZYX convention.

    Args:
        yaw: Rotation about the reference frame's Z-axis (Down), radians.
        pitch: Rotation about the resulting Y-axis, radians.
        roll: Rotation about the resulting X-axis, radians.

    Returns:
        3x3 rotation matrix mapping a vector from the rotated frame into the
        reference frame.
    """
    yaw = _validate_finite_scalar("yaw", yaw)
    pitch = _validate_finite_scalar("pitch", pitch)
    roll = _validate_finite_scalar("roll", roll)
    return Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()


#: Fixed relabelling from camera optical axes to body-aligned axes at
#: gimbal-zero (module header, convention #7): body_fwd = cam_Z,
#: body_right = cam_X, body_down = cam_Y.
_CAM_OPTICAL_TO_BODY_ALIGNED = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def ray_cam_to_world(
    ray_cam: np.ndarray,
    gimbal_pitch: float,
    gimbal_roll: float,
    gimbal_yaw: float,
    drone_roll: float,
    drone_pitch: float,
    drone_yaw: float,
) -> np.ndarray:
    """Rotate a camera-frame ray into the world (NED) frame.

    Chain: ray_world = R_body_to_world @ R_cam_to_body @ ray_cam.
    This is the current production path and implements
    `DEFAULT_GEOLOCATION_CONFIG.gimbal_reference_frame == BODY_RELATIVE`.

    `R_cam_to_body` is built in two steps (module header, conventions
    #4-#7): first a fixed relabelling from the camera optical frame (X
    right, Y down, Z forward) to a body-aligned frame where the boresight
    is body +X (forward) at gimbal-zero, then the gimbal's own ZYX rotation
    (gimbal_yaw, gimbal_pitch, gimbal_roll) applied in that body-aligned
    frame. # ASSUMPTION: gimbal_pitch = -pi/2 is straight down (nadir).

    `R_body_to_world` is the standard 3-2-1 attitude rotation (module
    header, convention #3) built from (drone_yaw, drone_pitch, drone_roll).
    If gimbal telemetry is Earth-relative, do not pass those angles here;
    use `ray_cam_to_world_from_earth_attitude` explicitly after validating
    the hardware telemetry frame.

    Args:
        ray_cam: Direction ray in the camera optical frame, shape (3,) --
            see `pixel_to_ray`.
        gimbal_pitch: Gimbal pitch relative to the body, radians. Positive
            = camera boresight tilts up; -pi/2 = straight down.
        gimbal_roll: Gimbal roll relative to the body, radians.
        gimbal_yaw: Gimbal yaw relative to the body, radians.
        drone_roll: Drone roll (NED body frame), radians. Positive = right
            wing down.
        drone_pitch: Drone pitch (NED body frame), radians. Positive = nose
            up.
        drone_yaw: Drone yaw / heading (NED), radians. 0 = North, positive
            = clockwise toward East (compass convention).

    Returns:
        Unit-length ``np.ndarray`` of shape (3,): the ray direction in the
        world NED frame, components (north, east, down).

    Raises:
        ValueError: if `ray_cam` is not a finite length-3 vector or any angle
            is non-finite.
    """
    if (
        DEFAULT_GEOLOCATION_CONFIG.gimbal_reference_frame
        is not GimbalReferenceFrame.BODY_RELATIVE
    ):
        raise ValueError(
            "ray_cam_to_world uses body-relative gimbal telemetry; call "
            "ray_cam_to_world_from_earth_attitude explicitly for "
            "Earth-relative telemetry."
        )

    ray_cam = _validate_vector3("ray_cam", ray_cam)
    gimbal = GimbalPose(gimbal_pitch, gimbal_roll, gimbal_yaw)
    drone_roll = _validate_finite_scalar("drone_roll", drone_roll)
    drone_pitch = _validate_finite_scalar("drone_pitch", drone_pitch)
    drone_yaw = _validate_finite_scalar("drone_yaw", drone_yaw)

    R_gimbal = _attitude_rotation(gimbal.yaw, gimbal.pitch, gimbal.roll)
    R_cam_to_body = R_gimbal @ _CAM_OPTICAL_TO_BODY_ALIGNED
    R_body_to_world = _attitude_rotation(drone_yaw, drone_pitch, drone_roll)

    ray_world = R_body_to_world @ (R_cam_to_body @ ray_cam)
    return ray_world / np.linalg.norm(ray_world)


def ray_cam_to_world_from_earth_attitude(
    ray_cam: np.ndarray,
    camera_pitch: float,
    camera_roll: float,
    camera_yaw: float,
) -> np.ndarray:
    """Rotate a camera ray into world NED using Earth-relative camera attitude.

    This optional compatibility helper is not used by `pixel_to_gps` or the
    current public pipeline. It exists for future hardware integrations where
    gimbal/camera telemetry is verified to be Earth-relative, such as a
    MAVLink source whose frame flags indicate a world-frame camera attitude.

    Chain: ray_world = R_camera_to_world @ R_cam_fixed @ ray_cam.

    This bypasses vehicle body attitude composition entirely. Use it only
    when `camera_pitch`, `camera_roll`, and `camera_yaw` already describe the
    camera's body-aligned frame relative to world NED using the same ZYX
    convention documented in the module header. If telemetry provides an
    optical-frame quaternion directly, adapt it before calling this helper.

    Args:
        ray_cam: Direction ray in the camera optical frame, shape (3,) --
            see `pixel_to_ray`.
        camera_pitch: Earth-relative camera pitch, radians.
        camera_roll: Earth-relative camera roll, radians.
        camera_yaw: Earth-relative camera yaw / heading, radians.

    Returns:
        Unit-length ``np.ndarray`` of shape (3,): the ray direction in the
        world NED frame, components (north, east, down).

    Raises:
        ValueError: if `ray_cam` is not a finite length-3 vector or any angle
            is non-finite.
    """
    ray_cam = _validate_vector3("ray_cam", ray_cam)
    camera_pitch = _validate_finite_scalar("camera_pitch", camera_pitch)
    camera_roll = _validate_finite_scalar("camera_roll", camera_roll)
    camera_yaw = _validate_finite_scalar("camera_yaw", camera_yaw)

    R_camera_to_world = _attitude_rotation(camera_yaw, camera_pitch, camera_roll)
    ray_world = R_camera_to_world @ (_CAM_OPTICAL_TO_BODY_ALIGNED @ ray_cam)
    return ray_world / np.linalg.norm(ray_world)


def ray_to_ground(ray_world: np.ndarray, altitude_agl: float) -> LocalNED:
    """Intersect a world-frame ray with the flat ground plane.

    The ground plane is `Down = altitude_agl` in a local NED frame centered
    on the drone (drone at local (0, 0, 0); the ground is `altitude_agl`
    meters below). `altitude_agl` is a positive height above ground in the
    caller's terms, and the same magnitude becomes a positive Down distance
    inside the NED computation because NED defines +Z as Down.

    # ASSUMPTION: see module header, convention #8 -- implemented as
    `t = altitude_agl / ray_world_down`, valid only when `ray_world_down >
    0` (the ray actually points down at the ground).

    Args:
        ray_world: Direction ray in the world NED frame, shape (3,) =
            (north, east, down) components.
        altitude_agl: Drone altitude above the (flat) ground plane, meters;
            must be positive.

    Returns:
        ``(north_m, east_m)``: ground intersection point relative to the
        drone's horizontal position, meters.

    Raises:
        ValueError: if `ray_world[2]` (Down component) is <= 0, i.e. the ray
            is horizontal or points upward and never intersects the ground;
            also if `altitude_agl` is not positive.
    """
    ray_world = _validate_vector3("ray_world", ray_world)
    down_distance_to_ground = _validate_positive_scalar("altitude_agl", altitude_agl)

    down_component = ray_world[2]
    if down_component <= 0.0:
        raise ValueError(
            f"ray_world has Down component {down_component!r} <= 0; ray is "
            "horizontal or pointing upward and never intersects the ground."
        )
    scale_to_ground = down_distance_to_ground / down_component
    north_m = scale_to_ground * ray_world[0]
    east_m = scale_to_ground * ray_world[1]
    return float(north_m), float(east_m)


def local_ned_to_gps(
    north_m: float, east_m: float, drone_lat: float, drone_lon: float
) -> GPSCoord:
    """Convert a local NED (north, east) offset to absolute GPS coordinates.

    Flat-earth approximation (module header, convention #10) using a fixed
    111320.0 m/deg for latitude, scaled by cos(drone_lat) for longitude.

    Args:
        north_m: Northward offset from the drone, meters (south if
            negative).
        east_m: Eastward offset from the drone, meters (west if negative).
        drone_lat: Drone latitude, decimal degrees.
        drone_lon: Drone longitude, decimal degrees.

    Returns:
        ``(lat, lon)`` in decimal degrees.

    Raises:
        ValueError: if any input is non-finite.
    """
    north_m = _validate_finite_scalar("north_m", north_m)
    east_m = _validate_finite_scalar("east_m", east_m)
    drone_lat = _validate_finite_scalar("drone_lat", drone_lat)
    drone_lon = _validate_finite_scalar("drone_lon", drone_lon)

    lat = drone_lat + north_m / METERS_PER_DEG_LAT
    lon = drone_lon + east_m / (METERS_PER_DEG_LAT * np.cos(np.radians(drone_lat)))
    return float(lat), float(lon)


def pixel_to_gps(
    u: float,
    v: float,
    drone_lat: float,
    drone_lon: float,
    altitude_agl: float,
    gimbal_pitch: float,
    gimbal_roll: float,
    gimbal_yaw: float,
    drone_roll: float,
    drone_pitch: float,
    drone_yaw: float,
    K: np.ndarray,
    dist_coeffs: Sequence[float] | np.ndarray | None = None,
) -> GPSCoord:
    """Geo-locate a single pixel detection to a GPS coordinate.

    Chains `pixel_to_ray` -> `ray_cam_to_world` -> `ray_to_ground` ->
    `local_ned_to_gps`. All angles are in radians, NED convention (see
    module header). `altitude_agl` is the drone's height in meters above
    the *local ground plane the flag sits on* (flat-earth assumption). It
    must not be altitude above Home, MSL, ellipsoid height, or another
    reference unless that reference is known to match the flag's ground
    plane.

    Because the flag lies flat on the ground, `(u, v)` should be the
    bottom-center pixel of the YOLO bounding box (the true ground-contact
    point), not the box center -- this is the caller's responsibility; this
    function does not inspect or enforce it.

    This function is cheap (a handful of small matrix ops, no I/O) and is
    intended to be called on every frame a flag is tracked; aggregate
    repeated calls with `aggregate_coords` to average out gimbal-angle
    jitter and flat-ground-assumption error rather than trusting a single
    frame.

    Args:
        u: Pixel column of the flag's ground-contact point.
        v: Pixel row of the flag's ground-contact point.
        drone_lat: Drone latitude at capture time, decimal degrees.
        drone_lon: Drone longitude at capture time, decimal degrees.
        altitude_agl: Drone height above the local projection ground plane,
            meters (must be positive).
        gimbal_pitch: Gimbal pitch relative to the body, radians (-pi/2 =
            straight down).
        gimbal_roll: Gimbal roll relative to the body, radians.
        gimbal_yaw: Gimbal yaw relative to the body, radians.
        drone_roll: Drone roll, radians (NED body frame).
        drone_pitch: Drone pitch, radians (NED body frame).
        drone_yaw: Drone yaw/heading, radians (0 = North, NED).
        K: 3x3 camera intrinsics matrix (see `build_intrinsics`).
        dist_coeffs: Optional distortion coefficients, see `pixel_to_ray`.

    Returns:
        ``(flag_lat, flag_lon)`` in decimal degrees.

    Raises:
        ValueError: if the resulting ray never intersects the ground (see
            `ray_to_ground`) or if required numeric inputs are invalid.
    """
    pixel = PixelCoordinate(u, v)
    drone = DronePose(
        drone_lat,
        drone_lon,
        altitude_agl,
        drone_roll,
        drone_pitch,
        drone_yaw,
    )
    gimbal = GimbalPose(gimbal_pitch, gimbal_roll, gimbal_yaw)

    ray_cam = pixel_to_ray(pixel.u, pixel.v, K, dist_coeffs)
    ray_world = ray_cam_to_world(
        ray_cam,
        gimbal.pitch,
        gimbal.roll,
        gimbal.yaw,
        drone.roll,
        drone.pitch,
        drone.yaw,
    )
    north_m, east_m = ray_to_ground(ray_world, drone.altitude_agl)
    return local_ned_to_gps(north_m, east_m, drone.lat, drone.lon)


def aggregate_coords(coords: list[GPSCoord]) -> GPSCoord:
    """Aggregate repeated per-frame GPS estimates of the same flag.

    Returns the simple per-axis median of (lat, lon) pairs.
    # ASSUMPTION: per-axis median (not a true 2D geometric median) is used
    -- explicitly allowed by the spec and far cheaper; at the small angular
    spread expected from gimbal jitter the two are practically
    indistinguishable, and per-axis median is robust to outlier frames
    (e.g. a momentary misdetection) the same way a geometric median would
    be.

    Args:
        coords: Non-empty list of `(lat, lon)` pairs, decimal degrees, e.g.
            the output of `pixel_to_gps` called once per tracked frame.

    Returns:
        ``(lat, lon)``: the per-axis median, decimal degrees.

    Raises:
        ValueError: if `coords` is empty.
    """
    coords_array = _as_finite_float_array("coords", coords)
    if coords_array.size == 0:
        raise ValueError("aggregate_coords requires at least one coordinate")
    if coords_array.ndim != 2 or coords_array.shape[1] != 2:
        raise ValueError(
            "coords must be a non-empty sequence of (lat, lon) coordinate pairs"
        )

    lats = coords_array[:, 0]
    lons = coords_array[:, 1]
    return float(np.median(lats)), float(np.median(lons))
