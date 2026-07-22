# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   06/06/2026

"""Tests for geolocate.py.

Fixtures model the SIYI A8 mini: a 4096x2160 sensor with ~93 deg diagonal
FOV. Focal length (assuming square pixels, fx == fy) is derived from the
diagonal FOV via the standard pinhole relation:

    diagonal_px = sqrt(width_px**2 + height_px**2)
    diagonal_px / 2 = f * tan(DFOV / 2)
    f = (diagonal_px / 2) / tan(DFOV / 2)

See `siyi_focal_length_px` below.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import geo.geolocate as geo

IMG_W = 4096
IMG_H = 2160
CX = 2048.0
CY = 1080.0
DFOV_DEG = 93.0

DRONE_LAT = 30.0  # deg, arbitrary fixed test latitude
DRONE_LON = 31.0  # deg, arbitrary fixed test longitude
ALTITUDE = 50.0  # m AGL


def siyi_focal_length_px(width: int, height: int, dfov_deg: float) -> float:
    """Derive a square-pixel focal length (px) from a diagonal FOV."""
    diagonal_px = math.hypot(width, height)
    return (diagonal_px / 2.0) / math.tan(math.radians(dfov_deg) / 2.0)


FOCAL_PX = siyi_focal_length_px(IMG_W, IMG_H, DFOV_DEG)
K = geo.build_intrinsics(FOCAL_PX, FOCAL_PX, CX, CY)

NADIR_PITCH = -math.pi / 2  # ASSUMPTION: straight down, see geolocate.py header


def _gps_delta_to_local(
    lat: float,
    lon: float,
    reference_lat: float = DRONE_LAT,
    reference_lon: float = DRONE_LON,
) -> tuple[float, float]:
    """Convert a GPS delta back to local north/east meters for assertions."""
    north_m = (lat - reference_lat) * geo.METERS_PER_DEG_LAT
    east_m = (
        (lon - reference_lon)
        * geo.METERS_PER_DEG_LAT
        * math.cos(math.radians(reference_lat))
    )
    return north_m, east_m


def _gps_to_pixel(
    north_m: float,
    east_m: float,
    altitude_agl: float,
    gimbal_pitch: float,
    gimbal_roll: float,
    gimbal_yaw: float,
    drone_roll: float,
    drone_pitch: float,
    drone_yaw: float,
    K: np.ndarray,
) -> tuple[float, float]:
    """Test-only inverse of the geo-location chain: ground point -> pixel.

    Builds the world-frame ray that hits (north_m, east_m) at the ground
    plane, then inverts `ray_cam_to_world` (rotation matrices are
    orthogonal, so inverse == transpose) and projects through K. Used only
    to construct round-trip test fixtures; not part of the public module.
    """
    ray_world = np.array([north_m, east_m, altitude_agl], dtype=np.float64)
    ray_world = ray_world / np.linalg.norm(ray_world)

    R_gimbal = geo._attitude_rotation(gimbal_yaw, gimbal_pitch, gimbal_roll)
    R_cam_to_body = R_gimbal @ geo._CAM_OPTICAL_TO_BODY_ALIGNED
    R_body_to_world = geo._attitude_rotation(drone_yaw, drone_pitch, drone_roll)

    ray_cam = R_cam_to_body.T @ (R_body_to_world.T @ ray_world)
    # Rescale so z == 1, matching pixel_to_ray's pinhole convention.
    ray_cam = ray_cam / ray_cam[2]
    pixel_h = K @ ray_cam
    return float(pixel_h[0]), float(pixel_h[1])


def test_nadir_centered():
    """Gimbal straight down, flag dead-center, zero drone attitude -> drone GPS."""
    lat, lon = geo.pixel_to_gps(
        u=CX,
        v=CY,
        drone_lat=DRONE_LAT,
        drone_lon=DRONE_LON,
        altitude_agl=ALTITUDE,
        gimbal_pitch=NADIR_PITCH,
        gimbal_roll=0.0,
        gimbal_yaw=0.0,
        drone_roll=0.0,
        drone_pitch=0.0,
        drone_yaw=0.0,
        K=K,
    )
    assert lat == pytest.approx(DRONE_LAT, abs=1e-6)
    assert lon == pytest.approx(DRONE_LON, abs=1e-6)


def test_nadir_off_center():
    """Off-center pixel at nadir -> ground offset matches the pixel's view angle."""
    du, dv = 300.0, -200.0  # pixels right of cx, above cy
    u, v = CX + du, CY + dv

    lat, lon = geo.pixel_to_gps(
        u=u,
        v=v,
        drone_lat=DRONE_LAT,
        drone_lon=DRONE_LON,
        altitude_agl=ALTITUDE,
        gimbal_pitch=NADIR_PITCH,
        gimbal_roll=0.0,
        gimbal_yaw=0.0,
        drone_roll=0.0,
        drone_pitch=0.0,
        drone_yaw=0.0,
        K=K,
    )

    # Independently derived expected offsets: a pinhole ray at pixel angle
    # theta from the optical axis hits the ground at altitude * tan(theta).
    # At nadir with zero drone attitude (see geolocate.py header, conventions
    # #6-#7): +u (image right) -> +East, +v (image down) -> -North (South).
    theta_u = math.atan2(du, FOCAL_PX)
    theta_v = math.atan2(dv, FOCAL_PX)
    expected_east_m = ALTITUDE * math.tan(theta_u)
    expected_north_m = -ALTITUDE * math.tan(theta_v)

    expected_lat, expected_lon = geo.local_ned_to_gps(
        expected_north_m, expected_east_m, DRONE_LAT, DRONE_LON
    )
    assert lat == pytest.approx(expected_lat, abs=1e-9)
    assert lon == pytest.approx(expected_lon, abs=1e-9)

    # Sanity check on direction: right-of-center pixel -> flag East of drone.
    assert lon > DRONE_LON
    # Above-center pixel (smaller v) -> flag North of drone.
    assert lat > DRONE_LAT


def test_tilted_gimbal():
    """Gimbal pitched 10 deg off nadir, flag centered -> ~50*tan(10deg) m offset."""
    tilt = math.radians(10.0)
    lat, lon = geo.pixel_to_gps(
        u=CX,
        v=CY,
        drone_lat=DRONE_LAT,
        drone_lon=DRONE_LON,
        altitude_agl=ALTITUDE,
        gimbal_pitch=NADIR_PITCH + tilt,
        gimbal_roll=0.0,
        gimbal_yaw=0.0,
        drone_roll=0.0,
        drone_pitch=0.0,
        drone_yaw=0.0,
        K=K,
    )
    expected_offset_m = ALTITUDE * math.tan(tilt)
    expected_lat, expected_lon = geo.local_ned_to_gps(
        expected_offset_m, 0.0, DRONE_LAT, DRONE_LON
    )
    assert lat == pytest.approx(expected_lat, abs=1e-9)
    assert lon == pytest.approx(expected_lon, abs=1e-9)
    assert expected_offset_m == pytest.approx(8.8163, abs=1e-3)


def test_yaw_rotation():
    """Same 10 deg tilt, drone yaw = 90 deg -> displacement rotates North to East."""
    tilt = math.radians(10.0)
    lat, lon = geo.pixel_to_gps(
        u=CX,
        v=CY,
        drone_lat=DRONE_LAT,
        drone_lon=DRONE_LON,
        altitude_agl=ALTITUDE,
        gimbal_pitch=NADIR_PITCH + tilt,
        gimbal_roll=0.0,
        gimbal_yaw=0.0,
        drone_roll=0.0,
        drone_pitch=0.0,
        drone_yaw=math.pi / 2,
        K=K,
    )
    expected_offset_m = ALTITUDE * math.tan(tilt)
    expected_lat, expected_lon = geo.local_ned_to_gps(
        0.0, expected_offset_m, DRONE_LAT, DRONE_LON
    )
    assert lat == pytest.approx(expected_lat, abs=1e-9)
    assert lon == pytest.approx(expected_lon, abs=1e-9)
    # Displacement should now be (almost) purely longitudinal, not latitudinal.
    assert lat == pytest.approx(DRONE_LAT, abs=1e-6)
    assert lon > DRONE_LON


def test_yaw_360_equals_yaw_0():
    """A full yaw revolution must preserve the same ground intersection."""
    common_kwargs = dict(
        u=CX + 220.0,
        v=CY - 180.0,
        drone_lat=DRONE_LAT,
        drone_lon=DRONE_LON,
        altitude_agl=ALTITUDE,
        gimbal_pitch=NADIR_PITCH + math.radians(7.0),
        gimbal_roll=math.radians(2.0),
        gimbal_yaw=math.radians(-3.0),
        drone_roll=math.radians(1.0),
        drone_pitch=math.radians(-2.0),
        K=K,
    )

    coord_yaw_0 = geo.pixel_to_gps(drone_yaw=0.0, **common_kwargs)
    coord_yaw_360 = geo.pixel_to_gps(drone_yaw=2.0 * math.pi, **common_kwargs)

    assert coord_yaw_360 == pytest.approx(coord_yaw_0, abs=1e-12)


def test_mirror_pixel_around_center_mirrors_east_west_at_nadir():
    """Left/right mirror pixels at nadir should mirror only East/West offset."""
    du = 300.0
    lat_left, lon_left = geo.pixel_to_gps(
        u=CX - du,
        v=CY,
        drone_lat=DRONE_LAT,
        drone_lon=DRONE_LON,
        altitude_agl=ALTITUDE,
        gimbal_pitch=NADIR_PITCH,
        gimbal_roll=0.0,
        gimbal_yaw=0.0,
        drone_roll=0.0,
        drone_pitch=0.0,
        drone_yaw=0.0,
        K=K,
    )
    lat_right, lon_right = geo.pixel_to_gps(
        u=CX + du,
        v=CY,
        drone_lat=DRONE_LAT,
        drone_lon=DRONE_LON,
        altitude_agl=ALTITUDE,
        gimbal_pitch=NADIR_PITCH,
        gimbal_roll=0.0,
        gimbal_yaw=0.0,
        drone_roll=0.0,
        drone_pitch=0.0,
        drone_yaw=0.0,
        K=K,
    )

    north_left, east_left = _gps_delta_to_local(lat_left, lon_left)
    north_right, east_right = _gps_delta_to_local(lat_right, lon_right)

    assert north_left == pytest.approx(0.0, abs=1e-9)
    assert north_right == pytest.approx(0.0, abs=1e-9)
    assert east_left < 0.0
    assert east_right > 0.0
    assert east_left == pytest.approx(-east_right, abs=1e-9)


def test_positive_drone_pitch_moves_nadir_center_north():
    """Documented positive pitch (nose up) tilts the nadir hit point north."""
    pitch = math.radians(5.0)
    expected_north_m = ALTITUDE * math.tan(pitch)

    lat_pos, lon_pos = geo.pixel_to_gps(
        u=CX,
        v=CY,
        drone_lat=DRONE_LAT,
        drone_lon=DRONE_LON,
        altitude_agl=ALTITUDE,
        gimbal_pitch=NADIR_PITCH,
        gimbal_roll=0.0,
        gimbal_yaw=0.0,
        drone_roll=0.0,
        drone_pitch=pitch,
        drone_yaw=0.0,
        K=K,
    )
    lat_neg, lon_neg = geo.pixel_to_gps(
        u=CX,
        v=CY,
        drone_lat=DRONE_LAT,
        drone_lon=DRONE_LON,
        altitude_agl=ALTITUDE,
        gimbal_pitch=NADIR_PITCH,
        gimbal_roll=0.0,
        gimbal_yaw=0.0,
        drone_roll=0.0,
        drone_pitch=-pitch,
        drone_yaw=0.0,
        K=K,
    )

    north_pos, east_pos = _gps_delta_to_local(lat_pos, lon_pos)
    north_neg, east_neg = _gps_delta_to_local(lat_neg, lon_neg)

    assert north_pos == pytest.approx(expected_north_m, abs=1e-9)
    assert north_neg == pytest.approx(-expected_north_m, abs=1e-9)
    assert east_pos == pytest.approx(0.0, abs=1e-9)
    assert east_neg == pytest.approx(0.0, abs=1e-9)


def test_roll_symmetry_at_nadir_center():
    """Equal and opposite drone roll should mirror the East/West ground offset."""
    roll = math.radians(5.0)
    expected_east_magnitude = ALTITUDE * math.tan(roll)

    lat_pos, lon_pos = geo.pixel_to_gps(
        u=CX,
        v=CY,
        drone_lat=DRONE_LAT,
        drone_lon=DRONE_LON,
        altitude_agl=ALTITUDE,
        gimbal_pitch=NADIR_PITCH,
        gimbal_roll=0.0,
        gimbal_yaw=0.0,
        drone_roll=roll,
        drone_pitch=0.0,
        drone_yaw=0.0,
        K=K,
    )
    lat_neg, lon_neg = geo.pixel_to_gps(
        u=CX,
        v=CY,
        drone_lat=DRONE_LAT,
        drone_lon=DRONE_LON,
        altitude_agl=ALTITUDE,
        gimbal_pitch=NADIR_PITCH,
        gimbal_roll=0.0,
        gimbal_yaw=0.0,
        drone_roll=-roll,
        drone_pitch=0.0,
        drone_yaw=0.0,
        K=K,
    )

    north_pos, east_pos = _gps_delta_to_local(lat_pos, lon_pos)
    north_neg, east_neg = _gps_delta_to_local(lat_neg, lon_neg)

    assert north_pos == pytest.approx(0.0, abs=1e-9)
    assert north_neg == pytest.approx(0.0, abs=1e-9)
    assert east_pos == pytest.approx(-expected_east_magnitude, abs=1e-9)
    assert east_neg == pytest.approx(expected_east_magnitude, abs=1e-9)


def test_hand_computed_center_tilt_with_arbitrary_yaw():
    """Centered tilted boresight should produce altitude*tan(tilt) along heading."""
    tilt = math.radians(12.0)
    yaw = math.radians(35.0)

    lat, lon = geo.pixel_to_gps(
        u=CX,
        v=CY,
        drone_lat=DRONE_LAT,
        drone_lon=DRONE_LON,
        altitude_agl=ALTITUDE,
        gimbal_pitch=NADIR_PITCH + tilt,
        gimbal_roll=0.0,
        gimbal_yaw=0.0,
        drone_roll=0.0,
        drone_pitch=0.0,
        drone_yaw=yaw,
        K=K,
    )

    ground_offset_m = ALTITUDE * math.tan(tilt)
    expected_north_m = ground_offset_m * math.cos(yaw)
    expected_east_m = ground_offset_m * math.sin(yaw)
    expected_lat = DRONE_LAT + expected_north_m / 111320.0
    expected_lon = DRONE_LON + expected_east_m / (
        111320.0 * math.cos(math.radians(DRONE_LAT))
    )

    assert lat == pytest.approx(expected_lat, abs=1e-9)
    assert lon == pytest.approx(expected_lon, abs=1e-9)


def test_default_gimbal_reference_frame_is_body_relative():
    assert (
        geo.DEFAULT_GEOLOCATION_CONFIG.gimbal_reference_frame
        is geo.GimbalReferenceFrame.BODY_RELATIVE
    )
    assert (
        geo.GeolocationConfig().gimbal_reference_frame
        is geo.GimbalReferenceFrame.BODY_RELATIVE
    )


def test_earth_attitude_helper_matches_body_relative_with_identity_drone():
    """Earth-relative helper matches the current path when body attitude is zero."""
    ray_cam = geo.pixel_to_ray(CX + 150.0, CY - 90.0, K)
    camera_pitch = NADIR_PITCH + math.radians(8.0)
    camera_roll = math.radians(2.0)
    camera_yaw = math.radians(20.0)

    body_relative_world_ray = geo.ray_cam_to_world(
        ray_cam,
        gimbal_pitch=camera_pitch,
        gimbal_roll=camera_roll,
        gimbal_yaw=camera_yaw,
        drone_roll=0.0,
        drone_pitch=0.0,
        drone_yaw=0.0,
    )
    earth_relative_world_ray = geo.ray_cam_to_world_from_earth_attitude(
        ray_cam,
        camera_pitch=camera_pitch,
        camera_roll=camera_roll,
        camera_yaw=camera_yaw,
    )

    np.testing.assert_allclose(
        earth_relative_world_ray,
        body_relative_world_ray,
        atol=1e-12,
    )


def test_round_trip():
    """Project a known ground point to a pixel, then recover it via pixel_to_gps."""
    drone_roll = math.radians(3.0)
    drone_pitch = math.radians(-4.0)
    drone_yaw = math.radians(35.0)
    gimbal_pitch = NADIR_PITCH + math.radians(6.0)
    gimbal_roll = math.radians(2.0)
    gimbal_yaw = math.radians(-8.0)

    target_north_m, target_east_m = 12.0, -7.0

    u, v = _gps_to_pixel(
        target_north_m,
        target_east_m,
        ALTITUDE,
        gimbal_pitch,
        gimbal_roll,
        gimbal_yaw,
        drone_roll,
        drone_pitch,
        drone_yaw,
        K,
    )
    # The synthetic pixel must land inside the sensor for a valid fixture.
    assert 0.0 <= u <= IMG_W
    assert 0.0 <= v <= IMG_H

    recovered_lat, recovered_lon = geo.pixel_to_gps(
        u=u,
        v=v,
        drone_lat=DRONE_LAT,
        drone_lon=DRONE_LON,
        altitude_agl=ALTITUDE,
        gimbal_pitch=gimbal_pitch,
        gimbal_roll=gimbal_roll,
        gimbal_yaw=gimbal_yaw,
        drone_roll=drone_roll,
        drone_pitch=drone_pitch,
        drone_yaw=drone_yaw,
        K=K,
    )

    expected_lat, expected_lon = geo.local_ned_to_gps(
        target_north_m, target_east_m, DRONE_LAT, DRONE_LON
    )
    assert recovered_lat == pytest.approx(expected_lat, abs=1e-9)
    assert recovered_lon == pytest.approx(expected_lon, abs=1e-9)


def test_build_intrinsics():
    k = geo.build_intrinsics(100.0, 200.0, 50.0, 60.0)
    assert k.shape == (3, 3)
    np.testing.assert_array_equal(
        k, np.array([[100.0, 0.0, 50.0], [0.0, 200.0, 60.0], [0.0, 0.0, 1.0]])
    )


def test_input_validation_rejects_invalid_intrinsics():
    with pytest.raises(ValueError, match="fx must be positive"):
        geo.build_intrinsics(0.0, 200.0, 50.0, 60.0)

    with pytest.raises(ValueError, match="3x3 camera intrinsics"):
        geo.pixel_to_ray(CX, CY, np.eye(2))

    bad_focal = K.copy()
    bad_focal[0, 0] = -1.0
    with pytest.raises(ValueError, match="focal lengths must be positive"):
        geo.pixel_to_ray(CX, CY, bad_focal)


def test_input_validation_rejects_bad_vectors_and_altitudes():
    with pytest.raises(ValueError, match="altitude_agl must be positive"):
        geo.ray_to_ground(np.array([0.0, 0.0, 1.0]), 0.0)

    with pytest.raises(ValueError, match="length-3 vector"):
        geo.ray_cam_to_world(
            np.array([1.0, 0.0]),
            gimbal_pitch=NADIR_PITCH,
            gimbal_roll=0.0,
            gimbal_yaw=0.0,
            drone_roll=0.0,
            drone_pitch=0.0,
            drone_yaw=0.0,
        )


def test_ray_to_ground_rejects_upward_ray():
    with pytest.raises(ValueError):
        geo.ray_to_ground(np.array([1.0, 0.0, 0.0]), ALTITUDE)
    with pytest.raises(ValueError):
        geo.ray_to_ground(np.array([0.0, 0.0, -0.1]), ALTITUDE)


def test_aggregate_coords_per_axis_median():
    coords = [(30.0001, 31.0002), (30.0003, 31.0001), (30.0002, 31.0009)]
    lat, lon = geo.aggregate_coords(coords)
    assert lat == pytest.approx(30.0002)
    assert lon == pytest.approx(31.0002)


def test_aggregate_coords_rejects_empty():
    with pytest.raises(ValueError):
        geo.aggregate_coords([])
