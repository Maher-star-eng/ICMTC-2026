<!--
Author: Maher Ahmed Maher Shola
Email:  maher.shola.officer@gmail.com
        M.shoala61308@student.aast.edu
Date:   14/07/2026
-->

# geo/

Geolocation math and telemetry ingestion — pixel coordinates plus drone pose
in, a GPS coordinate out. No detection or tracking logic lives here.

| File | Role |
|---|---|
| `geolocate.py` | Pure computation, no I/O, no hardware access. Converts a pixel coordinate + drone/gimbal pose + camera intrinsics into a GPS coordinate via `pixel_to_ray → ray_cam_to_world → ray_to_ground → local_ned_to_gps` (composed as `pixel_to_gps`), plus `aggregate_coords` for per-axis median aggregation. Follows the ArduPilot/aerospace NED convention throughout; every place the convention was ambiguous is documented with a numbered convention in the module header and a matching `# ASSUMPTION:` comment at its point of use. |
| `telemetry.py` | Raspberry Pi 5 companion-computer module. `TelemetryBuffer` runs a background MAVLink reader thread, decodes position/attitude/gimbal messages into timestamped, frozen `Pose` samples, and exposes `pose_at(t)` for interpolated lookup keyed on `time.monotonic()`. Deliberately rejects CSV-replay connection strings (see ASSUMPTION A12) after a documented silent-data-loss bug. |

Both modules are pure/hardware-free by design and fully covered by
[`tests/test_geolocate.py`](../tests/test_geolocate.py) and
[`tests/test_telemetry.py`](../tests/test_telemetry.py) — synthetic pose
samples and fixture math only, no real MAVLink connection is ever opened.
