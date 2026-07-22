# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   03/06/2026

"""test_config.py - hardware-free pytest suite for config.py.

Every test is pure: the `from_dicts` cases use in-memory dicts and the loader
cases use a pytest `tmp_path` with real YAML files.  No Raspberry Pi, camera,
MAVLink link, or network is ever involved.

The ten cases map 1:1 to the deliverable checklist:
  1  valid config              6  optional defaults applied
  2  immutability (frozen)     7  low-altitude warning
  3  missing required field    8  unknown key tolerated (warns, loads)
  4  out-of-range value        9  YAML round-trip == dict path
  5  bad enum                 10  missing file -> ConfigError (no leak)

Run:
    pytest test_config.py -v
"""

from __future__ import annotations

import dataclasses

import pytest
import yaml

from core.config import (
    AppConfig,
    CameraConfig,
    ConfigError,
    DEFAULT_CONFIDENCE_THRESHOLD,
    FramesConfig,
    MissionConfig,
    SPECIAL_CONFIDENCE_THRESHOLDS,
    from_dicts,
    load_config,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers - the canonical *valid* config, as three dicts.
# Individual tests copy and mutate these to exercise one failure at a time.

def _valid_camera() -> dict:
    return {
        "fx": 1200.0,
        "fy": 1205.0,
        "cx": 640.0,
        "cy": 360.0,
        "image_width": 1280,
        "image_height": 720,
        "dist_coeffs": [0.1, -0.05, 0.001, 0.0, 0.0],
    }


def _valid_mission() -> dict:
    return {
        "search_altitude_m": 80.0,
        "ground_speed_mps": 6.0,
        "target_flag_class": 3,
        "confidence_threshold": 0.6,
        "dedup_merge_distance_m": 5.0,
        "iou_threshold": 0.3,
        "min_hits": 3,
        "max_age": 5,
        "lane_overlap": 0.25,
        "logging": {"log_level": "INFO", "log_dir": "/var/log/mission"},
    }


def _valid_frames() -> dict:
    return {"gimbal_reference_frame": "body"}


# ---------------------------------------------------------------------------
# Test 1 - a well-formed set of dicts produces a fully populated AppConfig

def test_valid_config_populates_everything():
    """from_dicts on valid input returns a typed, fully populated AppConfig."""
    cfg = from_dicts(_valid_camera(), _valid_mission(), _valid_frames())

    assert isinstance(cfg, AppConfig)
    assert isinstance(cfg.camera, CameraConfig)
    assert isinstance(cfg.mission, MissionConfig)
    assert isinstance(cfg.frames, FramesConfig)

    # Camera - spot-check values and types.
    assert cfg.camera.fx == 1200.0
    assert cfg.camera.image_width == 1280
    assert isinstance(cfg.camera.image_width, int)
    # dist_coeffs is an immutable tuple, not a list.
    assert cfg.camera.dist_coeffs == (0.1, -0.05, 0.001, 0.0, 0.0)
    assert isinstance(cfg.camera.dist_coeffs, tuple)

    # Mission - the audit-critical target class and tracker params.
    assert cfg.mission.target_flag_class == 3
    assert cfg.mission.confidence_threshold == 0.6
    assert cfg.mission.iou_threshold == 0.3
    assert cfg.mission.min_hits == 3
    assert cfg.mission.max_age == 5
    assert cfg.mission.dedup_merge_distance_m == 5.0
    assert cfg.mission.lane_overlap == 0.25

    # Frames + logging primitives.
    assert cfg.frames.gimbal_reference_frame == "body"
    assert cfg.log_level == "INFO"
    assert cfg.log_dir == "/var/log/mission"


# ---------------------------------------------------------------------------
# Test 2 - every config dataclass is frozen (immutable)

@pytest.mark.parametrize(
    "attr, value",
    [
        ("search_altitude_m", 42.0),
        ("target_flag_class", 9),
    ],
)
def test_mission_is_immutable(attr, value):
    """Assigning to a frozen dataclass field raises FrozenInstanceError."""
    cfg = from_dicts(_valid_camera(), _valid_mission(), _valid_frames())
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(cfg.mission, attr, value)


def test_all_dataclasses_are_frozen():
    """Camera, Frames, and the top-level AppConfig are frozen too."""
    cfg = from_dicts(_valid_camera(), _valid_mission(), _valid_frames())
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.camera.fx = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.frames.gimbal_reference_frame = "earth"
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.log_level = "DEBUG"


# ---------------------------------------------------------------------------
# Test 3 - a missing required field raises ConfigError naming that field

def test_missing_target_flag_class_raises():
    """Omitting the required target_flag_class fails loud, naming the key."""
    mission = _valid_mission()
    del mission["target_flag_class"]

    with pytest.raises(ConfigError) as excinfo:
        from_dicts(_valid_camera(), mission, _valid_frames())

    message = str(excinfo.value)
    assert "target_flag_class" in message
    assert "mission.yaml" in message   # source is named too


def test_missing_camera_field_raises():
    """A missing camera intrinsic is caught and named (not a bare KeyError)."""
    camera = _valid_camera()
    del camera["fx"]
    with pytest.raises(ConfigError) as excinfo:
        from_dicts(camera, _valid_mission(), _valid_frames())
    assert "fx" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Test 4 - out-of-range values raise ConfigError with value + bound

def test_confidence_threshold_out_of_range():
    """confidence_threshold=1.7 is outside [0, 1] -> ConfigError."""
    mission = _valid_mission()
    mission["confidence_threshold"] = 1.7
    with pytest.raises(ConfigError) as excinfo:
        from_dicts(_valid_camera(), mission, _valid_frames())
    message = str(excinfo.value)
    assert "confidence_threshold" in message
    assert "1.7" in message


def test_negative_altitude_out_of_range():
    """A negative search altitude is below the 10 m floor -> ConfigError."""
    mission = _valid_mission()
    mission["search_altitude_m"] = -20.0
    with pytest.raises(ConfigError) as excinfo:
        from_dicts(_valid_camera(), mission, _valid_frames())
    assert "search_altitude_m" in str(excinfo.value)


def test_lane_overlap_upper_bound_is_exclusive():
    """lane_overlap must be < 1 (half-open); exactly 1.0 is rejected."""
    mission = _valid_mission()
    mission["lane_overlap"] = 1.0
    with pytest.raises(ConfigError) as excinfo:
        from_dicts(_valid_camera(), mission, _valid_frames())
    assert "lane_overlap" in str(excinfo.value)


def test_min_hits_must_be_positive_int():
    """min_hits=0 violates the >0 rule -> ConfigError."""
    mission = _valid_mission()
    mission["min_hits"] = 0
    with pytest.raises(ConfigError):
        from_dicts(_valid_camera(), mission, _valid_frames())


def test_bool_is_not_accepted_as_number():
    """A YAML bool must not silently pass as 1 for a numeric field (G4)."""
    mission = _valid_mission()
    mission["confidence_threshold"] = True
    with pytest.raises(ConfigError):
        from_dicts(_valid_camera(), mission, _valid_frames())


# ---------------------------------------------------------------------------
# Test 5 - an invalid gimbal frame enum raises ConfigError

def test_bad_gimbal_frame_enum():
    """gimbal_reference_frame='sideways' is not body|earth -> ConfigError."""
    frames = {"gimbal_reference_frame": "sideways"}
    with pytest.raises(ConfigError) as excinfo:
        from_dicts(_valid_camera(), _valid_mission(), frames)
    message = str(excinfo.value)
    assert "gimbal_reference_frame" in message
    assert "sideways" in message


def test_both_valid_frames_accepted():
    """Both 'body' and 'earth' load without error."""
    for frame in ("body", "earth"):
        cfg = from_dicts(
            _valid_camera(), _valid_mission(), {"gimbal_reference_frame": frame}
        )
        assert cfg.frames.gimbal_reference_frame == frame


# ---------------------------------------------------------------------------
# Test 6 - omitting an optional field applies the documented default

def test_optional_defaults_applied():
    """Omitting dist_coeffs and dedup_merge_distance_m uses their defaults."""
    camera = _valid_camera()
    del camera["dist_coeffs"]           # optional -> default ()
    mission = _valid_mission()
    del mission["dedup_merge_distance_m"]   # optional -> default 5.0
    del mission["logging"]              # optional -> log_level=INFO, log_dir=None

    cfg = from_dicts(camera, mission, _valid_frames())

    assert cfg.camera.dist_coeffs == ()
    assert cfg.mission.dedup_merge_distance_m == 5.0
    assert cfg.log_level == "INFO"
    assert cfg.log_dir is None


# ---------------------------------------------------------------------------
# Test 7 - a low search altitude loads but warns (the -5 pt rule)

def test_low_altitude_warns_but_loads():
    """search_altitude_m=40 is within [10,150] but <50, so it warns + loads."""
    mission = _valid_mission()
    mission["search_altitude_m"] = 40.0

    with pytest.warns(RuntimeWarning, match="below"):
        cfg = from_dicts(_valid_camera(), mission, _valid_frames())

    assert cfg.mission.search_altitude_m == 40.0


# ---------------------------------------------------------------------------
# Test 8 - an unknown/extra key warns but still loads (forward-compatible)

def test_unknown_key_tolerated():
    """An unrecognised key warns (RuntimeWarning) but the config still loads."""
    mission = _valid_mission()
    mission["future_experimental_knob"] = 123

    with pytest.warns(RuntimeWarning, match="future_experimental_knob"):
        cfg = from_dicts(_valid_camera(), mission, _valid_frames())

    # The known fields are unaffected.
    assert cfg.mission.target_flag_class == 3


# ---------------------------------------------------------------------------
# Test 9 - the YAML loader yields the same AppConfig as the dict path

def test_yaml_roundtrip_matches_dict_path(tmp_path):
    """Writing the dicts to YAML and load_config()-ing them equals from_dicts."""
    (tmp_path / "camera.yaml").write_text(yaml.safe_dump(_valid_camera()))
    (tmp_path / "mission.yaml").write_text(yaml.safe_dump(_valid_mission()))
    (tmp_path / "frames.yaml").write_text(yaml.safe_dump(_valid_frames()))

    from_yaml = load_config(tmp_path)
    from_dict = from_dicts(_valid_camera(), _valid_mission(), _valid_frames())

    # Frozen dataclasses compare by value, so the two must be identical.
    assert from_yaml == from_dict


# ---------------------------------------------------------------------------
# Test 10 - a missing dir/file raises ConfigError, not a leaked FileNotFoundError

def test_missing_dir_raises_configerror(tmp_path):
    """Pointing load_config at a non-existent dir yields a clear ConfigError."""
    missing = tmp_path / "does_not_exist"
    with pytest.raises(ConfigError) as excinfo:
        load_config(missing)
    message = str(excinfo.value)
    assert "not found" in message
    assert "camera.yaml" in message


def test_missing_single_file_raises_configerror(tmp_path):
    """If one of the three files is absent, load_config names it, not KeyError."""
    # Provide camera + frames but omit mission.yaml.
    (tmp_path / "camera.yaml").write_text(yaml.safe_dump(_valid_camera()))
    (tmp_path / "frames.yaml").write_text(yaml.safe_dump(_valid_frames()))

    with pytest.raises(ConfigError) as excinfo:
        load_config(tmp_path)
    assert "mission.yaml" in str(excinfo.value)


def test_missing_file_is_not_filenotfounderror(tmp_path):
    """The raised type is ConfigError - FileNotFoundError must not leak (G1)."""
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope")
    # And it is specifically NOT a bare OSError leaking through.
    try:
        load_config(tmp_path / "nope")
    except ConfigError:
        pass
    except FileNotFoundError:  # pragma: no cover
        pytest.fail("FileNotFoundError leaked instead of ConfigError")


# ---------------------------------------------------------------------------
# Detector per-class confidence thresholds (ASSUMPTION T1-T3).
# ---------------------------------------------------------------------------
def test_default_confidence_threshold_is_060():
    assert DEFAULT_CONFIDENCE_THRESHOLD == pytest.approx(0.60)


def test_special_confidence_thresholds_covers_exactly_the_four_low_confidence_flags():
    assert SPECIAL_CONFIDENCE_THRESHOLDS == {
        "Egypt": 0.25,
        "Iraq": 0.25,
        "Syria": 0.25,
        "Yemen": 0.25,
    }
