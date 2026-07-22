# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   05/07/2026

"""dry_run.py - first-ever full-pipeline dry run on the Mac (ICMTC 2026, Mission 2).

    capture -> Mission -> detector -> tracker -> MedianGeolocatorAdapter -> VisionSystem -> judge log

Every module above has been unit-tested in isolation with fakes; none have
been exercised together until this script runs.  The goal is to prove the
WIRING is correct - that real objects pass real data to each other without a
seam breaking - NOT to re-validate detection accuracy (that is already
measured rigorously against the held-out val set in
masked_validation_report.md / threshold_sweep_report.md /
resolution_and_remap_report.md).  See dry_run_report.md for the full,
human-readable writeup this script's output feeds.

REAL (not mocked): the trained model (best_ncnn_model/, via the real
`vision.detector.Detector` + `NcnnBackend`, real masking active), real
`vision.tracker.Tracker`, real `core.mission.MedianGeolocatorAdapter`, real
`ioutils.output.VisionSystem`, real `core.mission.Mission` state machine,
real `vision.capture.FileFrameSource` / `LatestFrameQueue`, real
`ioutils.output.write_judge_log`.

SYNTHETIC (clearly labeled, never used to claim a detection-accuracy
result): the footage (training/dry_run_footage/*.png - schematic tricolor
rectangles, NOT photographs of a physical flag; see PRIMARY RUN section
below and dry_run_report.md for why) and the telemetry (no MAVLink flight
exists to record from; SyntheticTelemetry below documents every value it
fabricates).

Run:
    python -m training.dry_run

Writes:
    training/dry_run_footage/*.png       (synthetic test frames, generated fresh each run)
    training/dry_run_judge_log.json      (real write_judge_log output, primary run)
    training/dry_run_judge_log_secondary.json  (secondary check's judge log)
    training/dry_run_raw_log.txt         (full captured stdout, for the report to quote)
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on sys.path

from core.config import from_dicts
from core.mission import Mission, MedianGeolocatorAdapter, MissionState
from geo.telemetry import Pose
from ioutils.output import write_judge_log
from vision.capture import FileFrameSource, LatestFrameQueue
from vision.detector import Detector, StubBackend, build_detector
from vision.tracker import Detection, Tracker

WEIGHTS_DIR = Path(__file__).resolve().parent / "runs/mission2/yolo11n_flags/weights/best_ncnn_model"
FOOTAGE_DIR = Path(__file__).resolve().parent / "dry_run_footage"
JUDGE_LOG_PRIMARY = Path(__file__).resolve().parent / "dry_run_judge_log.json"
JUDGE_LOG_SECONDARY = Path(__file__).resolve().parent / "dry_run_judge_log_secondary.json"

FRAME_W, FRAME_H = 640, 480
NUM_FRAMES = 10
FPS = 2.0  # slow, deliberate pacing so LatestFrameQueue's consumer never falls behind (see report)

BRIEFED_NAMES = ("Egypt", "Iraq", "Syria", "Yemen")  # the 4 low-confidence classes from prior work


def log(msg: str) -> None:
    print(f"[{time.monotonic():8.3f}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Synthetic footage - schematic tricolor rectangles, NOT photographs of a
# physical flag.  See dry_run_report.md for why (no demo/dryrun footage, no
# non-training real photo of a physical flag, exists anywhere in this repo).
# ---------------------------------------------------------------------------
def build_synthetic_footage() -> Path:
    FOOTAGE_DIR.mkdir(parents=True, exist_ok=True)
    for f in FOOTAGE_DIR.glob("*.png"):
        f.unlink()

    rng = np.random.default_rng(0)
    for i in range(NUM_FRAMES):
        frame = np.full((FRAME_H, FRAME_W, 3), (60, 120, 60), dtype=np.uint8)  # noisy "grass" bg
        frame = np.clip(
            frame.astype(np.int16) + rng.integers(-15, 15, size=frame.shape), 0, 255
        ).astype(np.uint8)

        # Frames 2..6 (0-indexed): a horizontal red/white/black tricolor
        # rectangle - all four briefed flags (Egypt/Iraq/Syria/Yemen) are
        # red-white-black horizontal tricolors with a center emblem this
        # schematic omits.  Position/size drift slightly frame to frame to
        # mimic a drone passing over a static target.  PLUMBING ONLY.
        if 2 <= i <= 6:
            drift = (i - 2) * 6
            x1, y1 = 250 + drift, 180
            x2, y2 = 380 + drift, 260
            band_h = (y2 - y1) // 3
            frame[y1 : y1 + band_h, x1:x2] = (0, 0, 200)  # red   (BGR)
            frame[y1 + band_h : y1 + 2 * band_h, x1:x2] = (230, 230, 230)  # white
            frame[y1 + 2 * band_h : y2, x1:x2] = (20, 20, 20)  # black

        cv2.imwrite(str(FOOTAGE_DIR / f"frame_{i:02d}.png"), frame)

    log(f"wrote {NUM_FRAMES} synthetic frames to {FOOTAGE_DIR}")
    return FOOTAGE_DIR


# ---------------------------------------------------------------------------
# Synthetic telemetry - no MAVLink flight exists to record from.  Every
# value here is fabricated for plausibility, not measured.  Documented in
# full in dry_run_report.md.
# ---------------------------------------------------------------------------
def _nadir_quat(pitch_rad: float) -> tuple[float, float, float, float]:
    """(w,x,y,z) gimbal quaternion for a given pitch, roll=yaw=0 (ZYX), the
    exact inverse of core.mission._quat_to_euler."""
    xyzw = Rotation.from_euler("ZYX", [0.0, pitch_rad, 0.0]).as_quat()
    return (float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2]))


class SyntheticTelemetry:
    """A plausible slow-nadir-pass Pose sequence - SYNTHETIC, not a real flight.

    ALL VALUES ARE FABRICATED (documented here, not measured):
      start_lat/start_lon = 30.0N, 31.0E - an arbitrary round-number
          coordinate, chosen for no reason beyond being a valid lat/lon.
      alt_agl = 55.0 m constant - matches this run's search_altitude_m.
      drift_deg_per_sec = 5e-6 deg/s northward (~0.56 m/s ground speed) -
          "slow nadir pass," chosen to move a visible-but-small amount over
          this run's ~5s duration.
      gimbal: nadir (pitch=-90deg), body-relative, roll=yaw=0, i.e. camera
          points straight down, vehicle attitude level.
      fix_type=3 (3D fix), h_acc_m=1.5 - plausible GPS quality, not measured.

    Two fault-injection windows (elapsed seconds since the first pose_at
    call), both OPTIONAL and independently configurable:
      gap_window: pose_at() returns None (a telemetry dropout) - real
          fault-injection mechanism requested by the dry-run task.
      glitch_window: pose_at() returns a Pose with gimbal pitch=-1deg
          instead of -90deg for ONE window only (a single bad-gimbal-frame -
          tilted nearly to the horizon; verified empirically via
          geo.pixel_to_gps to project ~1.08 km away from the nadir point at
          this run's 55 m altitude, reproducing mission.py's own documented
          ~2 km bug scenario at matching order of magnitude) - used only in
          the secondary check, to prove the median survives an outlier.
    """

    def __init__(
        self,
        *,
        start_lat: float = 30.0,
        start_lon: float = 31.0,
        alt_agl: float = 55.0,
        drift_deg_per_sec: float = 5e-6,
        gap_window: tuple[float, float] | None = None,
        glitch_window: tuple[float, float] | None = None,
    ) -> None:
        self._start_lat = start_lat
        self._start_lon = start_lon
        self._alt_agl = alt_agl
        self._drift = drift_deg_per_sec
        self._gap_window = gap_window
        self._glitch_window = glitch_window
        self._t0: float | None = None
        self._call_count = 0
        self._nadir_quat = _nadir_quat(-math.pi / 2)
        self._glitch_quat = _nadir_quat(math.radians(-1))
        self.call_log: list[dict] = []  # for the report: every pose_at() call, verbatim

    def pose_at(self, t: float) -> Pose | None:
        if self._t0 is None:
            self._t0 = t
        elapsed = t - self._t0
        idx = self._call_count
        self._call_count += 1

        if self._gap_window and self._gap_window[0] <= elapsed < self._gap_window[1]:
            self.call_log.append(
                {"call": idx, "elapsed": round(elapsed, 3), "gap": True,
                 "lat": None, "lon": None, "glitched_gimbal": False}
            )
            return None

        quat = self._nadir_quat
        glitched = False
        if self._glitch_window and self._glitch_window[0] <= elapsed < self._glitch_window[1]:
            quat = self._glitch_quat
            glitched = True

        pose = Pose(
            t=t,
            lat=self._start_lat + self._drift * elapsed,
            lon=self._start_lon,
            alt_agl=self._alt_agl,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            gimbal_quat=quat,
            gimbal_frame="body",
            fix_type=3,
            h_acc_m=1.5,
        )
        self.call_log.append(
            {"call": idx, "elapsed": round(elapsed, 3), "gap": False,
             "lat": pose.lat, "lon": pose.lon, "glitched_gimbal": glitched}
        )
        return pose


# ---------------------------------------------------------------------------
# Thin, read-only instrumentation shim around the REAL Detector.  Delegates
# 100% of detection behavior to the wrapped Detector - it changes nothing,
# only observes and logs (frame identity, per-frame class ids, timing) so
# this script can report on masking (#3) and frame-processing cadence
# without touching detector.py/mission.py production code.
# ---------------------------------------------------------------------------
class LoggingDetectorWrapper:
    def __init__(self, real_detector: Detector) -> None:
        self._real = real_detector
        self.target_class_ids = real_detector.target_class_ids  # exposed for Mission's MS9 check
        self.calls: list[dict] = []

    def detect(self, frame: np.ndarray):
        start = time.monotonic()
        detections = self._real.detect(frame)
        elapsed_ms = (time.monotonic() - start) * 1000.0
        self.calls.append(
            {
                "frame_object_id": id(frame),
                "elapsed_ms": elapsed_ms,
                "num_detections": len(detections),
                "class_ids": [d.class_id for d in detections],
            }
        )
        return detections


class LoggingGeolocatorWrapper:
    """Thin, read-only instrumentation shim around the REAL
    MedianGeolocatorAdapter - delegates 100% of the median-aggregation math
    to the wrapped instance; only observes.  Exists because
    `MedianGeolocatorAdapter.locate()` returns just the median, so this
    logs the SINGLE-FRAME point that was just appended (peeked from the
    real adapter's own `_history` right after the call) alongside the
    returned median - the direct evidence that the reported coordinate is
    an aggregate, not the latest single-frame estimate (the ~2 km bug's
    regression check, at the system level).
    """

    def __init__(self, real_geolocator) -> None:
        self._real = real_geolocator
        self.calls: list[dict] = []

    def locate(self, track, pose):
        before = list(self._real._history.get(track.id, []))
        median = self._real.locate(track, pose)
        after = list(self._real._history.get(track.id, []))
        new_point = after[-1] if len(after) > len(before) or after != before[-len(after):] else None
        self.calls.append(
            {
                "track_id": track.id,
                "pose_is_none": pose is None,
                "single_frame_coord": new_point,
                "median_coord": median,
                "history_len": len(after),
            }
        )
        return median


def resolve_class_ids_by_name(class_names: dict[int, str], wanted: tuple[str, ...]) -> frozenset[int]:
    """Same pattern as vision.detector.resolve_special_thresholds: build
    name->id from the model's own mapping, fail loud on any missing name.
    Never hardcodes a numeric id."""
    name_to_id = {name: cid for cid, name in class_names.items()}
    missing = [name for name in wanted if name not in name_to_id]
    if missing:
        raise RuntimeError(f"class names not found in model: {missing}")
    return frozenset(name_to_id[name] for name in wanted)


def run_mission_to_completion(mission: Mission, label: str) -> list[tuple[int, MissionState]]:
    """Drive `mission.step()` until a terminal state, logging every state
    change with the frame index it happened at."""
    transitions: list[tuple[int, MissionState]] = [(mission._frame_idx, mission.state)]
    log(f"[{label}] initial state: {mission.state.name} (frame_idx={mission._frame_idx})")
    last_state = mission.state
    while mission.state not in (MissionState.HANDOFF_COMPLETE, MissionState.DONE, MissionState.ABORT):
        mission.step()
        if mission.state != last_state:
            transitions.append((mission._frame_idx, mission.state))
            log(f"[{label}] frame_idx={mission._frame_idx}: {last_state.name} -> {mission.state.name}")
            last_state = mission.state
    return transitions


# ---------------------------------------------------------------------------
# PRIMARY RUN: real Detector (real ncnn model + real masking), real Tracker,
# real MedianGeolocatorAdapter, real VisionSystem, real Mission, real
# FileFrameSource/LatestFrameQueue, real write_judge_log.  Synthetic
# footage + synthetic telemetry, both clearly labeled above.
# ---------------------------------------------------------------------------
def primary_run(cfg, briefed_ids: frozenset[int], target_flag_class: int) -> dict:
    log("=" * 70)
    log("PRIMARY RUN: real trained model (ncnn), real masking, real pipeline")
    log("=" * 70)

    with open(WEIGHTS_DIR / "metadata.yaml") as f:
        class_names = yaml.safe_load(f)["names"]

    t_build_start = time.monotonic()
    real_detector = build_detector(
        cfg, backend="ncnn",
        param_path=str(WEIGHTS_DIR / "model.ncnn.param"),
        bin_path=str(WEIGHTS_DIR / "model.ncnn.bin"),
        class_names=class_names,
    )
    real_detector.set_target_classes(briefed_ids)
    build_time_s = time.monotonic() - t_build_start
    log(f"detector built in {build_time_s:.3f}s; detector.target_class_ids={sorted(real_detector.target_class_ids)}")
    assert real_detector.target_class_ids == briefed_ids, "masking not wired as intended"

    detector = LoggingDetectorWrapper(real_detector)

    tracker = Tracker(
        iou_threshold=cfg.mission.iou_threshold,
        min_hits=cfg.mission.min_hits,
        max_age=cfg.mission.max_age,
    )
    geolocator = MedianGeolocatorAdapter(cfg)
    telemetry = SyntheticTelemetry(gap_window=(1.8, 2.7))

    capture_source = FileFrameSource(str(FOOTAGE_DIR), realtime=True, fps=FPS)
    frame_queue = LatestFrameQueue(capture_source)

    mission = Mission(
        cfg,
        detector=detector,
        tracker=tracker,
        geolocator=geolocator,
        telemetry=telemetry,
        frame_queue=frame_queue,
        briefed_target_class_ids=briefed_ids,
        judge_log_path=str(JUDGE_LOG_PRIMARY),
    )

    t_run_start = time.monotonic()
    transitions = run_mission_to_completion(mission, "PRIMARY")
    run_time_s = time.monotonic() - t_run_start

    if mission.state is MissionState.HANDOFF_COMPLETE:
        mission.acknowledge_handoff()
        log(f"[PRIMARY] acknowledge_handoff() -> {mission.state.name}")

    frame_queue.close()

    final = mission.final_result()
    all_class_ids_seen = [cid for call in detector.calls for cid in call["class_ids"]]
    non_target_leak = [cid for cid in all_class_ids_seen if cid not in briefed_ids]
    distinct_frame_objects = len({call["frame_object_id"] for call in detector.calls})

    return {
        "build_time_s": build_time_s,
        "run_time_s": run_time_s,
        "transitions": transitions,
        "detector_calls": detector.calls,
        "all_class_ids_seen": all_class_ids_seen,
        "non_target_leak": non_target_leak,
        "telemetry_log": telemetry.call_log,
        "final_result": final,
        "degraded_count": mission.degraded_count,
        "frame_idx_final": mission._frame_idx,
        "geolocator_history": dict(geolocator._history),
        "target_class_ids": sorted(real_detector.target_class_ids),
        "distinct_frame_objects": distinct_frame_objects,
        "total_detect_calls": len(detector.calls),
    }


# ---------------------------------------------------------------------------
# SECONDARY CHECK - only meaningful if the primary run produced zero real
# detections (expected: synthetic schematic footage vs. a model trained on
# real photos).  Uses vision.detector.Detector + StubBackend (NOT the real
# trained model) so this does NOT stand in for "real Detector" anywhere -
# its only purpose is to prove Tracker -> MedianGeolocatorAdapter ->
# VisionSystem -> Mission -> judge log correctly confirm, geolocate,
# median-aggregate, and report a flag when a detection DOES occur, since
# the primary run gave nothing to observe that with.  Tracker,
# MedianGeolocatorAdapter, VisionSystem, and Mission are still the exact
# real classes; only the detector's model backend is swapped.
# ---------------------------------------------------------------------------
class _DriftingFlagScript:
    """Scripted stub detections: one static (but slightly jittering) flag
    box every frame, class_id = the briefed target_flag_class."""

    def __init__(self, class_id: int, base_box: tuple[float, float, float, float]) -> None:
        self._class_id = class_id
        self._base_box = base_box
        self._n = 0

    def __call__(self, model_input: np.ndarray) -> list[Detection]:
        i = self._n
        self._n += 1
        x1, y1, x2, y2 = self._base_box
        # Capped at 20 calls' worth of drift (40px total): Mission.step()'s
        # busy loop re-processes the same LatestFrameQueue slot far more
        # often than one call per distinct frame (see dry_run_report.md),
        # so an uncapped per-call drift would run the box off-frame in a
        # fraction of a second - capping keeps this "a static target with a
        # bit of box jitter," matching the stated intent.
        dx = min(i, 20) * 2.0
        return [Detection(box=(x1 + dx, y1, x2 + dx, y2), class_id=self._class_id, score=0.9, coord=None)]


def secondary_run(cfg, briefed_ids: frozenset[int], target_flag_class: int) -> dict:
    log("=" * 70)
    log("SECONDARY CHECK (stub detector - NOT the real model; see header)")
    log("=" * 70)

    scripted = _DriftingFlagScript(target_flag_class, base_box=(260.0, 180.0, 380.0, 260.0))
    detector = Detector(StubBackend(scripted), conf_threshold=0.1, nms_iou=0.45)

    tracker = Tracker(
        iou_threshold=cfg.mission.iou_threshold,
        min_hits=cfg.mission.min_hits,
        max_age=cfg.mission.max_age,
    )
    real_geolocator = MedianGeolocatorAdapter(cfg)
    geolocator = LoggingGeolocatorWrapper(real_geolocator)
    # Same gap window as primary, PLUS a single glitched-gimbal frame - the
    # ~2 km bug's regression check, at the system level (see mission.py
    # module header "THE FIX").
    telemetry = SyntheticTelemetry(gap_window=(1.8, 2.7), glitch_window=(3.3, 3.6))

    capture_source = FileFrameSource(str(FOOTAGE_DIR), realtime=True, fps=FPS)
    frame_queue = LatestFrameQueue(capture_source)

    mission = Mission(
        cfg,
        detector=detector,
        tracker=tracker,
        geolocator=geolocator,
        telemetry=telemetry,
        frame_queue=frame_queue,
        briefed_target_class_ids=briefed_ids,
        judge_log_path=str(JUDGE_LOG_SECONDARY),
    )

    t_run_start = time.monotonic()
    transitions = run_mission_to_completion(mission, "SECONDARY")
    run_time_s = time.monotonic() - t_run_start

    if mission.state is MissionState.HANDOFF_COMPLETE:
        mission.acknowledge_handoff()
        log(f"[SECONDARY] acknowledge_handoff() -> {mission.state.name}")

    frame_queue.close()

    final = mission.final_result()
    history = {tid: list(coords) for tid, coords in real_geolocator._history.items()}

    # Direct evidence for the ~2 km bug regression check: any call whose
    # single-frame point differs meaningfully (>50m, comfortably above GPS
    # jitter) from the median returned that same call.
    def _dist_m(a, b):
        if a is None or b is None:
            return None
        return math.hypot((a[0] - b[0]) * 111320.0, (a[1] - b[1]) * 111320.0 * math.cos(math.radians(a[0])))

    outlier_evidence = [
        {**call, "single_vs_median_m": _dist_m(call["single_frame_coord"], call["median_coord"])}
        for call in geolocator.calls
        if call["single_frame_coord"] is not None
        and _dist_m(call["single_frame_coord"], call["median_coord"]) is not None
        and _dist_m(call["single_frame_coord"], call["median_coord"]) > 50.0
    ]

    return {
        "run_time_s": run_time_s,
        "transitions": transitions,
        "telemetry_log": telemetry.call_log,
        "final_result": final,
        "degraded_count": mission.degraded_count,
        "frame_idx_final": mission._frame_idx,
        "geolocator_history": history,
        "geolocator_calls": geolocator.calls,
        "outlier_evidence": outlier_evidence,
    }


def main() -> None:
    overall_start = time.monotonic()
    build_synthetic_footage()

    with open(WEIGHTS_DIR / "metadata.yaml") as f:
        class_names = yaml.safe_load(f)["names"]
    briefed_ids = resolve_class_ids_by_name(class_names, BRIEFED_NAMES)
    target_flag_class = next(iter(sorted(briefed_ids)))  # Egypt's id (lowest of the four)
    log(f"resolved briefed_target_class_ids={sorted(briefed_ids)} (names={BRIEFED_NAMES}), "
        f"target_flag_class={target_flag_class}")

    camera = {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0,
              "image_width": FRAME_W, "image_height": FRAME_H}
    mission_cfg = {
        "search_altitude_m": 55.0, "ground_speed_mps": 5.0,
        "target_flag_class": target_flag_class,
        "confidence_threshold": 0.5, "iou_threshold": 0.3,
        "min_hits": 2, "max_age": 5, "lane_overlap": 0.2,
    }
    frames_cfg = {"gimbal_reference_frame": "body"}
    cfg = from_dicts(camera, mission_cfg, frames_cfg)

    primary = primary_run(cfg, briefed_ids, target_flag_class)

    secondary = None
    if not primary["all_class_ids_seen"]:
        log("primary run produced ZERO detections (as anticipated for schematic synthetic "
            "footage vs. a model trained on real photos) -> running the secondary wiring-only check")
        secondary = secondary_run(cfg, briefed_ids, target_flag_class)
    else:
        log("primary run DID produce detections; secondary check skipped (not needed)")

    total_time_s = time.monotonic() - overall_start

    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"total wall-clock time: {total_time_s:.3f}s")
    log(f"primary: {primary['frame_idx_final']} process_frame calls in {primary['run_time_s']:.3f}s "
        f"({primary['frame_idx_final']/max(primary['run_time_s'],1e-9):.2f} calls/s), "
        f"{primary['distinct_frame_objects']} distinct frame objects out of "
        f"{primary['total_detect_calls']} detect() calls (of {NUM_FRAMES} source images)")
    log(f"primary final_result: {primary['final_result']}")
    log(f"primary target_class_ids: {primary['target_class_ids']}")
    log(f"primary all_class_ids_seen (should be empty or subset of target ids): {sorted(set(primary['all_class_ids_seen']))}")
    log(f"primary non_target_leak (must be empty): {primary['non_target_leak']}")
    log(f"primary degraded_count: {primary['degraded_count']}")
    primary_gap_calls = [c for c in primary["telemetry_log"] if c.get("gap")]
    log(f"primary telemetry gap: {len(primary_gap_calls)} pose_at() calls returned None "
        f"(first={primary_gap_calls[0] if primary_gap_calls else None}, "
        f"last={primary_gap_calls[-1] if primary_gap_calls else None})")
    if secondary is not None:
        log(f"secondary: {secondary['frame_idx_final']} process_frame calls in {secondary['run_time_s']:.3f}s "
            f"({secondary['frame_idx_final']/max(secondary['run_time_s'],1e-9):.2f} calls/s)")
        log(f"secondary final_result: {secondary['final_result']}")
        log(f"secondary geolocator_history (final, bounded window): {secondary['geolocator_history']}")
        log(f"secondary degraded_count: {secondary['degraded_count']}")
        log(f"secondary outlier_evidence (single-frame vs median diverging by >50m), "
            f"{len(secondary['outlier_evidence'])} such call(s):")
        for ev in secondary["outlier_evidence"][:10]:
            log(f"    {ev}")
        gap_calls = [c for c in secondary["telemetry_log"] if c.get("gap")]
        log(f"secondary telemetry gap: {len(gap_calls)} pose_at() calls returned None "
            f"(first={gap_calls[0] if gap_calls else None}, last={gap_calls[-1] if gap_calls else None})")
        glitch_calls = [c for c in secondary["telemetry_log"] if c.get("glitched_gimbal")]
        log(f"secondary gimbal glitch: {len(glitch_calls)} pose_at() calls returned the "
            f"glitched (-1deg pitch) gimbal (first={glitch_calls[0] if glitch_calls else None}, "
            f"last={glitch_calls[-1] if glitch_calls else None})")

    if JUDGE_LOG_PRIMARY.exists():
        log(f"primary judge log ({JUDGE_LOG_PRIMARY}): {JUDGE_LOG_PRIMARY.read_text()}")
    else:
        log(f"primary judge log NOT WRITTEN at {JUDGE_LOG_PRIMARY}")
    if secondary is not None:
        if JUDGE_LOG_SECONDARY.exists():
            log(f"secondary judge log ({JUDGE_LOG_SECONDARY}): {JUDGE_LOG_SECONDARY.read_text()}")
        else:
            log(f"secondary judge log NOT WRITTEN at {JUDGE_LOG_SECONDARY}")


if __name__ == "__main__":
    main()
