# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   29/06/2026

"""webcam_live_test.py - live webcam smoke test for the real trained detector.

ICMTC 2026, Mission 2. Dev-only tooling (mirrors export.py / dry_run.py's
Mac-only, hardware-free conventions), except this one genuinely needs a
camera: it opens the Mac's webcam and runs the real
`vision.detector.Detector` (fresh ncnn export at
`training/runs/mission2/yolo11n_flags/weights/best_ncnn_model/`) on every
live frame, UNMASKED (`target_class_ids=None`) - the point is to see what
the model actually predicts for a real, physically-held flag, across all
196 classes, not to pre-filter to an expected answer. Per-class confidence
thresholds are the real production ones (`core.config.DEFAULT_CONFIDENCE_
THRESHOLD` / `SPECIAL_CONFIDENCE_THRESHOLDS`), resolved by name exactly like
`build_detector` does everywhere else in this codebase.

This is a live-detection smoke test, not a mission run: no tracker, no
geolocation, no Mission state machine - just capture -> detect -> draw, in
real time, so a human can hold a real flag in front of the camera and watch
the model's own top prediction update live.

Controls (focus must be on the preview window):
    q / ESC   quit; prints a session summary to stdout
    s         save the current frame (raw + overlay) to
              training/webcam_captures/ - useful for later building a real
              (non-synthetic) dry-run footage set, since the existing
              training/dry_run_footage/ is explicitly PLUMBING ONLY
              (schematic, not photographs - see dry_run_report.md).

Run:
    python -m training.webcam_live_test [--camera-index 0] [--mask]

    --mask restricts the live decode to the 4 briefed low-confidence classes
    (Egypt/Iraq/Syria/Yemen) via `Detector.set_target_classes`, exactly like
    a real mission - useful for confirming masking live against a real
    camera feed, but hides any other class the model might (correctly or
    incorrectly) predict for a non-briefed flag. Default: unmasked.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on sys.path

import cv2
import yaml

from core.config import from_dicts
from vision.detector import Detector, build_detector

WEIGHTS_DIR = (
    Path(__file__).resolve().parent
    / "runs/mission2/yolo11n_flags/weights/best_ncnn_model"
)
CAPTURE_DIR = Path(__file__).resolve().parent / "webcam_captures"

BRIEFED_NAMES = ("Egypt", "Iraq", "Syria", "Yemen")

_BOX_COLOR = (0, 220, 0)
_TEXT_COLOR = (255, 255, 255)
_HUD_COLOR = (0, 220, 255)


def _build_live_detector(mask: bool) -> tuple[Detector, dict[int, str]]:
    """Wire the real Detector against the fresh ncnn export (build_detector,
    exactly like core/mission.py's main()), and load its class-name map."""
    with open(WEIGHTS_DIR / "metadata.yaml") as f:
        class_names: dict[int, str] = yaml.safe_load(f)["names"]

    # A minimal, valid AppConfig - only camera intrinsics/mission fields
    # required to satisfy from_dicts(); intrinsics are unused by detect().
    camera = {
        "fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0,
        "image_width": 640, "image_height": 480,
    }
    mission_cfg = {
        "search_altitude_m": 55.0, "ground_speed_mps": 5.0,
        "target_flag_class": 50,  # Egypt; unused by detect() itself
        "confidence_threshold": 0.5, "iou_threshold": 0.3,
        "min_hits": 2, "max_age": 5, "lane_overlap": 0.2,
    }
    frames_cfg = {"gimbal_reference_frame": "body"}
    cfg = from_dicts(camera, mission_cfg, frames_cfg)

    detector = build_detector(
        cfg, backend="ncnn",
        param_path=str(WEIGHTS_DIR / "model.ncnn.param"),
        bin_path=str(WEIGHTS_DIR / "model.ncnn.bin"),
        class_names=class_names,
    )
    if mask:
        name_to_id = {name: cid for cid, name in class_names.items()}
        briefed_ids = frozenset(name_to_id[name] for name in BRIEFED_NAMES)
        detector.set_target_classes(briefed_ids)
    return detector, class_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--mask", action="store_true",
        help="Restrict live decode to the 4 briefed classes (Egypt/Iraq/Syria/Yemen).",
    )
    args = parser.parse_args()

    if not WEIGHTS_DIR.exists():
        raise SystemExit(
            f"ncnn export not found at {WEIGHTS_DIR} - run training/export.py "
            "(or re-export from best.pt) first."
        )

    detector, class_names = _build_live_detector(mask=args.mask)
    print(
        f"[webcam_live_test] detector.target_class_ids="
        f"{sorted(detector.target_class_ids) if detector.target_class_ids else None} "
        f"({'MASKED to briefed set' if args.mask else 'UNMASKED - all 196 classes eligible'})"
    )

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera index {args.camera_index}")

    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    class_hits: dict[str, list[float]] = {}
    frame_count = 0
    saved_count = 0
    t_start = time.monotonic()
    last_log_t = 0.0

    print("[webcam_live_test] running. Focus the preview window: 's' saves a frame, 'q'/ESC quits.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[webcam_live_test] camera read failed; stopping.")
                break
            frame_count += 1

            detections = detector.detect(frame)
            overlay = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = (int(round(v)) for v in det.box)
                name = class_names.get(det.class_id, f"id{det.class_id}")
                label = f"{name} {det.score:.2f}"
                cv2.rectangle(overlay, (x1, y1), (x2, y2), _BOX_COLOR, 2)
                cv2.putText(
                    overlay, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _TEXT_COLOR, 2, cv2.LINE_AA,
                )
                class_hits.setdefault(name, []).append(det.score)

            hud = (
                f"frame={frame_count} dets={len(detections)} "
                f"{'MASKED' if args.mask else 'UNMASKED'} | s: save  q: quit"
            )
            cv2.putText(
                overlay, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                _HUD_COLOR, 2, cv2.LINE_AA,
            )
            cv2.imshow("webcam_live_test (ICMTC Mission 2)", overlay)

            now = time.monotonic()
            if detections and now - last_log_t > 0.5:
                summary = ", ".join(
                    f"{class_names.get(d.class_id, d.class_id)}={d.score:.2f}"
                    for d in detections
                )
                print(f"[{now - t_start:7.2f}s] frame={frame_count} detections: {summary}")
                last_log_t = now

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                break
            if key == ord("s"):
                saved_count += 1
                stamp = time.strftime("%Y%m%d_%H%M%S")
                raw_path = CAPTURE_DIR / f"capture_{stamp}_{saved_count:03d}_raw.png"
                overlay_path = CAPTURE_DIR / f"capture_{stamp}_{saved_count:03d}_overlay.png"
                cv2.imwrite(str(raw_path), frame)
                cv2.imwrite(str(overlay_path), overlay)
                print(f"[webcam_live_test] saved {raw_path.name} (+ overlay)")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    elapsed = time.monotonic() - t_start
    print("=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)
    print(f"frames processed: {frame_count} in {elapsed:.2f}s ({frame_count / max(elapsed, 1e-9):.2f} fps)")
    print(f"frames saved: {saved_count} -> {CAPTURE_DIR}")
    if class_hits:
        print("classes seen (name: hit_count, min/avg/max confidence):")
        for name, scores in sorted(class_hits.items(), key=lambda kv: -len(kv[1])):
            print(
                f"  {name}: {len(scores)} frame(s), "
                f"min={min(scores):.2f} avg={sum(scores)/len(scores):.2f} max={max(scores):.2f}"
            )
    else:
        print("classes seen: none - no detection cleared its confidence threshold in this session.")


if __name__ == "__main__":
    main()
