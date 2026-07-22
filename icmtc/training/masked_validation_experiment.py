# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   28/06/2026

"""masked_validation_experiment.py - does class-masking recover the 4 collapsed flags?

ICMTC 2026, Mission 2.  Standalone validation script (not part of the
production pipeline; imports it read-only).  Empirically tests, per flag in
{Egypt, Iraq, Yemen, Syria}, whether deployment-style before-argmax class
masking recovers recall on that flag's own val images, versus (a) the
unmasked 196-class collapse and (b) Ultralytics' after-argmax `classes=`
filter (a pessimistic lower bound, NOT what real masking does).

CRITICAL: condition (C) reuses `export.masked_argmax_predictions` verbatim -
the exact mask-before-argmax function export.py already verifies against the
real exported tensor.  This script does not reimplement masking; it only
adds the glue to run it over the val set and score recall.  NMS and
letterbox-inversion after masking reuse `detector.Detector._postprocess`
directly (constructed with a no-op StubBackend, since `_postprocess` never
touches `self.backend`), so the only new logic in this file is: raw-tensor
extraction from the live torch model, GT-label parsing, and recall
scoring - everything about *how a detection becomes a detection* is real
production code.

Run:
    python masked_validation_experiment.py

Writes masked_validation_report.md next to this file.  CPU/MPS only, no
training, no dataset/weights mutation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np

import cv2  # noqa: E402  - dev-only image I/O, mirrors export.py's ASSUMPTION E8
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from training.export import masked_argmax_predictions, verify_masking
from vision.detector import (
    Detector,
    RawDetections,
    StubBackend,
    _DEFAULT_NMS_IOU,
    _decode_yolov8,
    _iou_one_to_many,
    _orient_attrs,
    _to_model_blob,
    letterbox,
)
from vision.tracker import Detection

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = _ROOT / "training" / "runs" / "mission2" / "yolo11n_flags" / "weights" / "best.pt"
DATA_ROOT = _ROOT / "training" / "flag detection.v1i.yolov11"
VAL_IMAGES = DATA_ROOT / "valid" / "images"
VAL_LABELS = DATA_ROOT / "valid" / "labels"
REPORT_PATH = Path(__file__).resolve().parent / "masked_validation_report.md"

CONF_THRESHOLD = 0.6  # real deployment threshold, per task spec
IOU_MATCH_THRESHOLD = 0.5  # GT<->prediction match threshold
INPUT_SIZE = 640  # matches args.yaml training imgsz and detector._DEFAULT_INPUT_SIZE
NMS_IOU = _DEFAULT_NMS_IOU  # 0.45, same as real deployment (detector.py ASSUMPTION D3)
# Supplementary operating points for condition C only, to distinguish "masking
# doesn't help at all" from "masking helps but the deployment threshold is too
# strict for these flags' confidence ceiling" - the two have very different
# mitigations (fly lower vs. lower the threshold for a masked-only pipeline).
SUPPLEMENTARY_THRESHOLDS = [0.25, 0.4]

TARGET_NAMES = ["Egypt", "Iraq", "Yemen", "Syria"]
DISSIMILAR_NAMES = ["Japan", "Brazil", "Sweden"]  # white/red-disc, green-yellow-blue, blue/yellow-cross
# Worst-case pairing drops one dissimilar flag to keep the briefed set at 4:
# [target, worst_sibling, D1, D2]. The two kept here match the task's own
# example ([Egypt, Iraq, Japan, Brazil]); Sweden is dropped for this set only.
WORST_CASE_DISSIMILAR_NAMES = ["Japan", "Brazil"]

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Model / class-index resolution
# ---------------------------------------------------------------------------
def resolve_class_ids(names: dict[int, str], wanted: list[str]) -> dict[str, int]:
    """Resolve flag names to class indices via the model's own names dict.

    Raises SystemExit with a precise message if any name is missing or
    duplicated - never guesses.
    """
    resolved: dict[str, int] = {}
    for name in wanted:
        matches = [idx for idx, n in names.items() if n == name]
        if len(matches) != 1:
            print(
                f"STOP: class name {name!r} resolved to {len(matches)} indices "
                f"in the model's names dict (expected exactly 1): {matches}"
            )
            raise SystemExit(1)
        resolved[name] = matches[0]
    return resolved


# ---------------------------------------------------------------------------
# GT label parsing
# ---------------------------------------------------------------------------
def load_gt_boxes(label_path: Path, img_w: int, img_h: int, class_id: int) -> list[tuple[float, float, float, float]]:
    """Parse YOLO-format labels; return original-pixel (x1,y1,x2,y2) boxes for class_id."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cid = int(parts[0])
        if cid != class_id:
            continue
        cx, cy, w, h = (float(v) for v in parts[1:5])
        x1 = (cx - w / 2.0) * img_w
        y1 = (cy - h / 2.0) * img_h
        x2 = (cx + w / 2.0) * img_w
        y2 = (cy + h / 2.0) * img_h
        boxes.append((x1, y1, x2, y2))
    return boxes


def find_images_with_class(labels_dir: Path, images_dir: Path, class_id: int) -> list[tuple[Path, Path]]:
    """Return [(image_path, label_path), ...] for every val image containing class_id."""
    pairs = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        found = any(
            line.split() and int(line.split()[0]) == class_id
            for line in label_path.read_text().splitlines()
            if line.split()
        )
        if not found:
            continue
        candidates = list(images_dir.glob(label_path.stem + ".*"))
        if not candidates:
            print(f"STOP: label file {label_path} has no matching image in {images_dir}")
            raise SystemExit(1)
        pairs.append((candidates[0], label_path))
    return pairs


# ---------------------------------------------------------------------------
# Raw-tensor extraction (the only new inference glue in this file)
# ---------------------------------------------------------------------------
def forward_raw(net: torch.nn.Module, frame: np.ndarray, input_size: int):
    """Letterbox + normalize + forward; return (raw_output as (1,4+nc,N) numpy, transform).

    Mirrors detector.TorchBackend.infer's preprocessing exactly, stopping
    before `_decode_yolov8` so the tensor can be fed to
    `export.masked_argmax_predictions` directly - the same seam export.py's
    `ncnn_forward_raw` exposes for the ncnn path.
    """
    model_input, transform = letterbox(frame, input_size)
    blob = _to_model_blob(model_input)
    with torch.no_grad():
        out = net(torch.from_numpy(blob).to(DEVICE))
    if isinstance(out, (tuple, list)):
        out = out[0]
    arr = out.detach().to("cpu").numpy() if hasattr(out, "detach") else np.asarray(out)
    return arr[None, ...] if arr.ndim == 2 else arr, transform


def _postprocess_via_detector(
    raw_dets: RawDetections, transform, conf: float, nms_iou: float, input_size: int = INPUT_SIZE
) -> list[Detection]:
    """Threshold + per-class NMS + letterbox-inversion, reusing Detector._postprocess verbatim."""
    dummy = Detector(backend=StubBackend([]), input_size=input_size, conf_threshold=conf, nms_iou=nms_iou)
    return dummy._postprocess(raw_dets, transform)


def decode_baseline(
    raw: np.ndarray, transform, conf: float, nms_iou: float, input_size: int = INPUT_SIZE
) -> list[Detection]:
    """Condition A: full 196-class decode (the real production decode path, unmasked)."""
    raw_dets = _decode_yolov8(raw, input_size)
    return _postprocess_via_detector(raw_dets, transform, conf, nms_iou, input_size)


def decode_masked(
    raw: np.ndarray, transform, target_ids: list[int], conf: float, nms_iou: float, input_size: int = INPUT_SIZE
) -> list[Detection]:
    """Condition C: before-argmax masking via export.masked_argmax_predictions, then real postprocess."""
    preds = masked_argmax_predictions(raw, target_ids, conf_threshold=conf)
    if preds:
        boxes = np.array([p.box for p in preds], dtype=np.float64)
        scores = np.array([p.score for p in preds], dtype=np.float64)
        class_ids = np.array([p.class_id for p in preds], dtype=np.int64)
    else:
        boxes = np.zeros((0, 4), dtype=np.float64)
        scores = np.zeros((0,), dtype=np.float64)
        class_ids = np.zeros((0,), dtype=np.int64)
    raw_dets = RawDetections(boxes=boxes, scores=scores, class_ids=class_ids)
    return _postprocess_via_detector(raw_dets, transform, conf, nms_iou, input_size)


def decode_after_argmax_filter(
    model: YOLO, frame: np.ndarray, target_ids: list[int], conf: float, nms_iou: float
) -> list[Detection]:
    """Condition B: Ultralytics' own classes= filter (post-argmax, pessimistic lower bound)."""
    results = model.predict(
        source=frame, imgsz=INPUT_SIZE, conf=conf, iou=nms_iou, classes=list(target_ids),
        device=DEVICE, verbose=False,
    )
    r = results[0]
    boxes = r.boxes
    dets: list[Detection] = []
    if boxes is not None and len(boxes) > 0:
        # boxes.xyxy/.cls/.conf type as Tensor | ndarray (.cpu() only exists on the
        # Tensor half), but this call is immediately after a fresh model.predict(),
        # before anything could have converted the data to numpy - always a live
        # Tensor here (see diagnostics_report.md C-6).
        xyxy = boxes.xyxy.cpu().numpy()  # type: ignore[union-attr]
        cls = boxes.cls.cpu().numpy()  # type: ignore[union-attr]
        confs = boxes.conf.cpu().numpy()  # type: ignore[union-attr]
        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            if x2 <= x1 or y2 <= y1:
                continue
            dets.append(Detection(box=(x1, y1, x2, y2), class_id=int(cls[i]), score=float(confs[i]), coord=None))
    return dets


# ---------------------------------------------------------------------------
# Recall scoring
# ---------------------------------------------------------------------------
@dataclass
class RecallResult:
    matched: int = 0
    total: int = 0

    @property
    def recall(self) -> float:
        return self.matched / self.total if self.total else float("nan")


def score_recall(gt_boxes: list[tuple[float, float, float, float]], dets: list[Detection], target_id: int) -> tuple[int, int]:
    """Return (matched, total) for one image's GT boxes of target_id against dets."""
    cand = np.array([d.box for d in dets if d.class_id == target_id], dtype=np.float64)
    matched = 0
    for gt in gt_boxes:
        if cand.shape[0] > 0:
            ious = _iou_one_to_many(np.array(gt, dtype=np.float64), cand)
            if float(ious.max()) >= IOU_MATCH_THRESHOLD:
                matched += 1
    return matched, len(gt_boxes)


def masked_score_ceiling(raw: np.ndarray, transform, target_id: int, briefed_ids: list[int], gt_boxes: list[tuple[float, float, float, float]]) -> list[float]:
    """For each GT box, the best masked target score among anchors whose center
    falls inside the box - the best this masked pipeline could ever do at that
    GT, independent of conf_threshold. Distinguishes "masking removed the
    competition but raw confidence is still capped" from "no signal at all".
    """
    attrs = _orient_attrs(raw)
    cx, cy = attrs[0, :], attrs[1, :]
    cls_scores = attrs[4:, :]
    masked = cls_scores[briefed_ids, :]  # (4, N)
    target_pos = briefed_ids.index(target_id)
    target_scores = masked[target_pos, :]
    local_argmax = masked.argmax(axis=0)

    ceilings = []
    for gt in gt_boxes:
        gx1, gy1, gx2, gy2 = transform.forward_box(gt)
        anchor_mask = (cx >= gx1) & (cx <= gx2) & (cy >= gy1) & (cy <= gy2)
        if not anchor_mask.any():
            ceilings.append(0.0)
            continue
        # Only anchors where target actually WINS the masked local argmax count -
        # a high target score at an anchor some other briefed class still wins
        # would never surface as a target detection regardless of threshold.
        wins_target = anchor_mask & (local_argmax == target_pos)
        ceilings.append(float(target_scores[wins_target].max()) if wins_target.any() else 0.0)
    return ceilings


# ---------------------------------------------------------------------------
# Guardrail: prove masking is genuinely BEFORE argmax on a real tensor.
# ---------------------------------------------------------------------------
def verify_before_argmax_invariant(
    net: torch.nn.Module,
    target_name: str,
    target_id: int,
    sibling_ids: list[int],
    dissimilar_ids: list[int],
    images_and_labels: list[tuple[Path, Path]],
) -> tuple[bool, str, Counter]:
    """Find a real anchor where the GLOBAL argmax (over all 196) is a sibling
    class but the LOCAL argmax (masked to target+3 dissimilar) is the target -
    proving masking re-assigns rather than drops. Also returns a Counter of
    global-argmax winners at every GT-covering anchor across this target's
    images, used later to pick the empirical worst-case sibling.

    Returns (found, detail_message, confusion_counter).
    """
    confusion: Counter = Counter()
    candidate_target_set = [target_id] + dissimilar_ids
    found_detail = ""
    found = False

    for img_path, label_path in images_and_labels:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        gts = load_gt_boxes(label_path, w, h, target_id)
        if not gts:
            continue

        raw, transform = forward_raw(net, frame, INPUT_SIZE)
        attrs = _orient_attrs(raw)  # (4+nc, N)
        cx, cy = attrs[0, :], attrs[1, :]
        cls_scores = attrs[4:, :]  # (nc, N)
        global_argmax = cls_scores.argmax(axis=0)  # (N,)

        masked_scores = cls_scores[candidate_target_set, :]  # (4, N)
        local_argmax_pos = masked_scores.argmax(axis=0)  # (N,) index into candidate_target_set
        local_argmax_class = np.array(candidate_target_set)[local_argmax_pos]

        for gt in gts:
            gx1, gy1, gx2, gy2 = transform.forward_box(gt)
            anchor_mask = (cx >= gx1) & (cx <= gx2) & (cy >= gy1) & (cy <= gy2)
            if not anchor_mask.any():
                continue
            global_winners = global_argmax[anchor_mask]
            confusion.update(int(v) for v in global_winners.tolist())

            if not found:
                local_winners = local_argmax_class[anchor_mask]
                offending = np.where((np.isin(global_winners, sibling_ids)) & (local_winners == target_id))[0]
                if offending.size > 0:
                    anchor_positions = np.where(anchor_mask)[0]
                    pick = anchor_positions[offending[0]]
                    global_cls = int(global_argmax[pick])
                    global_score = float(cls_scores[global_cls, pick])
                    target_score = float(cls_scores[target_id, pick])
                    found_detail = (
                        f"image={img_path.name} anchor={pick}: global argmax=class {global_cls} "
                        f"(score={global_score:.4f}) but masked (target+3 dissimilar) argmax="
                        f"target class {target_id} (score={target_score:.4f}) - re-assigned, not dropped."
                    )
                    found = True

    return found, found_detail, confusion


def load_net_and_names(model: YOLO, device: str) -> tuple[torch.nn.Module, dict[int, str]]:
    """Extract the underlying nn.Module + class-name map from a loaded YOLO model.

    Every caller checks WEIGHTS.exists() before constructing `model`, and WEIGHTS is
    hardcoded to end in .pt, which forces ultralytics' Model._load() down its
    load_checkpoint() branch every time - model.model is always a real nn.Module
    here, never the None/str branches ultralytics' own wide union type allows for
    other weight formats (see diagnostics_report.md C-7). resolution_and_remap_report.py
    and threshold_sweep_report.py call this same helper (as mv.load_net_and_names)
    instead of duplicating this attribute access.
    """
    assert isinstance(model.model, torch.nn.Module)
    net = model.model.to(device)
    net.eval()
    # ultralytics attaches `.names` dynamically to the wrapped nn.Module; nn.Module's
    # own __getattr__ stub (for registered params/buffers/submodules) can't know
    # about it, so pyright infers Tensor | Module instead of dict[int, str] here.
    names = cast("dict[int, str]", model.model.names)
    return net, names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    if not WEIGHTS.exists():
        print(f"STOP: weights not found at {WEIGHTS}")
        return 1
    if not VAL_IMAGES.exists() or not VAL_LABELS.exists():
        print(f"STOP: val images/labels not found under {DATA_ROOT}")
        return 1

    print(f"device: {DEVICE}")
    model = YOLO(str(WEIGHTS))
    net, names = load_net_and_names(model, DEVICE)

    all_wanted = TARGET_NAMES + DISSIMILAR_NAMES
    name_to_id = resolve_class_ids(names, all_wanted)
    print("resolved class ids:")
    for n in all_wanted:
        print(f"  {n}: {name_to_id[n]}")

    for n in DISSIMILAR_NAMES:
        assert n not in TARGET_NAMES

    dissimilar_ids = [name_to_id[n] for n in DISSIMILAR_NAMES]
    worst_case_dissimilar_ids = [name_to_id[n] for n in WORST_CASE_DISSIMILAR_NAMES]
    target_ids_all = {n: name_to_id[n] for n in TARGET_NAMES}

    # ---- Guardrail: before-argmax invariant + empirical confusion tally ----
    print("\n=== GUARDRAIL: verifying masking re-assigns (not drops) ===")
    invariant_ok = True
    confusion_by_target: dict[str, Counter] = {}
    images_by_target: dict[str, list[tuple[Path, Path]]] = {}

    for target_name in TARGET_NAMES:
        target_id = target_ids_all[target_name]
        sibling_ids = [target_ids_all[n] for n in TARGET_NAMES if n != target_name]
        images_and_labels = find_images_with_class(VAL_LABELS, VAL_IMAGES, target_id)
        images_by_target[target_name] = images_and_labels
        print(f"{target_name} (id={target_id}): {len(images_and_labels)} val images with GT")

        found, detail, confusion = verify_before_argmax_invariant(
            net, target_name, target_id, sibling_ids, dissimilar_ids, images_and_labels
        )
        confusion_by_target[target_name] = confusion
        if found:
            print(f"  [CONFIRMED] {detail}")
        else:
            print(f"  [NOT FOUND] no sibling-wins-globally/target-wins-masked anchor found for {target_name}")
            invariant_ok = False

    if not invariant_ok:
        print(
            "\nSTOP: could not confirm the before-argmax re-assignment invariant for "
            "at least one target flag. Refusing to produce a masked-recall report "
            "that might be silently relying on drop-not-reassign behavior."
        )
        return 1

    # sanity: verify_masking (export.py's own leak check) on one real tensor
    sample_target = TARGET_NAMES[0]
    sample_frame = None
    for candidate_img, _ in images_by_target[sample_target]:
        sample_frame = cv2.imread(str(candidate_img))
        if sample_frame is None:
            print(f"  WARNING: failed to decode {candidate_img}, skipping")
            continue
        break
    if sample_frame is None:
        print(f"STOP: no decodable sample image found for {sample_target} sanity check.")
        return 1
    sample_raw, _ = forward_raw(net, sample_frame, INPUT_SIZE)
    sample_target_ids = [target_ids_all[sample_target]] + dissimilar_ids
    mreport = verify_masking(sample_raw, sample_target_ids, conf_threshold=CONF_THRESHOLD)
    print(f"\nexport.verify_masking sanity check: passed={mreport.passed} detail={mreport.detail}")
    if not mreport.passed:
        print("STOP: export.py's own masking-leak check failed on a real tensor.")
        return 1

    # ---- Main experiment: A / B / C / C-worstcase per target ----
    print("\n=== MAIN EXPERIMENT ===")
    results: dict[str, dict[str, RecallResult]] = {}
    worst_case_partner: dict[str, str] = {}
    ceiling_summaries: dict[str, np.ndarray] = {}
    decode_stats: dict[str, tuple[int, int, int]] = {}  # target -> (scored, skipped, candidates)

    for target_name in TARGET_NAMES:
        target_id = target_ids_all[target_name]
        images_and_labels = images_by_target[target_name]
        briefed_ids = [target_id] + dissimilar_ids

        confusion = confusion_by_target[target_name]
        sibling_names = [n for n in TARGET_NAMES if n != target_name]
        sibling_counts = {n: confusion.get(target_ids_all[n], 0) for n in sibling_names}
        worst_sibling = max(sibling_counts.items(), key=lambda kv: kv[1])[0]
        worst_case_partner[target_name] = worst_sibling
        worst_case_ids = [target_id, target_ids_all[worst_sibling]] + worst_case_dissimilar_ids
        print(
            f"\n{target_name}: empirical sibling-confusion tally at GT anchors = {sibling_counts} "
            f"-> worst-case partner = {worst_sibling}"
        )

        r_a, r_b, r_c, r_cw = RecallResult(), RecallResult(), RecallResult(), RecallResult()
        r_supp = {t: RecallResult() for t in SUPPLEMENTARY_THRESHOLDS}
        r_a_supp = {t: RecallResult() for t in SUPPLEMENTARY_THRESHOLDS}
        ceilings: list[float] = []
        scored_count = 0
        skipped_count = 0

        for img_path, label_path in images_and_labels:
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"  WARNING: failed to decode {img_path}, skipping")
                skipped_count += 1
                continue
            h, w = frame.shape[:2]
            gts = load_gt_boxes(label_path, w, h, target_id)
            if not gts:
                continue
            scored_count += 1

            raw, transform = forward_raw(net, frame, INPUT_SIZE)

            dets_a = decode_baseline(raw, transform, CONF_THRESHOLD, NMS_IOU)
            dets_b = decode_after_argmax_filter(model, frame, briefed_ids, CONF_THRESHOLD, NMS_IOU)
            dets_c = decode_masked(raw, transform, briefed_ids, CONF_THRESHOLD, NMS_IOU)
            dets_cw = decode_masked(raw, transform, worst_case_ids, CONF_THRESHOLD, NMS_IOU)

            for r, dets in ((r_a, dets_a), (r_b, dets_b), (r_c, dets_c), (r_cw, dets_cw)):
                m, t = score_recall(gts, dets, target_id)
                r.matched += m
                r.total += t

            for thresh in SUPPLEMENTARY_THRESHOLDS:
                dets_supp = decode_masked(raw, transform, briefed_ids, thresh, NMS_IOU)
                m, t = score_recall(gts, dets_supp, target_id)
                r_supp[thresh].matched += m
                r_supp[thresh].total += t

                # Control: does lowering the threshold help WITHOUT masking too?
                # If baseline-at-low-threshold recall is much lower than
                # masked-at-low-threshold recall, the gain is attributable to
                # masking (removing the sibling's competing label), not merely
                # to a laxer threshold that would help regardless.
                dets_a_supp = decode_baseline(raw, transform, thresh, NMS_IOU)
                m, t = score_recall(gts, dets_a_supp, target_id)
                r_a_supp[thresh].matched += m
                r_a_supp[thresh].total += t

            ceilings.extend(masked_score_ceiling(raw, transform, target_id, briefed_ids, gts))

        results[target_name] = {
            "A": r_a, "B": r_b, "C": r_c, "CW": r_cw,
            **{f"C@{t}": r_supp[t] for t in SUPPLEMENTARY_THRESHOLDS},
            **{f"A@{t}": r_a_supp[t] for t in SUPPLEMENTARY_THRESHOLDS},
        }
        ceiling_arr = np.array(ceilings) if ceilings else np.zeros(0)
        ceiling_summaries[target_name] = ceiling_arr
        decode_stats[target_name] = (scored_count, skipped_count, len(images_and_labels))
        print(
            f"  processed: {scored_count} scored, {skipped_count} skipped (decode failure) "
            f"of {len(images_and_labels)} candidate images"
        )
        print(
            f"  A(baseline)={r_a.recall:.3f} ({r_a.matched}/{r_a.total})  "
            f"B(after-argmax)={r_b.recall:.3f} ({r_b.matched}/{r_b.total})  "
            f"C(true masking)={r_c.recall:.3f} ({r_c.matched}/{r_c.total})  "
            f"CW(worst-case, +{worst_sibling})={r_cw.recall:.3f} ({r_cw.matched}/{r_cw.total})"
        )
        supp_str = "  ".join(
            f"C@{t}={r_supp[t].recall:.3f} (vs unmasked A@{t}={r_a_supp[t].recall:.3f})"
            for t in SUPPLEMENTARY_THRESHOLDS
        )
        print(f"  supplementary: {supp_str}")
        if ceiling_arr.size:
            print(
                f"  masked score ceiling (best masked-argmax-winning score per GT box): "
                f"mean={ceiling_arr.mean():.3f} median={np.median(ceiling_arr):.3f} max={ceiling_arr.max():.3f}"
            )

    write_report(results, name_to_id, worst_case_partner, ceiling_summaries, decode_stats)
    print(f"\nReport written to {REPORT_PATH}")
    return 0


def verdict_for(recall: float) -> str:
    if recall != recall:  # NaN guard
        return "NO GT DATA"
    if recall >= 0.9:
        return "RECOVERED"
    return "STILL WEAK"


def write_report(
    results: dict[str, dict[str, RecallResult]],
    name_to_id: dict[str, int],
    worst_case_partner: dict[str, str],
    ceiling_summaries: dict[str, np.ndarray],
    decode_stats: dict[str, tuple[int, int, int]],
) -> None:
    lines = []
    lines.append("# Masked Validation Report")
    lines.append("")
    lines.append(
        "Empirical test of whether deployment-style before-argmax class masking "
        "recovers recall on the 4 red-white-black Arab tricolor flags that "
        "collapsed in full 196-class validation."
    )
    lines.append("")
    lines.append(f"- conf_threshold = {CONF_THRESHOLD} (real deployment threshold)")
    lines.append(f"- nms_iou = {NMS_IOU}")
    lines.append(f"- IoU match threshold = {IOU_MATCH_THRESHOLD}")
    lines.append(f"- input_size = {INPUT_SIZE}")
    lines.append(f"- device = {DEVICE}")
    lines.append("")
    lines.append("## Class indices used")
    lines.append("")
    lines.append("| Flag | class id |")
    lines.append("|---|---|")
    for n in TARGET_NAMES + DISSIMILAR_NAMES:
        lines.append(f"| {n} | {name_to_id[n]} |")
    lines.append("")
    lines.append("## Image processing coverage")
    lines.append("")
    lines.append(
        "Per target, images are candidates only if their GT labels contain that "
        "target's class id (a given image can be a candidate for more than one "
        "target). `scored` = successfully decoded and used for recall scoring; "
        "`skipped` = cv2.imread failed to decode the file (corrupt/missing) and "
        "the image was excluded, never counted as a valid sample."
    )
    lines.append("")
    lines.append("| Target | scored | skipped (decode failure) | candidates |")
    lines.append("|---|---|---|---|")
    total_scored = total_skipped = total_candidates = 0
    for n in TARGET_NAMES:
        scored, skipped, candidates = decode_stats[n]
        total_scored += scored
        total_skipped += skipped
        total_candidates += candidates
        lines.append(f"| {n} | {scored} | {skipped} | {candidates} |")
    lines.append(f"| **all targets** | **{total_scored}** | **{total_skipped}** | **{total_candidates}** |")
    lines.append("")
    if total_skipped:
        lines.append(
            f"**{total_skipped} image(s) were skipped due to decode failure across all "
            f"targets combined - the results above reflect a partial sweep for those "
            f"targets.**"
        )
    else:
        lines.append(
            "No images were skipped due to decode failure - every candidate image for "
            "every target was successfully scored."
        )
    lines.append("")
    lines.append(
        f"Dissimilar trio used for the briefed 4-flag set (all targets): "
        f"{', '.join(DISSIMILAR_NAMES)} - white field/red disc, green field/yellow "
        f"rhombus/blue globe, blue field/yellow Nordic cross respectively; none is "
        f"a horizontal red-white-black tricolor."
    )
    lines.append("")
    lines.append(
        f"C-worstcase set per target = [target, empirical worst sibling, "
        f"{WORST_CASE_DISSIMILAR_NAMES[0]}, {WORST_CASE_DISSIMILAR_NAMES[1]}] (Sweden "
        f"dropped to keep the set at 4, matching the task's own worked example). The "
        f"\"worst sibling\" is not arbitrary: for each target, every val image's GT "
        f"box is intersected with anchor centers, and the GLOBAL (full 196-class) "
        f"argmax winner at each such anchor is tallied across the other 3 "
        f"red-white-black flags; the most frequent winner is used as the worst-case "
        f"partner. Per-target partner used: "
        + ", ".join(f"{n}→{worst_case_partner[n]}" for n in TARGET_NAMES)
        + "."
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        "| Target | A: baseline (196-class) | B: after-argmax filter (`classes=`) | "
        "C: true before-argmax masking | C-worstcase (+2nd sibling) | Verdict (on C) |"
    )
    lines.append("|---|---|---|---|---|---|")
    for target_name in TARGET_NAMES:
        target_results = results[target_name]
        partner = worst_case_partner[target_name]
        lines.append(
            f"| {target_name} | {target_results['A'].recall:.3f} ({target_results['A'].matched}/{target_results['A'].total}) | "
            f"{target_results['B'].recall:.3f} ({target_results['B'].matched}/{target_results['B'].total}) | "
            f"{target_results['C'].recall:.3f} ({target_results['C'].matched}/{target_results['C'].total}) | "
            f"{target_results['CW'].recall:.3f} ({target_results['CW'].matched}/{target_results['CW'].total}, +{partner}) | "
            f"{verdict_for(target_results['C'].recall)} |"
        )
    lines.append("")
    lines.append("## Per-flag verdicts")
    lines.append("")
    for target_name in TARGET_NAMES:
        r = results[target_name]["C"]
        lines.append(f"- **{target_name}**: {verdict_for(r.recall)} (true-masking recall = {r.recall:.3f})")
    lines.append("")
    lines.append("## Diagnostic: is the deployment threshold (0.6) itself the bottleneck?")
    lines.append("")
    lines.append(
        "For each target, the masked score ceiling is the best target-class score "
        "achieved (under true before-argmax masking, at any anchor overlapping a GT "
        "box) where the target *already wins* the restricted argmax - i.e. the best "
        "this masked pipeline could ever report for that GT, independent of "
        "conf_threshold. If the ceiling is well below 0.6, masking has correctly "
        "removed the sibling competition but the network's absolute confidence in "
        "the correct class is still too low to clear the deployment threshold - a "
        "different failure mode than \"masking doesn't help\", with a different fix "
        "(lower the threshold for a masked-only pipeline, or get more pixels on "
        "target) rather than \"this is unrecoverable\"."
    )
    lines.append("")
    lines.append(
        "A@t is the UNMASKED (full 196-class) recall at that same lower threshold - "
        "the control. If C@t clears A@t by a wide margin, the gain is attributable "
        "to masking (removing the sibling's competing label), not merely to a "
        "laxer threshold that would have helped anyway."
    )
    lines.append("")
    header = "| Target | ceiling mean | ceiling median | ceiling max | C@0.6 |"
    sep = "|---|---|---|---|---|"
    for t in SUPPLEMENTARY_THRESHOLDS:
        header += f" C@{t} | A@{t} (control) |"
        sep += "---|---|"
    lines.append(header)
    lines.append(sep)
    for target_name in TARGET_NAMES:
        c = ceiling_summaries[target_name]
        target_results = results[target_name]
        mean_s = f"{c.mean():.3f}" if c.size else "n/a"
        med_s = f"{np.median(c):.3f}" if c.size else "n/a"
        max_s = f"{c.max():.3f}" if c.size else "n/a"
        row = f"| {target_name} | {mean_s} | {med_s} | {max_s} | {target_results['C'].recall:.3f} ({target_results['C'].matched}/{target_results['C'].total}) |"
        for t in SUPPLEMENTARY_THRESHOLDS:
            row += (
                f" {target_results[f'C@{t}'].recall:.3f} ({target_results[f'C@{t}'].matched}/{target_results[f'C@{t}'].total}) |"
                f" {target_results[f'A@{t}'].recall:.3f} ({target_results[f'A@{t}'].matched}/{target_results[f'A@{t}'].total}) |"
            )
        lines.append(row)
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    recovered = [n for n in TARGET_NAMES if verdict_for(results[n]["C"].recall) == "RECOVERED"]
    weak = [n for n in TARGET_NAMES if verdict_for(results[n]["C"].recall) == "STILL WEAK"]
    if recovered:
        lines.append(
            f"Masking recovers recall for: {', '.join(recovered)}. For these flags the "
            f"196-class collapse was a competition artifact (siblings splitting the "
            f"score mass), not a genuine pixels-on-target limit - the visual signal "
            f"was there all along, just outvoted."
        )
        lines.append("")
    if weak:
        lines.append(
            f"Masking does NOT recover, at conf=0.6: {', '.join(weak)}. For each of "
            f"these, check the ceiling table above before concluding it is a pure "
            f"resolution limit:"
        )
        for n in weak:
            c = ceiling_summaries[n]
            target_results = results[n]
            ceiling_max = float(c.max()) if c.size else 0.0
            if ceiling_max >= CONF_THRESHOLD:
                lines.append(
                    f"  - **{n}**: the masked score ceiling reaches {ceiling_max:.3f} on at "
                    f"least one GT box (>= {CONF_THRESHOLD}), yet recall@0.6 is "
                    f"{target_results['C'].recall:.3f} - the model IS occasionally confident enough even "
                    f"masked, but not consistently. Genuine partial resolution limit, not a "
                    f"pure threshold artifact."
                )
            elif any(target_results[f"C@{t}"].recall > target_results["C"].recall + 1e-9 for t in SUPPLEMENTARY_THRESHOLDS):
                best_t = max(SUPPLEMENTARY_THRESHOLDS, key=lambda t: target_results[f"C@{t}"].recall)
                masked_gain = target_results[f"C@{best_t}"].recall - target_results[f"A@{best_t}"].recall
                lines.append(
                    f"  - **{n}**: masking correctly removes the sibling competition (see "
                    f"the before-argmax invariant above), but the network's own confidence "
                    f"in {n} tops out at {ceiling_max:.3f} - below the 0.6 deployment "
                    f"threshold. Recall at a lower operating point (conf={best_t}) rises to "
                    f"{target_results[f'C@{best_t}'].recall:.3f} masked vs. only {target_results[f'A@{best_t}'].recall:.3f} "
                    f"unmasked at the same threshold (control) - a "
                    f"{masked_gain:+.3f} recall gain attributable to masking itself, not to "
                    f"the lower threshold alone. This is a THRESHOLD problem, not "
                    f"strictly a resolution problem: consider a lower conf_threshold "
                    f"specifically when the masked target set is known, since masking "
                    f"already eliminates the false-positive risk from siblings that a "
                    f"lower global threshold would otherwise invite."
                )
            else:
                lines.append(
                    f"  - **{n}**: the masked score ceiling itself is only {ceiling_max:.3f} "
                    f"and recall does not recover even at conf={min(SUPPLEMENTARY_THRESHOLDS)}. "
                    f"This is a genuine pixels-on-target resolution limit at ~45px, not a "
                    f"competition or threshold artifact - masking cannot fix it. Treat as "
                    f"high-risk if briefed as the target: budget for a lower pass altitude "
                    f"or accept location-only scoring if flown at ~50m."
                )
        lines.append("")
    lines.append(
        "The A vs C gap (where recall jumps from baseline to true masking) is the "
        "part of the 196-class collapse attributable purely to sibling competition; "
        "the B column is shown only as the pessimistic (and incorrect-for-deployment) "
        "lower bound from Ultralytics' own `classes=` filter, which deletes "
        "misclassified-as-sibling detections instead of recovering them."
    )
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
