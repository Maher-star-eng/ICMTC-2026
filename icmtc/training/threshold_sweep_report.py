# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   03/07/2026

"""threshold_sweep_report.py - find the optimal masked-pipeline confidence
threshold (ICMTC 2026, Mission 2).

Builds on masked_validation_report.md and resolution_and_remap_report.md,
which established: the 4 weak flags (Egypt, Iraq, Yemen, Syria) are found
(class-agnostic recall @0.25 is 0.89-1.00) but the deployment threshold
(conf=0.6) cuts off their own low-but-real confidence. This script sweeps
conf in {0.25..0.60} under TRUE before-argmax masking (the real deployment
decode) and answers three questions together, because a threshold change is
pipeline-wide, not per-flag:

  Part 1 - does a lower threshold recover the 4 weak flags?
  Part 2 - does it quietly damage the 192 healthy classes (sampled 20)?
  Part 3 - does it invite false positives on the 444 background (no-flag)
           val images?

Every decode step reuses masked_validation_experiment.py's functions
(mv.forward_raw, mv.decode_masked, mv.score_recall) verbatim - which
themselves reuse export.masked_argmax_predictions and
detector.Detector._postprocess / _decode_yolov8 / letterbox. The only new
logic here is greedy one-to-one precision/FP matching (mv only had recall)
and background-image false-positive counting.

Run:
    python threshold_sweep_report.py

Writes threshold_sweep_report.md next to this file. CPU/MPS only, no
training, no dataset/weights mutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import cv2  # noqa: E402

import training.masked_validation_experiment as mv
from vision.detector import _iou_one_to_many
from vision.tracker import Detection

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPORT_PATH = Path(__file__).resolve().parent / "threshold_sweep_report.md"

THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

WEAK_TARGETS = mv.TARGET_NAMES  # Egypt, Iraq, Yemen, Syria
WEAK_DISSIMILAR = mv.DISSIMILAR_NAMES  # Japan, Brazil, Sweden - reused verbatim

# 20 healthy flags spread across the class list (alphabet/region), excluding
# the 4 weak ones. Deliberately includes Japan/Brazil/Sweden (the weak
# flags' own decoy trio) since they are healthy classes in their own right.
HEALTHY_TARGETS = [
    "Japan", "Brazil", "Sweden", "France", "Germany", "Kenya", "Vietnam",
    "Chile", "Nepal", "Canada", "Mexico", "Nigeria", "Poland", "Thailand",
    "Turkey", "Argentina", "Norway", "Ghana", "Indonesia", "Peru",
]
DEFAULT_DECOY_TRIO = ["Japan", "Brazil", "Sweden"]
# Used only for the 3 healthy targets that ARE the default trio, so a
# target never appears as its own decoy.
ALT_DECOY_TRIO = ["Kenya", "Chile", "Nepal"]

BG_BRIEFED_NAMES = ["Egypt", "Japan", "Brazil", "Sweden"]  # Part 3's representative briefed set

RECALL_KNEE_FRACTION = 0.9  # knee = highest threshold retaining >=90% of max swept recall
HEALTHY_F1_DEGRADATION_BUDGET = 0.02  # max acceptable drop in worst-case healthy F1 vs conf=0.6
BG_FP_BUDGET_PER_IMAGE = 0.5  # max acceptable rise in background FP/image vs conf=0.6


@dataclass
class ThresholdMetrics:
    recall: float = float("nan")
    precision: float = float("nan")
    f1: float = float("nan")
    fp_per_image: float = float("nan")
    matched: int = 0
    total_gt: int = 0
    tp: int = 0
    fp: int = 0
    num_images: int = 0


# ---------------------------------------------------------------------------
# New glue: greedy one-to-one precision/FP matching (mv only has recall)
# ---------------------------------------------------------------------------
def precision_fp(
    gt_boxes: list[tuple[float, float, float, float]], dets: list[Detection], target_id: int
) -> tuple[int, int]:
    """Greedy one-to-one match of target-class predictions against target GT
    boxes at IoU>=0.5, highest-score predictions matched first. Returns
    (tp, fp) - tp = predictions that claimed an unused GT box, fp = the rest.
    """
    preds = sorted((d for d in dets if d.class_id == target_id), key=lambda d: -d.score)
    if not preds:
        return 0, 0
    gt_used = [False] * len(gt_boxes)
    tp = 0
    gt_arr = np.array(gt_boxes, dtype=np.float64) if gt_boxes else np.zeros((0, 4))
    for pred in preds:
        if gt_arr.shape[0] == 0:
            continue
        ious = _iou_one_to_many(np.array(pred.box, dtype=np.float64), gt_arr)
        best_idx, best_iou = -1, 0.0
        for i, iou in enumerate(ious):
            if not gt_used[i] and iou >= mv.IOU_MATCH_THRESHOLD and iou > best_iou:
                best_idx, best_iou = i, float(iou)
        if best_idx >= 0:
            gt_used[best_idx] = True
            tp += 1
    fp = len(preds) - tp
    return tp, fp


# ---------------------------------------------------------------------------
# Per-target threshold sweep (Parts 1 and 2 share this)
# ---------------------------------------------------------------------------
def sweep_target(
    net, target_id: int, briefed_ids: list[int], images_and_labels: list[tuple[Path, Path]]
) -> dict[float, ThresholdMetrics]:
    """Cache one raw forward pass per image (threshold-independent), then
    sweep THRESHOLDS cheaply via mv.decode_masked + mv.score_recall +
    precision_fp - all real decode/masking/NMS/letterbox code, unchanged.
    """
    cached = []
    for img_path, label_path in images_and_labels:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  WARNING: failed to decode {img_path}, skipping")
            continue
        h, w = frame.shape[:2]
        gts = mv.load_gt_boxes(label_path, w, h, target_id)
        if not gts:
            continue
        raw, transform = mv.forward_raw(net, frame, mv.INPUT_SIZE)
        cached.append((raw, transform, gts))

    results: dict[float, ThresholdMetrics] = {}
    for t in THRESHOLDS:
        matched_total = tot_gt = tp_total = fp_total = 0
        for raw, transform, gts in cached:
            dets = mv.decode_masked(raw, transform, briefed_ids, t, mv.NMS_IOU)
            m, tot = mv.score_recall(gts, dets, target_id)
            matched_total += m
            tot_gt += tot
            tp, fp = precision_fp(gts, dets, target_id)
            tp_total += tp
            fp_total += fp

        recall = matched_total / tot_gt if tot_gt else float("nan")
        if tp_total + fp_total == 0:
            precision = float("nan")
            f1 = 0.0 if tot_gt else float("nan")
        else:
            precision = tp_total / (tp_total + fp_total)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fp_per_image = fp_total / len(cached) if cached else float("nan")

        results[t] = ThresholdMetrics(
            recall=recall, precision=precision, f1=f1, fp_per_image=fp_per_image,
            matched=matched_total, total_gt=tot_gt, tp=tp_total, fp=fp_total, num_images=len(cached),
        )
    return results


def sweep_background(net, briefed_ids: list[int], bg_images: list[Path]) -> dict[float, dict]:
    """Cache raw forward passes for background images, sweep thresholds.
    Every detection on a background image is a false positive by construction
    (these val images carry no label file - zero ground-truth objects).
    """
    cached = []
    for img_path in bg_images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  WARNING: failed to decode {img_path}, skipping")
            continue
        raw, transform = mv.forward_raw(net, frame, mv.INPUT_SIZE)
        cached.append((raw, transform))

    results: dict[float, dict] = {}
    for t in THRESHOLDS:
        total_fp = 0
        for raw, transform in cached:
            dets = mv.decode_masked(raw, transform, briefed_ids, t, mv.NMS_IOU)
            total_fp += len(dets)
        results[t] = {
            "total_fp": total_fp,
            "fp_per_image": total_fp / len(cached) if cached else float("nan"),
            "num_images": len(cached),
        }
    return results


def find_background_images(images_dir: Path, labels_dir: Path) -> list[Path]:
    """Val images with NO corresponding label file - true negatives."""
    label_stems = {p.stem for p in labels_dir.glob("*.txt")}
    return sorted(p for p in images_dir.iterdir() if p.is_file() and p.stem not in label_stems)


# ---------------------------------------------------------------------------
# Knee / steep-climb analysis
# ---------------------------------------------------------------------------
def recall_knee(metrics_by_threshold: dict[float, ThresholdMetrics]) -> dict:
    """Highest (most conservative) threshold whose recall retains >=90% of
    the max recall seen anywhere in the sweep. If recall never plateaus
    (max is at the lowest swept threshold and nothing else comes close),
    that is reported plainly, not smoothed over.
    """
    recalls = {t: metrics_by_threshold[t].recall for t in THRESHOLDS}
    valid = {t: r for t, r in recalls.items() if r == r}
    if not valid:
        return {"knee_threshold": None, "max_recall": float("nan"), "recall_at_knee": float("nan")}
    max_recall = max(valid.values())
    candidates = [t for t, r in valid.items() if r >= RECALL_KNEE_FRACTION * max_recall]
    knee = max(candidates) if candidates else min(valid.keys())
    return {"knee_threshold": knee, "max_recall": max_recall, "recall_at_knee": recalls[knee]}


def steepest_fp_climb(bg_by_threshold: dict[float, dict]) -> dict | None:
    """The adjacent-threshold step (as threshold decreases) with the largest
    increase in FP/image on background images."""
    thr_desc = sorted(THRESHOLDS, reverse=True)
    deltas = []
    for i in range(len(thr_desc) - 1):
        t_hi, t_lo = thr_desc[i], thr_desc[i + 1]
        d = bg_by_threshold[t_lo]["fp_per_image"] - bg_by_threshold[t_hi]["fp_per_image"]
        deltas.append((t_hi, t_lo, d))
    if not deltas:
        return None
    t_hi, t_lo, d = max(deltas, key=lambda x: x[2])
    return {"from": t_hi, "to": t_lo, "delta_fp_per_image": d, "all_flat": all(x[2] == 0 for x in deltas)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    if not mv.WEIGHTS.exists():
        print(f"STOP: weights not found at {mv.WEIGHTS}")
        return 1
    if not mv.VAL_IMAGES.exists() or not mv.VAL_LABELS.exists():
        print(f"STOP: val images/labels not found under {mv.DATA_ROOT}")
        return 1

    print(f"device: {mv.DEVICE}")
    model = mv.YOLO(str(mv.WEIGHTS))
    net, names = mv.load_net_and_names(model, mv.DEVICE)

    all_wanted = WEAK_TARGETS + WEAK_DISSIMILAR + HEALTHY_TARGETS + DEFAULT_DECOY_TRIO + ALT_DECOY_TRIO + BG_BRIEFED_NAMES
    name_to_id = mv.resolve_class_ids(names, sorted(set(all_wanted)))
    print(f"resolved {len(name_to_id)} class ids")

    weak_dissimilar_ids = [name_to_id[n] for n in WEAK_DISSIMILAR]
    default_decoy_ids = [name_to_id[n] for n in DEFAULT_DECOY_TRIO]
    alt_decoy_ids = [name_to_id[n] for n in ALT_DECOY_TRIO]
    bg_briefed_ids = [name_to_id[n] for n in BG_BRIEFED_NAMES]

    # ---- Part 1: the 4 weak flags ----
    print("\n=== PART 1: weak flags (masked, deployment-faithful) ===")
    part1: dict[str, dict[float, ThresholdMetrics]] = {}
    for target_name in WEAK_TARGETS:
        target_id = name_to_id[target_name]
        briefed_ids = [target_id] + weak_dissimilar_ids
        images = mv.find_images_with_class(mv.VAL_LABELS, mv.VAL_IMAGES, target_id)
        print(f"{target_name} ({len(images)} images):")
        part1[target_name] = sweep_target(net, target_id, briefed_ids, images)
        for t in THRESHOLDS:
            m = part1[target_name][t]
            print(f"  t={t}: recall={m.recall:.3f} precision={m.precision:.3f} f1={m.f1:.3f} fp/img={m.fp_per_image:.3f}")

    # ---- Part 2: 20 healthy flags ----
    print("\n=== PART 2: 20 healthy flags (regression guard) ===")
    part2: dict[str, dict[float, ThresholdMetrics]] = {}
    for target_name in HEALTHY_TARGETS:
        target_id = name_to_id[target_name]
        decoy_ids = alt_decoy_ids if target_name in DEFAULT_DECOY_TRIO else default_decoy_ids
        briefed_ids = [target_id] + decoy_ids
        images = mv.find_images_with_class(mv.VAL_LABELS, mv.VAL_IMAGES, target_id)
        part2[target_name] = sweep_target(net, target_id, briefed_ids, images)
        m6 = part2[target_name][0.6]
        m25 = part2[target_name][0.25]
        print(
            f"{target_name} ({len(images)} images): f1@0.6={m6.f1:.3f} -> f1@0.25={m25.f1:.3f}  "
            f"fp/img@0.6={m6.fp_per_image:.3f} -> fp/img@0.25={m25.fp_per_image:.3f}"
        )

    # ---- Part 3: background images ----
    print("\n=== PART 3: background (no-flag) images ===")
    bg_images = find_background_images(mv.VAL_IMAGES, mv.VAL_LABELS)
    print(f"{len(bg_images)} background images found (expected 444 per training log)")
    part3 = sweep_background(net, bg_briefed_ids, bg_images)
    for t in THRESHOLDS:
        d = part3[t]
        print(f"  t={t}: total_fp={d['total_fp']} fp/img={d['fp_per_image']:.4f}")

    write_report(name_to_id, part1, part2, part3, bg_images)
    print(f"\nReport written to {REPORT_PATH}")
    return 0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def fmt(x: float, nd: int = 3) -> str:
    return "nan" if x != x else f"{x:.{nd}f}"


def write_report(
    name_to_id: dict[str, int],
    part1: dict[str, dict[float, ThresholdMetrics]],
    part2: dict[str, dict[float, ThresholdMetrics]],
    part3: dict[float, dict],
    bg_images: list[Path],
) -> None:
    lines: list[str] = []
    lines.append("# Threshold Sweep Report")
    lines.append("")
    lines.append(
        "Sweeps the masked-pipeline confidence threshold to find the operating point that "
        "recovers the 4 weak flags (Egypt, Iraq, Yemen, Syria) without damaging the 192 "
        "healthy classes or inviting false positives on background images. Builds on "
        "masked_validation_report.md and resolution_and_remap_report.md (not re-derived here)."
    )
    lines.append("")
    lines.append(f"- thresholds swept = {THRESHOLDS}")
    lines.append(f"- imgsz = {mv.INPUT_SIZE}, nms_iou = {mv.NMS_IOU}, IoU match = {mv.IOU_MATCH_THRESHOLD}, device = {mv.DEVICE}")
    lines.append("- decode = TRUE before-argmax masking (`export.masked_argmax_predictions`), NOT Ultralytics `classes=`")
    lines.append("")

    # --- Part 1 ---
    lines.append("## Part 1 - the 4 weak flags (masked)")
    lines.append("")
    lines.append("Briefed set per flag = `[target, Japan, Brazil, Sweden]`, same dissimilar trio as the prior reports.")
    lines.append("")
    for target_name in WEAK_TARGETS:
        lines.append(f"### {target_name} (class id {name_to_id[target_name]})")
        lines.append("")
        lines.append("| threshold | recall | precision | F1 | FP/image |")
        lines.append("|---|---|---|---|---|")
        for t in THRESHOLDS:
            m = part1[target_name][t]
            lines.append(f"| {t} | {fmt(m.recall)} ({m.matched}/{m.total_gt}) | {fmt(m.precision)} | {fmt(m.f1)} | {fmt(m.fp_per_image)} |")
        lines.append("")

    # --- Part 2 ---
    lines.append("## Part 2 - 20 healthy flags (regression guard)")
    lines.append("")
    lines.append(
        "Briefed set per flag = `[target] + 3 dissimilar decoys` (Japan/Brazil/Sweden, or "
        "Kenya/Chile/Nepal for those 3 flags themselves, so a flag never decoys itself)."
    )
    lines.append("")
    lines.append("| threshold | mean recall | mean precision | mean F1 | worst-case F1 (min) | mean FP/image |")
    lines.append("|---|---|---|---|---|---|")
    for t in THRESHOLDS:
        recalls = [part2[n][t].recall for n in HEALTHY_TARGETS if part2[n][t].recall == part2[n][t].recall]
        precs = [part2[n][t].precision for n in HEALTHY_TARGETS if part2[n][t].precision == part2[n][t].precision]
        f1s = [part2[n][t].f1 for n in HEALTHY_TARGETS if part2[n][t].f1 == part2[n][t].f1]
        fps = [part2[n][t].fp_per_image for n in HEALTHY_TARGETS if part2[n][t].fp_per_image == part2[n][t].fp_per_image]
        worst_f1 = min(f1s) if f1s else float("nan")
        worst_name = min(HEALTHY_TARGETS, key=lambda n: part2[n][t].f1 if part2[n][t].f1 == part2[n][t].f1 else 1.0)
        lines.append(
            f"| {t} | {fmt(np.mean(recalls) if recalls else float('nan'))} | "
            f"{fmt(np.mean(precs) if precs else float('nan'))} | "
            f"{fmt(np.mean(f1s) if f1s else float('nan'))} | "
            f"{fmt(worst_f1)} ({worst_name}) | {fmt(np.mean(fps) if fps else float('nan'))} |"
        )
    lines.append("")
    lines.append("### Per-flag detail (Part 2)")
    lines.append("")
    lines.append("| Flag | F1@0.6 | F1@0.25 | FP/img@0.6 | FP/img@0.25 |")
    lines.append("|---|---|---|---|---|")
    for target_name in HEALTHY_TARGETS:
        m6 = part2[target_name][0.6]
        m25 = part2[target_name][0.25]
        lines.append(f"| {target_name} | {fmt(m6.f1)} | {fmt(m25.f1)} | {fmt(m6.fp_per_image)} | {fmt(m25.fp_per_image)} |")
    lines.append("")

    # --- Part 3 ---
    lines.append("## Part 3 - false positives on background images")
    lines.append("")
    lines.append(
        f"{len(bg_images)} background (no-label, no-flag) val images, briefed set = "
        f"`[Egypt, Japan, Brazil, Sweden]` (representative). Every detection here is a false "
        f"positive by construction - these images have zero ground-truth objects."
    )
    lines.append("")
    lines.append("| threshold | total false detections | FP per background image |")
    lines.append("|---|---|---|")
    for t in THRESHOLDS:
        d = part3[t]
        lines.append(f"| {t} | {d['total_fp']} | {fmt(d['fp_per_image'], 4)} |")
    lines.append("")

    # --- Knee analysis ---
    lines.append("## Knee analysis")
    lines.append("")
    lines.append(
        f"Recall knee = highest (most conservative) threshold that still retains >= "
        f"{int(RECALL_KNEE_FRACTION*100)}% of the max recall seen anywhere in the sweep for "
        f"that flag - i.e. where further lowering the threshold stops paying off."
    )
    lines.append("")
    lines.append("| Flag | knee threshold | recall at knee | max recall in sweep (at 0.25) |")
    lines.append("|---|---|---|---|")
    for target_name in WEAK_TARGETS:
        k = recall_knee(part1[target_name])
        note = "" if k["knee_threshold"] != min(THRESHOLDS) else " (never plateaus in this sweep range)"
        lines.append(f"| {target_name} | {k['knee_threshold']}{note} | {fmt(k['recall_at_knee'])} | {fmt(k['max_recall'])} |")
    lines.append("")

    climb = steepest_fp_climb(part3)
    if climb and climb["all_flat"]:
        lines.append(
            f"Background FP/image is flat at 0 across the entire swept range (0.25-0.60) - "
            f"there is no climb to identify; lowering the threshold this far did not invite any "
            f"false detections on this val set's 444 background images."
        )
    elif climb:
        lines.append(
            f"Steepest background-FP climb: between conf={climb['from']} and conf={climb['to']}, "
            f"FP/image rises by {climb['delta_fp_per_image']:.4f} - the single largest step-to-step "
            f"jump in the Part 3 sweep."
        )
    lines.append("")

    # --- Recommendation ---
    lines.append("## Recommendation")
    lines.append("")

    lines.append("### Single global threshold?")
    lines.append("")
    baseline_worst_f1 = min(part2[n][0.6].f1 for n in HEALTHY_TARGETS if part2[n][0.6].f1 == part2[n][0.6].f1)
    baseline_bg_fp = part3[0.6]["fp_per_image"]
    chosen = None
    for t in THRESHOLDS:  # ascending: take the FIRST (lowest) threshold that stays within budget
        worst_f1_t = min((part2[n][t].f1 for n in HEALTHY_TARGETS if part2[n][t].f1 == part2[n][t].f1), default=float("nan"))
        bg_fp_t = part3[t]["fp_per_image"]
        if worst_f1_t == worst_f1_t and (baseline_worst_f1 - worst_f1_t) <= HEALTHY_F1_DEGRADATION_BUDGET and (bg_fp_t - baseline_bg_fp) <= BG_FP_BUDGET_PER_IMAGE:
            chosen = t
            break
    lines.append(
        f"Decision rule (stated, not black-box): pick the LOWEST threshold where worst-case "
        f"healthy-class F1 has not dropped more than {HEALTHY_F1_DEGRADATION_BUDGET} from its "
        f"conf=0.6 value ({fmt(baseline_worst_f1)}), AND background FP/image has not risen more "
        f"than {BG_FP_BUDGET_PER_IMAGE} from its conf=0.6 value ({fmt(baseline_bg_fp, 4)})."
    )
    lines.append("")
    if chosen is not None:
        mean_weak_recall = np.mean([part1[n][chosen].recall for n in WEAK_TARGETS])
        lines.append(
            f"**Recommended global threshold: {chosen}.** At this threshold: mean weak-flag "
            f"recall = {fmt(mean_weak_recall)}, worst-case healthy F1 = "
            f"{fmt(min((part2[n][chosen].f1 for n in HEALTHY_TARGETS if part2[n][chosen].f1==part2[n][chosen].f1), default=float('nan')))}, "
            f"background FP/image = {fmt(part3[chosen]['fp_per_image'], 4)}."
        )
    else:
        lines.append(
            "**No threshold in this sweep stays within budget on both healthy F1 and background "
            "FP.** Lowering the threshold at all costs more than the stated budget somewhere in "
            "Part 2 or Part 3 - see the class-specific recommendation below instead."
        )
    lines.append("")

    lines.append("### Is a class-specific threshold warranted?")
    lines.append("")
    weak_knee_thresholds = [recall_knee(part1[n])["knee_threshold"] for n in WEAK_TARGETS]
    weak_best_t = min(t for t in weak_knee_thresholds if t is not None) if any(weak_knee_thresholds) else min(THRESHOLDS)
    worst_f1_at_weak_best = min((part2[n][weak_best_t].f1 for n in HEALTHY_TARGETS if part2[n][weak_best_t].f1 == part2[n][weak_best_t].f1), default=float("nan"))
    bg_fp_at_weak_best = part3[weak_best_t]["fp_per_image"]
    f1_cost = baseline_worst_f1 - worst_f1_at_weak_best
    bg_cost = bg_fp_at_weak_best - baseline_bg_fp
    lines.append(
        f"At the weak flags' best knee threshold ({weak_best_t}): worst-case healthy F1 = "
        f"{fmt(worst_f1_at_weak_best)} (cost vs 0.6: {f1_cost:+.3f}), background FP/image = "
        f"{fmt(bg_fp_at_weak_best, 4)} (cost vs 0.6: {bg_cost:+.4f})."
    )
    lines.append("")
    if f1_cost > HEALTHY_F1_DEGRADATION_BUDGET or bg_cost > BG_FP_BUDGET_PER_IMAGE:
        lines.append(
            f"**YES - a class-specific threshold is warranted.** The cost at the weak flags' best "
            f"threshold exceeds the stated budget on {'healthy F1' if f1_cost > HEALTHY_F1_DEGRADATION_BUDGET else ''}"
            f"{' and ' if f1_cost > HEALTHY_F1_DEGRADATION_BUDGET and bg_cost > BG_FP_BUDGET_PER_IMAGE else ''}"
            f"{'background FP' if bg_cost > BG_FP_BUDGET_PER_IMAGE else ''}. Recommend: "
            f"**conf={weak_best_t} when the briefed set includes any of {{Egypt, Iraq, Yemen, Syria}}, "
            f"else conf=0.6.** This is also architecturally natural: the briefed target IDs are "
            f"already known at mission-config time, so this is a lookup, not a runtime heuristic."
        )
    else:
        lines.append(
            f"**NO - a single global threshold of {weak_best_t} is sufficient.** The cost to "
            f"healthy classes and background FP rate at this threshold is within the stated "
            f"budget; a class-specific rule would add complexity without a measurable benefit."
        )
    lines.append("")

    lines.append("### Egypt's residual risk")
    lines.append("")
    egypt_recalls = {t: part1["Egypt"][t].recall for t in THRESHOLDS}
    best_t = max(egypt_recalls.items(), key=lambda kv: kv[1])[0]
    best_recall = egypt_recalls[best_t]
    if best_recall >= 0.5:
        lines.append(
            f"Egypt reaches class-correct recall >= 0.5 at conf={best_t} (recall={fmt(best_recall)})."
        )
    else:
        lines.append(
            f"**No threshold in this sweep gets Egypt's class-correct recall above 0.5.** Best in "
            f"sweep: conf={best_t}, recall={fmt(best_recall)}. Despite 0.967 class-agnostic recall "
            f"@0.25 (resolution_and_remap_report.md), Egypt's OWN raw confidence at its own GT "
            f"location is too low, too often, to cross even a permissive threshold with correct "
            f"labeling - this is a genuine residual risk. If Egypt is the briefed target, do not "
            f"expect reliable ID-scoring from threshold-lowering alone; combine with the "
            f"resolution/altitude mitigation from resolution_and_remap_report.md, or budget for "
            f"location-only points on this flag."
        )
    lines.append("")

    lines.append("### Tracker context")
    lines.append("")
    lines.append(
        "`tracker.py`'s `Tracker` requires `hits >= min_hits` (default `min_hits=3`) before a "
        "track is confirmed - a detection has to repeat across consecutive-ish frames before it "
        "commits. The FP/image numbers above are single-frame detector output and are an UPPER "
        "BOUND on real-world impact: transient single-frame false positives (the kind a lower "
        "threshold invites) are exactly what `min_hits` is designed to suppress. This does not "
        "eliminate the false-positive cost - a persistent false signal (e.g. a real flag-colored "
        "object staying in frame) would still confirm - but it means the raw FP/image rate above "
        "overstates the operational false-track rate."
    )
    REPORT_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
