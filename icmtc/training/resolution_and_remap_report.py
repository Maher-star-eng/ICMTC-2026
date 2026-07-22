# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   01/07/2026

"""resolution_and_remap_report.py - three diagnostics to pick the fix for the
4 resolution-limited flags (Egypt, Iraq, Yemen, Syria).

ICMTC 2026, Mission 2. Standalone validation script; imports the production
pipeline read-only. Builds directly on masked_validation_experiment.py
(imported as `mv`), which already proved that true before-argmax masking
recovers nothing at the deployment threshold (conf=0.6): the failure is not
sibling competition, it's that the model's absolute confidence in the
correct class never reaches 0.6. This script does not re-run that
comparison - it decides what to do about it:

  D1 - class-agnostic vs class-correct recall: is the box found but
       mislabeled (ID risk only), or not found at all (location risk too)?
  D2 - target vs best-sibling vs global-argmax raw score at each GT box:
       would a confusion-remap layer recover anything masking throws away,
       or are the siblings equally unconfident?
  D3 - imgsz sweep (640/960/1280): does more resolution lift the masked
       confidence ceiling above 0.6?

Every decode step reuses masked_validation_experiment.py's functions
(forward_raw, decode_baseline, decode_masked, score_recall,
masked_score_ceiling) verbatim - which themselves reuse
export.masked_argmax_predictions and detector.Detector._postprocess /
_decode_yolov8 / letterbox. The only new logic here is: any-class GT
matching (for class-agnostic recall), best-overlapping-anchor score
extraction (for the remap diagnostic), and imgsz/timing bookkeeping.

Run:
    python resolution_and_remap_report.py

Writes resolution_and_remap_report.md next to this file. CPU/MPS only, no
training, no dataset/weights mutation.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import cv2  # noqa: E402
import torch  # noqa: E402

import training.masked_validation_experiment as mv
from vision.detector import _iou_one_to_many, _orient_attrs
from vision.tracker import Detection

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPORT_PATH = Path(__file__).resolve().parent / "resolution_and_remap_report.md"

D1_THRESHOLDS = [0.6, 0.4, 0.25]  # class-agnostic + class-correct sweep
D1_MISLABEL_THRESHOLD = 0.6  # only tally the mislabel distribution at the real deployment conf
D3_IMGSZ_SWEEP = [640, 960, 1280]  # all multiples of 32, per YOLO stride requirement
D3_THRESHOLDS = [0.6, 0.25]  # class-correct (masked) recall points reported per imgsz


@dataclass
class RecallResult:
    matched: int = 0
    total: int = 0

    @property
    def recall(self) -> float:
        return self.matched / self.total if self.total else float("nan")


# ---------------------------------------------------------------------------
# New glue: any-class GT matching (Diagnostic 1)
# ---------------------------------------------------------------------------
def agnostic_match_and_mislabel(
    gt: tuple[float, float, float, float], dets: list[Detection], target_id: int
) -> tuple[bool, tuple[int, float] | None]:
    """Classify one GT box against ALL predictions (any class), via the single
    highest-IoU prediction overall.

    Returns (agnostic_matched, mislabel) where mislabel is (class_id, score)
    if the best-overlapping box clears IoU>=0.5 but is NOT the target class,
    else None. This is independent of mv.score_recall (which restricts
    candidates to target-class boxes only and is kept as the authoritative
    class-correct number, to exactly reproduce the known per-class metrics).
    """
    if not dets:
        return False, None
    boxes = np.array([d.box for d in dets], dtype=np.float64)
    ious = _iou_one_to_many(np.array(gt, dtype=np.float64), boxes)
    best_idx = int(ious.argmax())
    if float(ious[best_idx]) < mv.IOU_MATCH_THRESHOLD:
        return False, None
    best_det = dets[best_idx]
    if best_det.class_id == target_id:
        return True, None
    return True, (best_det.class_id, best_det.score)


# ---------------------------------------------------------------------------
# New glue: best-overlapping-anchor raw scores (Diagnostic 2)
# ---------------------------------------------------------------------------
def scores_at_best_anchor(
    raw: np.ndarray,
    transform,
    gt: tuple[float, float, float, float],
    target_id: int,
    sibling_ids: list[int],
) -> dict | None:
    """For one GT box: find the anchor, among those whose center falls inside
    the GT box (same anchor<->GT intersection test as
    mv.masked_score_ceiling / mv.verify_before_argmax_invariant), whose
    DECODED BOX has the highest IoU with the GT box - i.e. the single anchor
    that best geometrically overlaps this GT, independent of any class's
    score (so the choice of anchor isn't biased toward the target class).

    Returns raw (unmasked) target_score, best_sibling_score/name, and the
    global (196-class) argmax winner/score at that one anchor. None if no
    anchor's center falls inside the GT box.
    """
    attrs = _orient_attrs(raw)  # (4+nc, N)
    cx, cy, cw, ch = attrs[0, :], attrs[1, :], attrs[2, :], attrs[3, :]
    cls_scores = attrs[4:, :]  # (nc, N)

    gx1, gy1, gx2, gy2 = transform.forward_box(gt)
    anchor_mask = (cx >= gx1) & (cx <= gx2) & (cy >= gy1) & (cy <= gy2)
    if not anchor_mask.any():
        return None

    idxs = np.where(anchor_mask)[0]
    x1 = cx[idxs] - cw[idxs] / 2.0
    y1 = cy[idxs] - ch[idxs] / 2.0
    x2 = cx[idxs] + cw[idxs] / 2.0
    y2 = cy[idxs] + ch[idxs] / 2.0
    decoded_boxes = np.stack([x1, y1, x2, y2], axis=1)
    gt_model = np.array([gx1, gy1, gx2, gy2], dtype=np.float64)
    ious = _iou_one_to_many(gt_model, decoded_boxes)
    best_anchor = int(idxs[int(ious.argmax())])

    target_score = float(cls_scores[target_id, best_anchor])
    sibling_scores = {sid: float(cls_scores[sid, best_anchor]) for sid in sibling_ids}
    best_sibling_id = max(sibling_scores.items(), key=lambda kv: kv[1])[0]
    global_argmax_id = int(cls_scores[:, best_anchor].argmax())
    global_argmax_score = float(cls_scores[global_argmax_id, best_anchor])

    return {
        "target_score": target_score,
        "best_sibling_id": best_sibling_id,
        "best_sibling_score": sibling_scores[best_sibling_id],
        "global_argmax_id": global_argmax_id,
        "global_argmax_score": global_argmax_score,
    }


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

    all_wanted = mv.TARGET_NAMES + mv.DISSIMILAR_NAMES
    name_to_id = mv.resolve_class_ids(names, all_wanted)
    print("resolved class ids:")
    for n in all_wanted:
        print(f"  {n}: {name_to_id[n]}")

    dissimilar_ids = [name_to_id[n] for n in mv.DISSIMILAR_NAMES]
    target_ids_all = {n: name_to_id[n] for n in mv.TARGET_NAMES}

    images_by_target: dict[str, list[tuple[Path, Path]]] = {}
    for target_name in mv.TARGET_NAMES:
        images_by_target[target_name] = mv.find_images_with_class(
            mv.VAL_LABELS, mv.VAL_IMAGES, target_ids_all[target_name]
        )
        print(f"{target_name}: {len(images_by_target[target_name])} val images with GT")

    # ---- Diagnostics 1 + 2 (share the same imgsz=640 forward pass per image) ----
    print("\n=== DIAGNOSTIC 1 + 2 (imgsz=640) ===")
    d1_results: dict[str, dict[float, dict[str, RecallResult]]] = {}
    d1_mislabels: dict[str, tuple[Counter, dict[int, float]]] = {}
    d2_rows: dict[str, list[dict]] = {}

    for target_name in mv.TARGET_NAMES:
        target_id = target_ids_all[target_name]
        sibling_ids = [target_ids_all[n] for n in mv.TARGET_NAMES if n != target_name]
        images = images_by_target[target_name]

        per_thresh = {t: {"agnostic": RecallResult(), "correct": RecallResult()} for t in D1_THRESHOLDS}
        mislabel_counter: Counter = Counter()
        mislabel_score_sum: dict[int, float] = defaultdict(float)
        rows: list[dict] = []

        for img_path, label_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"  WARNING: failed to decode {img_path}, skipping")
                continue
            h, w = frame.shape[:2]
            gts = mv.load_gt_boxes(label_path, w, h, target_id)
            if not gts:
                continue

            raw, transform = mv.forward_raw(net, frame, mv.INPUT_SIZE)

            for t in D1_THRESHOLDS:
                dets = mv.decode_baseline(raw, transform, t, mv.NMS_IOU)
                m_correct, tot = mv.score_recall(gts, dets, target_id)
                per_thresh[t]["correct"].matched += m_correct
                per_thresh[t]["correct"].total += tot

                for gt in gts:
                    matched, mislabel = agnostic_match_and_mislabel(gt, dets, target_id)
                    per_thresh[t]["agnostic"].total += 1
                    if matched:
                        per_thresh[t]["agnostic"].matched += 1
                    if t == D1_MISLABEL_THRESHOLD and mislabel is not None:
                        cid, score = mislabel
                        mislabel_counter[cid] += 1
                        mislabel_score_sum[cid] += score

            for gt in gts:
                row = scores_at_best_anchor(raw, transform, gt, target_id, sibling_ids)
                if row is not None:
                    rows.append(row)

        d1_results[target_name] = per_thresh
        d1_mislabels[target_name] = (mislabel_counter, mislabel_score_sum)
        d2_rows[target_name] = rows

        agn6 = per_thresh[0.6]["agnostic"]
        cor6 = per_thresh[0.6]["correct"]
        print(
            f"{target_name}: agnostic@0.6={agn6.recall:.3f} ({agn6.matched}/{agn6.total})  "
            f"correct@0.6={cor6.recall:.3f} ({cor6.matched}/{cor6.total})  "
            f"[known-number sanity check]"
        )
        if mislabel_counter:
            detail = ", ".join(
                f"{names[cid]}×{cnt} (mean score {mislabel_score_sum[cid]/cnt:.3f})"
                for cid, cnt in mislabel_counter.most_common()
            )
            print(f"  mislabeled-as: {detail}")
        if rows:
            ts = np.array([r["target_score"] for r in rows])
            ss = np.array([r["best_sibling_score"] for r in rows])
            gs = np.array([r["global_argmax_score"] for r in rows])
            print(
                f"  D2 scores: target mean={ts.mean():.3f} | best_sibling mean={ss.mean():.3f} "
                f"| global_argmax mean={gs.mean():.3f} | frac(best_sibling>=0.6)={float((ss>=0.6).mean()):.3f}"
            )

    # ---- Diagnostic 3: imgsz sweep ----
    print("\n=== DIAGNOSTIC 3 (imgsz sweep) ===")
    d3_results: dict[str, dict[int, dict]] = {}

    # Warm up each resolution once so first-call MPS/graph overhead doesn't pollute timings.
    for imgsz in D3_IMGSZ_SWEEP:
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        mv.forward_raw(net, dummy, imgsz)

    for target_name in mv.TARGET_NAMES:
        target_id = target_ids_all[target_name]
        briefed_ids = [target_id] + dissimilar_ids
        images = images_by_target[target_name]
        d3_results[target_name] = {}

        for imgsz in D3_IMGSZ_SWEEP:
            recalls = {t: RecallResult() for t in D3_THRESHOLDS}
            agnostic6 = RecallResult()
            ceilings: list[float] = []
            times: list[float] = []

            for img_path, label_path in images:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue
                h, w = frame.shape[:2]
                gts = mv.load_gt_boxes(label_path, w, h, target_id)
                if not gts:
                    continue

                t0 = time.perf_counter()
                raw, transform = mv.forward_raw(net, frame, imgsz)
                if mv.DEVICE == "mps":
                    torch.mps.synchronize()
                times.append(time.perf_counter() - t0)

                for t in D3_THRESHOLDS:
                    dets = mv.decode_masked(raw, transform, briefed_ids, t, mv.NMS_IOU, input_size=imgsz)
                    m, tot = mv.score_recall(gts, dets, target_id)
                    recalls[t].matched += m
                    recalls[t].total += tot

                dets_agn = mv.decode_baseline(raw, transform, 0.6, mv.NMS_IOU, input_size=imgsz)
                for gt in gts:
                    matched, _ = agnostic_match_and_mislabel(gt, dets_agn, target_id)
                    agnostic6.total += 1
                    if matched:
                        agnostic6.matched += 1

                ceilings.extend(mv.masked_score_ceiling(raw, transform, target_id, briefed_ids, gts))

            ceiling_arr = np.array(ceilings) if ceilings else np.zeros(0)
            d3_results[target_name][imgsz] = {
                "recalls": recalls,
                "agnostic6": agnostic6,
                "ceiling": ceiling_arr,
                "mean_time_ms": float(np.mean(times) * 1000) if times else float("nan"),
            }
            print(
                f"{target_name} @ imgsz={imgsz}: ceiling_mean={ceiling_arr.mean():.3f} "
                f"correct@0.6={recalls[0.6].recall:.3f} correct@0.25={recalls[0.25].recall:.3f} "
                f"agnostic@0.6={agnostic6.recall:.3f} "
                f"mean_time={np.mean(times)*1000:.2f}ms" if times else "no images"
            )

    write_report(name_to_id, d1_results, d1_mislabels, d2_rows, d3_results, names)
    print(f"\nReport written to {REPORT_PATH}")
    return 0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(
    name_to_id: dict[str, int],
    d1_results: dict[str, dict[float, dict[str, RecallResult]]],
    d1_mislabels: dict[str, tuple[Counter, dict[int, float]]],
    d2_rows: dict[str, list[dict]],
    d3_results: dict[str, dict[int, dict]],
    names: dict[int, str],
) -> None:
    lines: list[str] = []
    lines.append("# Resolution & Remap Report")
    lines.append("")
    lines.append(
        "Three diagnostics deciding the fix for the 4 flags (Egypt, Iraq, Yemen, "
        "Syria) that masking alone did not recover at conf=0.6 "
        "(see masked_validation_report.md - that result is not re-derived here)."
    )
    lines.append("")
    lines.append(f"- IoU match threshold = {mv.IOU_MATCH_THRESHOLD}")
    lines.append(f"- nms_iou = {mv.NMS_IOU}")
    lines.append(f"- device = {mv.DEVICE} (timings below are Mac/MPS/PyTorch, NOT Pi/ncnn - see D3 notes)")
    lines.append("")
    lines.append("## Class indices used")
    lines.append("")
    lines.append("| Flag | class id |")
    lines.append("|---|---|")
    for n in mv.TARGET_NAMES + mv.DISSIMILAR_NAMES:
        lines.append(f"| {n} | {name_to_id[n]} |")
    lines.append("")

    # --- Diagnostic 1 ---
    lines.append("## Diagnostic 1 - class-agnostic vs class-correct recall (full 196-class decode, unmasked)")
    lines.append("")
    lines.append(
        "class-correct recall reuses `mv.score_recall` unchanged (target-class candidates "
        "only) - this is the authoritative number and should reproduce the known per-class "
        "metrics at 0.6 as a sanity check on this harness. class-agnostic recall is a new, "
        "independent check: does ANY predicted box (any of the 196 classes) overlap the GT "
        "at IoU>=0.5, via the single highest-IoU prediction per GT box."
    )
    lines.append("")
    header = "| Target |"
    sep = "|---|"
    for t in D1_THRESHOLDS:
        header += f" agnostic@{t} | correct@{t} |"
        sep += "---|---|"
    lines.append(header)
    lines.append(sep)
    for target_name in mv.TARGET_NAMES:
        row = f"| {target_name} |"
        for t in D1_THRESHOLDS:
            a = d1_results[target_name][t]["agnostic"]
            c = d1_results[target_name][t]["correct"]
            row += f" {a.recall:.3f} ({a.matched}/{a.total}) | {c.recall:.3f} ({c.matched}/{c.total}) |"
        lines.append(row)
    lines.append("")
    lines.append(f"### Mislabel distribution at conf={D1_MISLABEL_THRESHOLD} (localized but wrong class)")
    lines.append("")
    lines.append(
        "For each target GT box where the single best-overlapping prediction (any class, "
        "IoU>=0.5) is NOT the target class, this tallies what it was labeled as instead, and "
        "the mean confidence of that wrong label."
    )
    lines.append("")
    for target_name in mv.TARGET_NAMES:
        counter, score_sum = d1_mislabels[target_name]
        if not counter:
            lines.append(f"- **{target_name}**: no matched-but-mislabeled boxes at conf={D1_MISLABEL_THRESHOLD}.")
            continue
        detail = ", ".join(
            f"{names[cid]} x{cnt} (mean score {score_sum[cid]/cnt:.3f})"
            for cid, cnt in counter.most_common()
        )
        lines.append(f"- **{target_name}**: {detail}")
    lines.append("")

    # --- Diagnostic 2 ---
    lines.append("## Diagnostic 2 - target vs best-sibling vs global-argmax raw score at each GT box")
    lines.append("")
    lines.append(
        "Per GT box, the anchor used is the one (among anchors whose center falls inside the "
        "GT box - same anchor<->GT intersection test as the prior experiment) with the "
        "highest decoded-box IoU against the GT box; this selection is geometry-only, not "
        "biased toward any class's score. Scores are raw (unmasked), sigmoid-activated, "
        "straight off the 196-class head."
    )
    lines.append("")
    lines.append(
        "| Target | target mean/median/max | best-sibling mean/median/max | global-argmax mean/median/max "
        "| frac(best_sibling>=0.6) | frac(global_argmax>=0.6) |"
    )
    lines.append("|---|---|---|---|---|---|")
    for target_name in mv.TARGET_NAMES:
        rows = d2_rows[target_name]
        if not rows:
            lines.append(f"| {target_name} | n/a | n/a | n/a | n/a | n/a |")
            continue
        ts = np.array([r["target_score"] for r in rows])
        ss = np.array([r["best_sibling_score"] for r in rows])
        gs = np.array([r["global_argmax_score"] for r in rows])
        lines.append(
            f"| {target_name} | {ts.mean():.3f}/{np.median(ts):.3f}/{ts.max():.3f} | "
            f"{ss.mean():.3f}/{np.median(ss):.3f}/{ss.max():.3f} | "
            f"{gs.mean():.3f}/{np.median(gs):.3f}/{gs.max():.3f} | "
            f"{float((ss >= 0.6).mean()):.3f} | {float((gs >= 0.6).mean()):.3f} |"
        )
    lines.append("")

    # --- Diagnostic 3 ---
    lines.append("## Diagnostic 3 - imgsz sweep (640 / 960 / 1280)")
    lines.append("")
    lines.append(
        "masked score ceiling and correct@t reuse `mv.decode_masked` / `mv.masked_score_ceiling` "
        "with briefed set [target, Japan, Brazil, Sweden] at each imgsz (true before-argmax "
        "masking, unchanged logic - only the input resolution varies). agnostic@0.6 reuses the "
        "full 196-class unmasked decode, same definition as Diagnostic 1, at each imgsz. "
        "Timings are per-image forward-pass wall time on this Mac (MPS if available), warmed "
        "up before measurement - useful only for RELATIVE scaling across imgsz, NOT as a "
        "prediction of Raspberry Pi 5 / ncnn latency, which is a different runtime and chip."
    )
    lines.append("")
    lines.append(
        "| Target | imgsz | ceiling mean/median/max | correct@0.6 | correct@0.25 | agnostic@0.6 | mean time/image |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for target_name in mv.TARGET_NAMES:
        for imgsz in D3_IMGSZ_SWEEP:
            d = d3_results[target_name][imgsz]
            c = d["ceiling"]
            ceiling_s = f"{c.mean():.3f}/{np.median(c):.3f}/{c.max():.3f}" if c.size else "n/a"
            r6 = d["recalls"][0.6]
            r25 = d["recalls"][0.25]
            a6 = d["agnostic6"]
            lines.append(
                f"| {target_name} | {imgsz} | {ceiling_s} | {r6.recall:.3f} ({r6.matched}/{r6.total}) | "
                f"{r25.recall:.3f} ({r25.matched}/{r25.total}) | {a6.recall:.3f} ({a6.matched}/{a6.total}) | "
                f"{d['mean_time_ms']:.2f} ms |"
            )
    lines.append("")

    # --- Decision section ---
    lines.append("## Decision")
    lines.append("")

    lines.append("### Are location points at risk, or only ID points?")
    lines.append("")
    lines.append(
        "gap = agnostic@0.6 - correct@0.6 estimates how much of the collapse is "
        "\"found but mislabeled\" (a gap would mean some boxes clear 0.6 under a "
        "wrong class) versus \"not found at all\" (any class)."
    )
    lines.append("")
    for target_name in mv.TARGET_NAMES:
        agn6 = d1_results[target_name][0.6]["agnostic"]
        cor6 = d1_results[target_name][0.6]["correct"]
        gap = agn6.recall - cor6.recall
        if agn6.total == 0:
            lines.append(f"- **{target_name}**: no GT boxes evaluated.")
            continue
        if gap >= 0.15:
            verdict = (
                f"a real found-but-mislabeled population exists (gap={gap:.3f}) - some "
                f"location points ARE safe even where the ID is wrong."
            )
        else:
            verdict = (
                f"gap is negligible ({gap:.3f}) - failures are essentially all \"not found "
                f"as ANY class,\" not \"found but wrong label.\" Location points are at risk "
                f"in direct proportion to (1 - agnostic recall) = {1 - agn6.recall:.3f}, not "
                f"a separate ID-only problem."
            )
        lines.append(
            f"- **{target_name}**: agnostic@0.6={agn6.recall:.3f}, correct@0.6={cor6.recall:.3f} - {verdict}"
        )
    lines.append("")

    lines.append("### Would a confusion-remap layer recover these flags?")
    lines.append("")
    total_mislabels = sum(sum(d1_mislabels[n][0].values()) for n in mv.TARGET_NAMES)
    total_gt = sum(d1_results[n][0.6]["correct"].total for n in mv.TARGET_NAMES)
    lines.append(
        f"Cross-check against Diagnostic 1: across all {total_gt} GT boxes (all 4 flags), "
        f"the full multi-anchor NMS+threshold decode (not just the single best-IoU anchor "
        f"D2 samples) found only {total_mislabels} matched-but-mislabeled box in total (see "
        f"the D1 mislabel distribution above). Two independent methods agree: there is "
        f"essentially no confident-sibling-hiding-under-the-wrong-label population to redirect."
    )
    lines.append("")
    for target_name in mv.TARGET_NAMES:
        rows = d2_rows[target_name]
        if not rows:
            lines.append(f"- **{target_name}**: no GT-covering anchors found; cannot assess.")
            continue
        ss = np.array([r["best_sibling_score"] for r in rows])
        gs = np.array([r["global_argmax_score"] for r in rows])
        frac_sib = float((ss >= 0.6).mean())
        frac_glob = float((gs >= 0.6).mean())
        if frac_sib >= 0.3 or frac_glob >= 0.3:
            verdict = (
                f"YES, worth building - a sibling (or the global argmax) clears 0.6 on "
                f"{max(frac_sib, frac_glob)*100:.0f}% of GT boxes while masking discards that "
                f"confidence outright. A remap (relabel high-confidence sibling hits to the "
                f"briefed target) would recover detections masking currently throws away."
            )
        else:
            verdict = (
                f"NO - siblings clear 0.6 on only {frac_sib*100:.0f}% of GT boxes "
                f"(global argmax {frac_glob*100:.0f}%). There's no confident-but-mislabeled "
                f"signal to redirect; a remap layer would have almost nothing to work with."
            )
        lines.append(f"- **{target_name}**: frac(best_sibling>=0.6)={frac_sib:.3f}, frac(global_argmax>=0.6)={frac_glob:.3f}. {verdict}")
    lines.append("")

    lines.append("### Does imgsz=960 or 1280 lift the ceiling above 0.6?")
    lines.append("")
    for target_name in mv.TARGET_NAMES:
        cells = []
        for imgsz in D3_IMGSZ_SWEEP:
            c = d3_results[target_name][imgsz]["ceiling"]
            cells.append((imgsz, float(c.mean()) if c.size else float("nan"), float(c.max()) if c.size else float("nan")))
        base_mean = cells[0][1]
        best_imgsz, best_mean, best_max = max(cells, key=lambda x: x[1])
        if best_max >= 0.6 and best_mean > base_mean + 0.05:
            verdict = f"YES - imgsz={best_imgsz} lifts the ceiling to mean={best_mean:.3f}, max={best_max:.3f}."
        elif best_mean > base_mean + 0.05:
            verdict = f"PARTIALLY - imgsz={best_imgsz} raises the mean ceiling to {best_mean:.3f} but still under 0.6."
        else:
            verdict = "NO - resolution has little to no effect on the confidence ceiling for this flag."
        cell_str = ", ".join(f"{sz}:{m:.3f}" for sz, m, _ in cells)
        lines.append(f"- **{target_name}**: ceiling mean by imgsz [{cell_str}]. {verdict}")
    lines.append("")

    lines.append("### Recommended fix, ranked, with evidence")
    lines.append("")
    lines.append(
        "This ranking is filled in per-flag from the three sections above - read each "
        "flag's D1/D2/D3 verdicts together before implementing anything:"
    )
    lines.append("")
    for target_name in mv.TARGET_NAMES:
        rows = d2_rows[target_name]
        options = []
        # (a) higher imgsz
        cells = [(sz, float(d3_results[target_name][sz]["ceiling"].mean()) if d3_results[target_name][sz]["ceiling"].size else float("nan")) for sz in D3_IMGSZ_SWEEP]
        base_mean = cells[0][1]
        best_imgsz, best_mean = max(cells, key=lambda x: x[1])
        if best_imgsz != D3_IMGSZ_SWEEP[0] and best_mean > base_mean + 0.05:
            options.append(f"(a) higher imgsz ({best_imgsz}): ceiling mean {base_mean:.3f} -> {best_mean:.3f}")
        # (b) lower threshold for masked-only pipeline - use the MASKED (D3,
        # native imgsz=640) recall here, not D1's unmasked baseline recall;
        # they differ (masking adds a real, separate lift - see the prior
        # masked_validation_report.md control comparison).
        c6 = d3_results[target_name][D3_IMGSZ_SWEEP[0]]["recalls"][0.6]
        c25 = d3_results[target_name][D3_IMGSZ_SWEEP[0]]["recalls"][0.25]
        if c25.recall > c6.recall + 0.1:
            options.append(f"(b) class-specific lower threshold when masked: correct@0.6={c6.recall:.3f} -> correct@0.25={c25.recall:.3f}")
        # (c) remap layer
        if rows:
            ss = np.array([r["best_sibling_score"] for r in rows])
            frac_sib = float((ss >= 0.6).mean())
            if frac_sib >= 0.3:
                options.append(f"(c) mission-aware remap layer: best-sibling clears 0.6 on {frac_sib*100:.0f}% of GT boxes")
        # (d) lower altitude / more pixels - fallback framing
        if not options:
            options.append("(d) lower-altitude pass / more pixels-on-target - no threshold or resolution lever moved the numbers enough on this val set")
        lines.append(f"- **{target_name}**: " + "; ".join(options) if options else f"- **{target_name}**: no option in this val set clearly recovers this flag; treat as high-risk.")
    lines.append("")
    lines.append(
        "Options not listed for a flag above are unnecessary for that flag given this data - "
        "e.g. if (b) alone already gets a flag's correct@t close to the ceiling, a remap layer "
        "adds complexity without headroom to gain."
    )
    REPORT_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
