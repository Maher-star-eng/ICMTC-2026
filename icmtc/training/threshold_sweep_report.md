# Threshold Sweep Report

Sweeps the masked-pipeline confidence threshold to find the operating point that recovers the 4 weak flags (Egypt, Iraq, Yemen, Syria) without damaging the 192 healthy classes or inviting false positives on background images. Builds on masked_validation_report.md and resolution_and_remap_report.md (not re-derived here).

- thresholds swept = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
- imgsz = 640, nms_iou = 0.45, IoU match = 0.5, device = mps
- decode = TRUE before-argmax masking (`export.masked_argmax_predictions`), NOT Ultralytics `classes=`

## Part 1 - the 4 weak flags (masked)

Briefed set per flag = `[target, Japan, Brazil, Sweden]`, same dissimilar trio as the prior reports.

### Egypt (class id 50)

| threshold | recall | precision | F1 | FP/image |
|---|---|---|---|---|
| 0.25 | 0.300 (9/30) | 1.000 | 0.462 | 0.000 |
| 0.3 | 0.167 (5/30) | 1.000 | 0.286 | 0.000 |
| 0.35 | 0.100 (3/30) | 1.000 | 0.182 | 0.000 |
| 0.4 | 0.067 (2/30) | 1.000 | 0.125 | 0.000 |
| 0.45 | 0.033 (1/30) | 1.000 | 0.065 | 0.000 |
| 0.5 | 0.000 (0/30) | nan | 0.000 | 0.000 |
| 0.55 | 0.000 (0/30) | nan | 0.000 | 0.000 |
| 0.6 | 0.000 (0/30) | nan | 0.000 | 0.000 |

### Iraq (class id 78)

| threshold | recall | precision | F1 | FP/image |
|---|---|---|---|---|
| 0.25 | 0.774 (24/31) | 1.000 | 0.873 | 0.000 |
| 0.3 | 0.548 (17/31) | 1.000 | 0.708 | 0.000 |
| 0.35 | 0.226 (7/31) | 1.000 | 0.368 | 0.000 |
| 0.4 | 0.032 (1/31) | 1.000 | 0.062 | 0.000 |
| 0.45 | 0.000 (0/31) | nan | 0.000 | 0.000 |
| 0.5 | 0.000 (0/31) | nan | 0.000 | 0.000 |
| 0.55 | 0.000 (0/31) | nan | 0.000 | 0.000 |
| 0.6 | 0.000 (0/31) | nan | 0.000 | 0.000 |

### Yemen (class id 193)

| threshold | recall | precision | F1 | FP/image |
|---|---|---|---|---|
| 0.25 | 0.464 (13/28) | 1.000 | 0.634 | 0.000 |
| 0.3 | 0.357 (10/28) | 1.000 | 0.526 | 0.000 |
| 0.35 | 0.214 (6/28) | 1.000 | 0.353 | 0.000 |
| 0.4 | 0.143 (4/28) | 1.000 | 0.250 | 0.000 |
| 0.45 | 0.107 (3/28) | 1.000 | 0.194 | 0.000 |
| 0.5 | 0.107 (3/28) | 1.000 | 0.194 | 0.000 |
| 0.55 | 0.107 (3/28) | 1.000 | 0.194 | 0.000 |
| 0.6 | 0.071 (2/28) | 1.000 | 0.133 | 0.000 |

### Syria (class id 170)

| threshold | recall | precision | F1 | FP/image |
|---|---|---|---|---|
| 0.25 | 0.742 (23/31) | 1.000 | 0.852 | 0.000 |
| 0.3 | 0.677 (21/31) | 1.000 | 0.808 | 0.000 |
| 0.35 | 0.645 (20/31) | 1.000 | 0.784 | 0.000 |
| 0.4 | 0.581 (18/31) | 1.000 | 0.735 | 0.000 |
| 0.45 | 0.516 (16/31) | 1.000 | 0.681 | 0.000 |
| 0.5 | 0.484 (15/31) | 1.000 | 0.652 | 0.000 |
| 0.55 | 0.355 (11/31) | 1.000 | 0.524 | 0.000 |
| 0.6 | 0.323 (10/31) | 1.000 | 0.488 | 0.000 |

## Part 2 - 20 healthy flags (regression guard)

Briefed set per flag = `[target] + 3 dissimilar decoys` (Japan/Brazil/Sweden, or Kenya/Chile/Nepal for those 3 flags themselves, so a flag never decoys itself).

| threshold | mean recall | mean precision | mean F1 | worst-case F1 (min) | mean FP/image |
|---|---|---|---|---|---|
| 0.25 | 1.000 | 1.000 | 1.000 | 1.000 (Japan) | 0.000 |
| 0.3 | 1.000 | 1.000 | 1.000 | 1.000 (Japan) | 0.000 |
| 0.35 | 0.998 | 1.000 | 0.999 | 0.980 (Poland) | 0.000 |
| 0.4 | 0.998 | 1.000 | 0.999 | 0.980 (Poland) | 0.000 |
| 0.45 | 0.998 | 1.000 | 0.999 | 0.980 (Poland) | 0.000 |
| 0.5 | 0.995 | 1.000 | 0.997 | 0.964 (Indonesia) | 0.000 |
| 0.55 | 0.995 | 1.000 | 0.997 | 0.964 (Indonesia) | 0.000 |
| 0.6 | 0.991 | 1.000 | 0.995 | 0.936 (Poland) | 0.000 |

### Per-flag detail (Part 2)

| Flag | F1@0.6 | F1@0.25 | FP/img@0.6 | FP/img@0.25 |
|---|---|---|---|---|
| Japan | 1.000 | 1.000 | 0.000 | 0.000 |
| Brazil | 1.000 | 1.000 | 0.000 | 0.000 |
| Sweden | 1.000 | 1.000 | 0.000 | 0.000 |
| France | 1.000 | 1.000 | 0.000 | 0.000 |
| Germany | 1.000 | 1.000 | 0.000 | 0.000 |
| Kenya | 1.000 | 1.000 | 0.000 | 0.000 |
| Vietnam | 1.000 | 1.000 | 0.000 | 0.000 |
| Chile | 1.000 | 1.000 | 0.000 | 0.000 |
| Nepal | 1.000 | 1.000 | 0.000 | 0.000 |
| Canada | 1.000 | 1.000 | 0.000 | 0.000 |
| Mexico | 1.000 | 1.000 | 0.000 | 0.000 |
| Nigeria | 1.000 | 1.000 | 0.000 | 0.000 |
| Poland | 0.936 | 1.000 | 0.000 | 0.000 |
| Thailand | 1.000 | 1.000 | 0.000 | 0.000 |
| Turkey | 1.000 | 1.000 | 0.000 | 0.000 |
| Argentina | 1.000 | 1.000 | 0.000 | 0.000 |
| Norway | 1.000 | 1.000 | 0.000 | 0.000 |
| Ghana | 1.000 | 1.000 | 0.000 | 0.000 |
| Indonesia | 0.964 | 1.000 | 0.000 | 0.000 |
| Peru | 1.000 | 1.000 | 0.000 | 0.000 |

## Part 3 - false positives on background images

444 background (no-label, no-flag) val images, briefed set = `[Egypt, Japan, Brazil, Sweden]` (representative). Every detection here is a false positive by construction - these images have zero ground-truth objects.

| threshold | total false detections | FP per background image |
|---|---|---|
| 0.25 | 0 | 0.0000 |
| 0.3 | 0 | 0.0000 |
| 0.35 | 0 | 0.0000 |
| 0.4 | 0 | 0.0000 |
| 0.45 | 0 | 0.0000 |
| 0.5 | 0 | 0.0000 |
| 0.55 | 0 | 0.0000 |
| 0.6 | 0 | 0.0000 |

## Knee analysis

Recall knee = highest (most conservative) threshold that still retains >= 90% of the max recall seen anywhere in the sweep for that flag - i.e. where further lowering the threshold stops paying off.

| Flag | knee threshold | recall at knee | max recall in sweep (at 0.25) |
|---|---|---|---|
| Egypt | 0.25 (never plateaus in this sweep range) | 0.300 | 0.300 |
| Iraq | 0.25 (never plateaus in this sweep range) | 0.774 | 0.774 |
| Yemen | 0.25 (never plateaus in this sweep range) | 0.464 | 0.464 |
| Syria | 0.3 | 0.677 | 0.742 |

Background FP/image is flat at 0 across the entire swept range (0.25-0.60) - there is no climb to identify; lowering the threshold this far did not invite any false detections on this val set's 444 background images.

## Recommendation

### Single global threshold?

Decision rule (stated, not black-box): pick the LOWEST threshold where worst-case healthy-class F1 has not dropped more than 0.02 from its conf=0.6 value (0.936), AND background FP/image has not risen more than 0.5 from its conf=0.6 value (0.0000).

**Recommended global threshold: 0.25.** At this threshold: mean weak-flag recall = 0.570, worst-case healthy F1 = 1.000, background FP/image = 0.0000.

### Is a class-specific threshold warranted?

At the weak flags' best knee threshold (0.25): worst-case healthy F1 = 1.000 (cost vs 0.6: -0.064), background FP/image = 0.0000 (cost vs 0.6: +0.0000).

**NO - a single global threshold of 0.25 is sufficient.** The cost to healthy classes and background FP rate at this threshold is within the stated budget; a class-specific rule would add complexity without a measurable benefit.

### Egypt's residual risk

**No threshold in this sweep gets Egypt's class-correct recall above 0.5.** Best in sweep: conf=0.25, recall=0.300. Despite 0.967 class-agnostic recall @0.25 (resolution_and_remap_report.md), Egypt's OWN raw confidence at its own GT location is too low, too often, to cross even a permissive threshold with correct labeling - this is a genuine residual risk. If Egypt is the briefed target, do not expect reliable ID-scoring from threshold-lowering alone; combine with the resolution/altitude mitigation from resolution_and_remap_report.md, or budget for location-only points on this flag.

### Tracker context

`tracker.py`'s `Tracker` requires `hits >= min_hits` (default `min_hits=3`) before a track is confirmed - a detection has to repeat across consecutive-ish frames before it commits. The FP/image numbers above are single-frame detector output and are an UPPER BOUND on real-world impact: transient single-frame false positives (the kind a lower threshold invites) are exactly what `min_hits` is designed to suppress. This does not eliminate the false-positive cost - a persistent false signal (e.g. a real flag-colored object staying in frame) would still confirm - but it means the raw FP/image rate above overstates the operational false-track rate.