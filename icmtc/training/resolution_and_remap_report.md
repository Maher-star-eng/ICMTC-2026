# Resolution & Remap Report

Three diagnostics deciding the fix for the 4 flags (Egypt, Iraq, Yemen, Syria) that masking alone did not recover at conf=0.6 (see masked_validation_report.md - that result is not re-derived here).

- IoU match threshold = 0.5
- nms_iou = 0.45
- device = mps (timings below are Mac/MPS/PyTorch, NOT Pi/ncnn - see D3 notes)

## Class indices used

| Flag | class id |
|---|---|
| Egypt | 50 |
| Iraq | 78 |
| Yemen | 193 |
| Syria | 170 |
| Japan | 83 |
| Brazil | 23 |
| Sweden | 168 |

## Diagnostic 1 - class-agnostic vs class-correct recall (full 196-class decode, unmasked)

class-correct recall reuses `mv.score_recall` unchanged (target-class candidates only) - this is the authoritative number and should reproduce the known per-class metrics at 0.6 as a sanity check on this harness. class-agnostic recall is a new, independent check: does ANY predicted box (any of the 196 classes) overlap the GT at IoU>=0.5, via the single highest-IoU prediction per GT box.

| Target | agnostic@0.6 | correct@0.6 | agnostic@0.4 | correct@0.4 | agnostic@0.25 | correct@0.25 |
|---|---|---|---|---|---|---|
| Egypt | 0.000 (0/30) | 0.000 (0/30) | 0.267 (8/30) | 0.067 (2/30) | 0.967 (29/30) | 0.233 (7/30) |
| Iraq | 0.032 (1/31) | 0.000 (0/31) | 0.323 (10/31) | 0.032 (1/31) | 1.000 (31/31) | 0.677 (21/31) |
| Yemen | 0.071 (2/28) | 0.071 (2/28) | 0.250 (7/28) | 0.143 (4/28) | 0.893 (25/28) | 0.429 (12/28) |
| Syria | 0.323 (10/31) | 0.323 (10/31) | 0.677 (21/31) | 0.581 (18/31) | 0.935 (29/31) | 0.710 (22/31) |

### Mislabel distribution at conf=0.6 (localized but wrong class)

For each target GT box where the single best-overlapping prediction (any class, IoU>=0.5) is NOT the target class, this tallies what it was labeled as instead, and the mean confidence of that wrong label.

- **Egypt**: no matched-but-mislabeled boxes at conf=0.6.
- **Iraq**: Syria x1 (mean score 0.608)
- **Yemen**: no matched-but-mislabeled boxes at conf=0.6.
- **Syria**: no matched-but-mislabeled boxes at conf=0.6.

## Diagnostic 2 - target vs best-sibling vs global-argmax raw score at each GT box

Per GT box, the anchor used is the one (among anchors whose center falls inside the GT box - same anchor<->GT intersection test as the prior experiment) with the highest decoded-box IoU against the GT box; this selection is geometry-only, not biased toward any class's score. Scores are raw (unmasked), sigmoid-activated, straight off the 196-class head.

| Target | target mean/median/max | best-sibling mean/median/max | global-argmax mean/median/max | frac(best_sibling>=0.6) | frac(global_argmax>=0.6) |
|---|---|---|---|---|---|
| Egypt | 0.067/0.000/0.256 | 0.111/0.000/0.449 | 0.111/0.000/0.449 | 0.000 | 0.000 |
| Iraq | 0.052/0.000/0.287 | 0.078/0.000/0.410 | 0.081/0.000/0.410 | 0.000 | 0.000 |
| Yemen | 0.096/0.000/0.668 | 0.087/0.000/0.318 | 0.126/0.000/0.668 | 0.000 | 0.036 |
| Syria | 0.204/0.092/0.844 | 0.090/0.045/0.355 | 0.223/0.219/0.844 | 0.000 | 0.097 |

## Diagnostic 3 - imgsz sweep (640 / 960 / 1280)

masked score ceiling and correct@t reuse `mv.decode_masked` / `mv.masked_score_ceiling` with briefed set [target, Japan, Brazil, Sweden] at each imgsz (true before-argmax masking, unchanged logic - only the input resolution varies). agnostic@0.6 reuses the full 196-class unmasked decode, same definition as Diagnostic 1, at each imgsz. Timings are per-image forward-pass wall time on this Mac (MPS if available), warmed up before measurement - useful only for RELATIVE scaling across imgsz, NOT as a prediction of Raspberry Pi 5 / ncnn latency, which is a different runtime and chip.

| Target | imgsz | ceiling mean/median/max | correct@0.6 | correct@0.25 | agnostic@0.6 | mean time/image |
|---|---|---|---|---|---|---|
| Egypt | 640 | 0.219/0.190/0.490 | 0.000 (0/30) | 0.300 (9/30) | 0.000 (0/30) | 13.39 ms |
| Egypt | 960 | 0.252/0.241/0.508 | 0.000 (0/30) | 0.433 (13/30) | 0.033 (1/30) | 20.51 ms |
| Egypt | 1280 | 0.216/0.203/0.392 | 0.000 (0/30) | 0.300 (9/30) | 0.067 (2/30) | 32.03 ms |
| Iraq | 640 | 0.298/0.301/0.446 | 0.000 (0/31) | 0.774 (24/31) | 0.032 (1/31) | 14.59 ms |
| Iraq | 960 | 0.289/0.301/0.435 | 0.000 (0/31) | 0.710 (22/31) | 0.032 (1/31) | 21.22 ms |
| Iraq | 1280 | 0.293/0.305/0.521 | 0.000 (0/31) | 0.677 (21/31) | 0.097 (3/31) | 32.91 ms |
| Yemen | 640 | 0.255/0.219/0.768 | 0.071 (2/28) | 0.464 (13/28) | 0.071 (2/28) | 14.48 ms |
| Yemen | 960 | 0.343/0.340/0.856 | 0.107 (3/28) | 0.607 (17/28) | 0.107 (3/28) | 19.94 ms |
| Yemen | 1280 | 0.432/0.439/0.709 | 0.250 (7/28) | 0.786 (22/28) | 0.250 (7/28) | 31.35 ms |
| Syria | 640 | 0.460/0.474/0.865 | 0.323 (10/31) | 0.742 (23/31) | 0.323 (10/31) | 13.44 ms |
| Syria | 960 | 0.548/0.611/0.889 | 0.516 (16/31) | 0.903 (28/31) | 0.516 (16/31) | 19.52 ms |
| Syria | 1280 | 0.558/0.613/0.908 | 0.516 (16/31) | 0.839 (26/31) | 0.516 (16/31) | 31.69 ms |

## Decision

### Are location points at risk, or only ID points?

gap = agnostic@0.6 - correct@0.6 estimates how much of the collapse is "found but mislabeled" (a gap would mean some boxes clear 0.6 under a wrong class) versus "not found at all" (any class).

- **Egypt**: agnostic@0.6=0.000, correct@0.6=0.000 - gap is negligible (0.000) - failures are essentially all "not found as ANY class," not "found but wrong label." Location points are at risk in direct proportion to (1 - agnostic recall) = 1.000, not a separate ID-only problem.
- **Iraq**: agnostic@0.6=0.032, correct@0.6=0.000 - gap is negligible (0.032) - failures are essentially all "not found as ANY class," not "found but wrong label." Location points are at risk in direct proportion to (1 - agnostic recall) = 0.968, not a separate ID-only problem.
- **Yemen**: agnostic@0.6=0.071, correct@0.6=0.071 - gap is negligible (0.000) - failures are essentially all "not found as ANY class," not "found but wrong label." Location points are at risk in direct proportion to (1 - agnostic recall) = 0.929, not a separate ID-only problem.
- **Syria**: agnostic@0.6=0.323, correct@0.6=0.323 - gap is negligible (0.000) - failures are essentially all "not found as ANY class," not "found but wrong label." Location points are at risk in direct proportion to (1 - agnostic recall) = 0.677, not a separate ID-only problem.

### Would a confusion-remap layer recover these flags?

Cross-check against Diagnostic 1: across all 120 GT boxes (all 4 flags), the full multi-anchor NMS+threshold decode (not just the single best-IoU anchor D2 samples) found only 1 matched-but-mislabeled box in total (see the D1 mislabel distribution above). Two independent methods agree: there is essentially no confident-sibling-hiding-under-the-wrong-label population to redirect.

- **Egypt**: frac(best_sibling>=0.6)=0.000, frac(global_argmax>=0.6)=0.000. NO - siblings clear 0.6 on only 0% of GT boxes (global argmax 0%). There's no confident-but-mislabeled signal to redirect; a remap layer would have almost nothing to work with.
- **Iraq**: frac(best_sibling>=0.6)=0.000, frac(global_argmax>=0.6)=0.000. NO - siblings clear 0.6 on only 0% of GT boxes (global argmax 0%). There's no confident-but-mislabeled signal to redirect; a remap layer would have almost nothing to work with.
- **Yemen**: frac(best_sibling>=0.6)=0.000, frac(global_argmax>=0.6)=0.036. NO - siblings clear 0.6 on only 0% of GT boxes (global argmax 4%). There's no confident-but-mislabeled signal to redirect; a remap layer would have almost nothing to work with.
- **Syria**: frac(best_sibling>=0.6)=0.000, frac(global_argmax>=0.6)=0.097. NO - siblings clear 0.6 on only 0% of GT boxes (global argmax 10%). There's no confident-but-mislabeled signal to redirect; a remap layer would have almost nothing to work with.

### Does imgsz=960 or 1280 lift the ceiling above 0.6?

- **Egypt**: ceiling mean by imgsz [640:0.219, 960:0.252, 1280:0.216]. NO - resolution has little to no effect on the confidence ceiling for this flag.
- **Iraq**: ceiling mean by imgsz [640:0.298, 960:0.289, 1280:0.293]. NO - resolution has little to no effect on the confidence ceiling for this flag.
- **Yemen**: ceiling mean by imgsz [640:0.255, 960:0.343, 1280:0.432]. YES - imgsz=1280 lifts the ceiling to mean=0.432, max=0.709.
- **Syria**: ceiling mean by imgsz [640:0.460, 960:0.548, 1280:0.558]. YES - imgsz=1280 lifts the ceiling to mean=0.558, max=0.908.

### Recommended fix, ranked, with evidence

This ranking is filled in per-flag from the three sections above - read each flag's D1/D2/D3 verdicts together before implementing anything:

- **Egypt**: (b) class-specific lower threshold when masked: correct@0.6=0.000 -> correct@0.25=0.300
- **Iraq**: (b) class-specific lower threshold when masked: correct@0.6=0.000 -> correct@0.25=0.774
- **Yemen**: (a) higher imgsz (1280): ceiling mean 0.255 -> 0.432; (b) class-specific lower threshold when masked: correct@0.6=0.071 -> correct@0.25=0.464
- **Syria**: (a) higher imgsz (1280): ceiling mean 0.460 -> 0.558; (b) class-specific lower threshold when masked: correct@0.6=0.323 -> correct@0.25=0.742

Options not listed for a flag above are unnecessary for that flag given this data - e.g. if (b) alone already gets a flag's correct@t close to the ceiling, a remap layer adds complexity without headroom to gain.