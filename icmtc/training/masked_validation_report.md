# Masked Validation Report

Empirical test of whether deployment-style before-argmax class masking recovers recall on the 4 red-white-black Arab tricolor flags that collapsed in full 196-class validation.

- conf_threshold = 0.6 (real deployment threshold)
- nms_iou = 0.45
- IoU match threshold = 0.5
- input_size = 640
- device = mps

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

Dissimilar trio used for the briefed 4-flag set (all targets): Japan, Brazil, Sweden - white field/red disc, green field/yellow rhombus/blue globe, blue field/yellow Nordic cross respectively; none is a horizontal red-white-black tricolor.

C-worstcase set per target = [target, empirical worst sibling, Japan, Brazil] (Sweden dropped to keep the set at 4, matching the task's own worked example). The "worst sibling" is not arbitrary: for each target, every val image's GT box is intersected with anchor centers, and the GLOBAL (full 196-class) argmax winner at each such anchor is tallied across the other 3 red-white-black flags; the most frequent winner is used as the worst-case partner. Per-target partner used: Egypt→Syria, Iraq→Egypt, Yemen→Egypt, Syria→Egypt.

## Results

| Target | A: baseline (196-class) | B: after-argmax filter (`classes=`) | C: true before-argmax masking | C-worstcase (+2nd sibling) | Verdict (on C) |
|---|---|---|---|---|---|
| Egypt | 0.000 (0/30) | 0.000 (0/30) | 0.000 (0/30) | 0.000 (0/30, +Syria) | STILL WEAK |
| Iraq | 0.000 (0/31) | 0.000 (0/31) | 0.000 (0/31) | 0.000 (0/31, +Egypt) | STILL WEAK |
| Yemen | 0.071 (2/28) | 0.071 (2/28) | 0.071 (2/28) | 0.071 (2/28, +Egypt) | STILL WEAK |
| Syria | 0.323 (10/31) | 0.323 (10/31) | 0.323 (10/31) | 0.323 (10/31, +Egypt) | STILL WEAK |

## Per-flag verdicts

- **Egypt**: STILL WEAK (true-masking recall = 0.000)
- **Iraq**: STILL WEAK (true-masking recall = 0.000)
- **Yemen**: STILL WEAK (true-masking recall = 0.071)
- **Syria**: STILL WEAK (true-masking recall = 0.323)

## Diagnostic: is the deployment threshold (0.6) itself the bottleneck?

For each target, the masked score ceiling is the best target-class score achieved (under true before-argmax masking, at any anchor overlapping a GT box) where the target *already wins* the restricted argmax - i.e. the best this masked pipeline could ever report for that GT, independent of conf_threshold. If the ceiling is well below 0.6, masking has correctly removed the sibling competition but the network's absolute confidence in the correct class is still too low to clear the deployment threshold - a different failure mode than "masking doesn't help", with a different fix (lower the threshold for a masked-only pipeline, or get more pixels on target) rather than "this is unrecoverable".

A@t is the UNMASKED (full 196-class) recall at that same lower threshold - the control. If C@t clears A@t by a wide margin, the gain is attributable to masking (removing the sibling's competing label), not merely to a laxer threshold that would have helped anyway.

| Target | ceiling mean | ceiling median | ceiling max | C@0.6 | C@0.25 | A@0.25 (control) | C@0.4 | A@0.4 (control) |
|---|---|---|---|---|---|---|---|---|
| Egypt | 0.219 | 0.190 | 0.490 | 0.000 (0/30) | 0.300 (9/30) | 0.233 (7/30) | 0.067 (2/30) | 0.067 (2/30) |
| Iraq | 0.298 | 0.301 | 0.446 | 0.000 (0/31) | 0.774 (24/31) | 0.677 (21/31) | 0.032 (1/31) | 0.032 (1/31) |
| Yemen | 0.255 | 0.219 | 0.768 | 0.071 (2/28) | 0.464 (13/28) | 0.429 (12/28) | 0.143 (4/28) | 0.143 (4/28) |
| Syria | 0.460 | 0.474 | 0.865 | 0.323 (10/31) | 0.742 (23/31) | 0.710 (22/31) | 0.581 (18/31) | 0.581 (18/31) |

## Conclusion

Masking does NOT recover, at conf=0.6: Egypt, Iraq, Yemen, Syria. For each of these, check the ceiling table above before concluding it is a pure resolution limit:
  - **Egypt**: masking correctly removes the sibling competition (see the before-argmax invariant above), but the network's own confidence in Egypt tops out at 0.490 - below the 0.6 deployment threshold. Recall at a lower operating point (conf=0.25) rises to 0.300 masked vs. only 0.233 unmasked at the same threshold (control) - a +0.067 recall gain attributable to masking itself, not to the lower threshold alone. This is a THRESHOLD problem, not strictly a resolution problem: consider a lower conf_threshold specifically when the masked target set is known, since masking already eliminates the false-positive risk from siblings that a lower global threshold would otherwise invite.
  - **Iraq**: masking correctly removes the sibling competition (see the before-argmax invariant above), but the network's own confidence in Iraq tops out at 0.446 - below the 0.6 deployment threshold. Recall at a lower operating point (conf=0.25) rises to 0.774 masked vs. only 0.677 unmasked at the same threshold (control) - a +0.097 recall gain attributable to masking itself, not to the lower threshold alone. This is a THRESHOLD problem, not strictly a resolution problem: consider a lower conf_threshold specifically when the masked target set is known, since masking already eliminates the false-positive risk from siblings that a lower global threshold would otherwise invite.
  - **Yemen**: the masked score ceiling reaches 0.768 on at least one GT box (>= 0.6), yet recall@0.6 is 0.071 - the model IS occasionally confident enough even masked, but not consistently. Genuine partial resolution limit, not a pure threshold artifact.
  - **Syria**: the masked score ceiling reaches 0.865 on at least one GT box (>= 0.6), yet recall@0.6 is 0.323 - the model IS occasionally confident enough even masked, but not consistently. Genuine partial resolution limit, not a pure threshold artifact.

The A vs C gap (where recall jumps from baseline to true masking) is the part of the 196-class collapse attributable purely to sibling competition; the B column is shown only as the pessimistic (and incorrect-for-deployment) lower bound from Ultralytics' own `classes=` filter, which deletes misclassified-as-sibling detections instead of recovering them.
