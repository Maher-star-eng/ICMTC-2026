<!--
Author: Maher Ahmed Maher Shola
Email:  maher.shola.officer@gmail.com
        M.shoala61308@student.aast.edu
Date:   07/07/2026
-->

# Flag Detection Dataset

Training data for the ICMTC 2026 UAVC-9 Mission 2 flag detector: 196 national
flags, captured from an aerial/top-down perspective for drone-based
detection, identification, and GPS geolocation.

- **Roboflow project:** [flag-detection-0swrg, v1](https://app.roboflow.com/maher-ahmed/flag-detection-0swrg/1)
  (workspace `maher-ahmed`, license: Private)
- **Local copy:** `training/flag detection.v1i.yolov11/` (YOLOv11/Ultralytics
  export format)

## Numbers (from the dataset actually in this repo)

Counted directly from `training/flag detection.v1i.yolov11/{split}/images`
and `{split}/labels` — an image with no matching `.txt` label file is a
background/negative image (no flag present, used to teach the model what
"no target" looks like).

| Split | Images | Annotated (flag) | Background/negative |
|-------|-------:|------------------:|---------------------:|
| train | 24,138 | 22,680 | 1,458 |
| valid |  2,290 |  1,846 |   444 |
| test  |  1,384 |  1,160 |   224 |
| **Total** | **27,812** | **25,686** | **2,126** |

- **Classes:** 196, one per country flag (see `data.yaml` → `names`). Full
  list is every UN-recognized country plus a few commonly-included
  territories (Kosovo, Palestine, Taiwan, Vatican City).
- **Image dimensions:** 640×640, stretch-resized by Roboflow's
  pre-processing (matches this project's detector input size, see
  `vision/detector.py` ASSUMPTION D1).
- **Annotation format:** YOLO `.txt` labels, one file per image, one line
  per box: `class_id x_center y_center width height`, all normalized to
  `[0, 1]`.
- **Augmentation:** Roboflow generated 10 versions of each annotated source
  image — random 90° rotation (or none), ±40% brightness, ±27% exposure,
  0–1.8px Gaussian blur, salt-and-pepper noise on ~1.92% of pixels, and
  ±45° bounding-box rotation.

### Background images

The 2,126 background/negative images are not blank filler — filename
evidence (`playground_*`, `ScreenShot*`, `DAAG_*`) shows they were pulled in
from other Roboflow projects (soccer-field, tarp, and airport/desert
detection datasets) as diverse no-flag scenes, so the model learns to stay
quiet on backgrounds it will actually see from altitude rather than only on
blank frames.

## How `training/train.py` uses this

`train.py` defaults to `training/flag detection.v1i.yolov11/data.yaml`
(`_DEFAULT_DATA_YAML`, resolved relative to the script's own location, not
hardcoded to any machine). It parses `data.yaml` (via PyYAML if available,
with a hand-rolled fallback parser otherwise — see the module docstring),
validates that every image referenced by a split actually has a matching
label-or-background pair, computes dataset-quality statistics (background
ratio, per-class imbalance, low-count-class warnings), and only then hands
the path to Ultralytics for training.

## Pointing the code at a fresh download

1. Export a new version from the
   [Roboflow project](https://app.roboflow.com/maher-ahmed/flag-detection-0swrg/1)
   in **YOLOv11** format.
2. Unzip it anywhere under `training/` (or elsewhere on disk).
3. Point `train.py` at the new `data.yaml` with `--data`:

   ```bash
   python -m training.train --data "/path/to/new/export/data.yaml"
   ```

   Everything downstream (`export.py`, `masked_validation_experiment.py`,
   `resolution_and_remap_report.py`, `threshold_sweep_report.py`) reads the
   trained weights and dataset paths from CLI flags or `logs/train_summary.json`
   rather than hardcoding this dataset's location, so a fresh dataset only
   needs to be wired in at the `train.py` step.
