# Colab Runner

Use this when you want to rerun the corrected event-level benchmark and the
stronger v2 results in Google Colab.

## Cell 1: install packages

```python
!pip -q install pandas numpy scikit-learn scipy matplotlib ultralytics opencv-python-headless
```

## Cell 2: clone or upload the repository

Recommended:

```python
!git clone YOUR_GITHUB_REPO_URL fall-fusion
%cd fall-fusion
```

If you upload files manually instead, upload the whole repository folder and
`%cd` into it.

## Cell 3: run the conservative UR benchmark

```python
!python scripts/ur_event_benchmark.py \
  --data-dir data/urfall \
  --out-dir results/ur_event \
  --bootstrap 2000
```

## Cell 4: run stronger UR depth + wearable model

```python
!python scripts/experiment_stronger_event_models.py \
  --data-dir data/urfall \
  --out-dir results/stronger_event
```

## Cell 5: run YOLO-pose modern video baseline

```python
!python scripts/ur_yolo_pose_baseline.py \
  --data-dir data/urfall \
  --video-dir data/urfall_videos \
  --pose-cache results/yolo_pose/pose_features.csv \
  --out-dir results/yolo_pose \
  --imgsz 256
```

## Cell 6: run full multimodal UR model

Run this after Cell 5:

```python
!python scripts/ur_full_multimodal_event_forest.py \
  --data-dir data/urfall \
  --pose-cache results/yolo_pose/pose_features.csv \
  --out-dir results/full_multimodal
```

## Cell 7: run external UMAFall replication

```python
!python scripts/umafall_external_benchmark.py \
  --data-dir data/umafall \
  --out-dir results/umafall
```

## Cell 8: inspect tables

```python
import pandas as pd

display(pd.read_csv("results/ur_event/ur_event_summary.csv"))
display(pd.read_csv("results/stronger_event/summary.csv"))
display(pd.read_csv("results/yolo_pose/yolo_pose_summary.csv"))
display(pd.read_csv("results/full_multimodal/summary.csv"))
display(pd.read_csv("results/umafall/umafall_summary.csv"))
```

## Cell 9: download all results

```python
from google.colab import files
!zip -r all_fall_results.zip results
files.download("all_fall_results.zip")
```
