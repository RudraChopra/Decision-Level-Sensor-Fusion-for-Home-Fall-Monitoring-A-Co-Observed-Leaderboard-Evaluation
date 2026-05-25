# Causal Multimodal Event Fusion for Fall Monitoring

This repository contains the code and manuscript source used for the revised fall-monitoring experiments:

- UR Fall event-level benchmark
- stronger UR depth + wearable temporal baselines
- YOLOv8 pose video baseline
- full depth + wearable + YOLO-pose multimodal fusion
- UMAFall external subject-held-out replication

The repository intentionally does **not** include downloaded datasets, videos, model weights, or large generated caches. The scripts download public datasets where possible.

## Quick Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the Main Results

Run commands from the repository root.

### 1. Conservative UR Event Benchmark

```bash
python scripts/ur_event_benchmark.py \
  --data-dir data/urfall \
  --out-dir results/ur_event \
  --bootstrap 2000
```

### 2. Stronger UR Depth + Wearable Event Models

This reuses the UR files downloaded in step 1.

```bash
python scripts/experiment_stronger_event_models.py \
  --data-dir data/urfall \
  --out-dir results/stronger_event
```

### 3. YOLO-Pose Video Baseline

This downloads UR RGB videos and a YOLOv8 pose checkpoint.

```bash
python scripts/ur_yolo_pose_baseline.py \
  --data-dir data/urfall \
  --video-dir data/urfall_videos \
  --pose-cache results/yolo_pose/pose_features.csv \
  --out-dir results/yolo_pose \
  --imgsz 256
```

### 4. Full Multimodal UR Fusion

Run after the YOLO-pose baseline, because it reuses the cached pose features.

```bash
python scripts/ur_full_multimodal_event_forest.py \
  --data-dir data/urfall \
  --pose-cache results/yolo_pose/pose_features.csv \
  --out-dir results/full_multimodal
```

### 5. UMAFall External Replication

```bash
python scripts/umafall_external_benchmark.py \
  --data-dir data/umafall \
  --out-dir results/umafall
```

## Key Results From Local Run

The strongest UR model was the full depth + wearable + YOLO-pose HGB event model:

- 30/30 falls detected
- 0 false alert episodes
- event F1: 1.000
- median latency: 0.451 s

External UMAFall replication:

- subject-held-out logistic inertial F1: 0.995
- AUROC: 1.000
- 746 trials from 19 subjects

## Manuscript

The Overleaf-ready paper is in:

```text
manuscript/revised_manuscript_v2.tex
manuscript/references.bib
```

