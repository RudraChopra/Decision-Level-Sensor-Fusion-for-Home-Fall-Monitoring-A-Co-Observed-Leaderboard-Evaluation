# Decision Level Sensor Fusion for Home Fall Monitoring

This repository contains the reproducibility package for the paper:

**Decision Level Sensor Fusion for Home Fall Monitoring: A Co Observed Leaderboard Evaluation**

## What to run

Run the main notebook:

`notebooks/ur_fall_decision_fusion_benchmark.ipynb`

The notebook downloads the UR Fall Detection Dataset from the public UR endpoint, builds the co observed camera plus accelerometer frame table, checks the label mapping, trains the detectors, runs the decision fusion leaderboard, runs the stress protocol, and writes figures and tables.

## Critical label sanity check

The binary fall label is:

* `raw_label = -1` means upright normal, mapped to `label = 0`
* `raw_label = 0` means transitional fall alert, mapped to `label = 1`
* `raw_label = 1` means on ground fall alert, mapped to `label = 1`

The notebook stops if the fall frame rate is not in the expected range around 0.352.

## Repository structure

```text
notebooks/ur_fall_decision_fusion_benchmark.ipynb
paper/main.tex
paper/fig_system.png
paper/fig_f1_stress.png
paper/fig_far_stress.png
figures/
results/
```

## Main reported result

Temporal Logistic Fusion improves F1 over Camera Only from 0.842 to 0.863 on the corrected co observed benchmark. Rule AND Tuned achieves the lowest false alert rate, but lower recall.

## Requirements

See `requirements.txt`.
