# Corrected UR Fall Fusion Leaderboard Summary

## Label sanity check

Rows: 11505

Sequences: 70

Raw label counts: {-1: 7451, 1: 2335, 0: 1719}

Corrected binary label counts: {0: 7451, 1: 4054}

Corrected alert rate: 0.352


## Clean leaderboard

| method                        |   f1_mean |   precision_mean |   recall_mean |   balanced_accuracy_mean |   false_alert_rate_mean |
|:------------------------------|----------:|-----------------:|--------------:|-------------------------:|------------------------:|
| Temporal Logistic Fusion      |  0.862725 |         0.895028 |      0.835304 |                 0.891455 |               0.0523945 |
| Temporal Random Forest        |  0.851567 |         0.86006  |      0.849561 |                 0.887882 |               0.073798  |
| Calibrated Logistic Fusion    |  0.850657 |         0.88759  |      0.823882 |                 0.884009 |               0.0558653 |
| Temporal Extra Trees          |  0.849815 |         0.86418  |      0.843142 |                 0.886285 |               0.070572  |
| Temporal HistGradientBoosting |  0.847351 |         0.846574 |      0.856269 |                 0.88606  |               0.0841487 |
| Max OR Fusion                 |  0.84675  |         0.858184 |      0.839223 |                 0.881327 |               0.0765694 |
| Rule OR Tuned                 |  0.84675  |         0.858184 |      0.839223 |                 0.881327 |               0.0765694 |
| Naive Independence            |  0.846229 |         0.843119 |      0.854735 |                 0.883378 |               0.0879794 |
| Camera Only                   |  0.841919 |         0.889841 |      0.809959 |                 0.87771  |               0.0545386 |
| Mean Late Fusion              |  0.832728 |         0.798365 |      0.875427 |                 0.875846 |               0.123735  |
| Product Agreement Fusion      |  0.817046 |         0.82636  |      0.815054 |                 0.859584 |               0.0958861 |
| Min Agreement Fusion          |  0.790156 |         0.799603 |      0.795131 |                 0.840447 |               0.114237  |
| Rule AND Tuned                |  0.767347 |         0.926128 |      0.661576 |                 0.815555 |               0.0304655 |
| Accelerometer Only            |  0.740553 |         0.74252  |      0.767109 |                 0.802451 |               0.162206  |


## Best method by stress severity

|   severity | method                   |   f1_mean |   precision_mean |   recall_mean |   false_alert_rate_mean |
|-----------:|:-------------------------|----------:|-----------------:|--------------:|------------------------:|
|        0   | Temporal Logistic Fusion |  0.862725 |         0.895028 |      0.835304 |               0.0523945 |
|        0.2 | Temporal Logistic Fusion |  0.865219 |         0.906608 |      0.831748 |               0.0448526 |
|        0.4 | Temporal Logistic Fusion |  0.872185 |         0.90926  |      0.840349 |               0.0441856 |
|        0.6 | Temporal Logistic Fusion |  0.87149  |         0.898381 |      0.849014 |               0.0504647 |
|        0.8 | Temporal Logistic Fusion |  0.867276 |         0.894196 |      0.845235 |               0.0526728 |
|        1   | Temporal Logistic Fusion |  0.861681 |         0.877952 |      0.849736 |               0.0624026 |