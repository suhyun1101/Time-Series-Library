# Phase 3-5B Full Co-TSFA Corrupted Evaluation

## Purpose

This document records the corrupted evaluation results of the full Co-TSFA-style checkpoint.

The goal is to compare the full Co-TSFA-style model with the Base TimesNet model under the same clean and corrupted test conditions.

---

## Evaluated Model

| Item | Value |
|---|---|
| Model | CoTSFATimesNet |
| Dataset | ETTh1 |
| Seq Len | 96 |
| Label Len | 48 |
| Pred Len | 96 |
| Training Setting | full_10epochs_best_epoch9_lambda0.1 |
| Device | CPU |
| Checkpoint | `checkpoints_cotsfa/cotsfa_etth1_96_full/checkpoint.pth` |

The checkpoint is stored locally and is not committed to GitHub.

---

## Full Training Summary

The full Co-TSFA-style model was trained for up to 10 epochs with early stopping patience 3.

The best validation loss was obtained at epoch 9.

| Item | Value |
|---|---:|
| Best epoch | 9 |
| Best clean validation loss | 0.720933 |
| Lambda align | 0.1 |
| Batch size | 32 |
| Max train epochs | 10 |

---

## Evaluation Conditions

| Condition | Description |
|---|---|
| clean | Original clean test input |
| continuous_input_only | Smooth continuous perturbation applied only to input x |
| gaussian_pointwise_input_only | Gaussian pointwise noise applied only to input x |
| missing_observation_input_only | Missing observations applied only to input x |
| pointwise_spike_input_only | Large spikes applied only to input x |
| level_shift_input_output | Persistent level shift applied to both x and y |
| variance_burst_input_output | Variance burst applied to both x and y |

---

## Full Co-TSFA Evaluation Result

| Condition | MSE | MAE |
|---|---:|---:|
| clean | 0.432055 | 0.429688 |
| continuous_input_only | 0.433156 | 0.430436 |
| gaussian_pointwise_input_only | 0.437743 | 0.432339 |
| missing_observation_input_only | 0.424353 | 0.427020 |
| pointwise_spike_input_only | 0.563061 | 0.488398 |
| level_shift_input_output | 0.574503 | 0.517950 |
| variance_burst_input_output | 1.545277 | 0.823017 |

Detailed CSV result is saved in:

- `experiment_logs/phase3_cotsfa_full_corrupted_eval_results.csv`

---

## Comparison with Base TimesNet

| Condition | Base TimesNet MSE | Full Co-TSFA MSE | Difference |
|---|---:|---:|---:|
| clean | 0.388676 | 0.432055 | +0.043379 |
| continuous_input_only | 0.389327 | 0.433156 | +0.043829 |
| gaussian_pointwise_input_only | 0.390497 | 0.437743 | +0.047246 |
| missing_observation_input_only | 0.391175 | 0.424353 | +0.033178 |
| pointwise_spike_input_only | 0.447148 | 0.563061 | +0.115913 |
| level_shift_input_output | 0.615161 | 0.574503 | -0.040658 |
| variance_burst_input_output | 1.552697 | 1.545277 | -0.007420 |

---

## Interpretation

The full Co-TSFA-style model improved substantially compared to the earlier pilot checkpoint.

Compared with the Base TimesNet model, the full Co-TSFA model has worse clean performance and worse performance on several input-only anomalies.

However, the full Co-TSFA model improved MSE under the input-output level shift condition and slightly improved MSE under the variance burst condition.

This suggests that the Co-TSFA-style alignment objective may help with some forecast-relevant anomalies, especially persistent input-output changes.

At the same time, the degradation under clean, small input-only, and pointwise spike conditions suggests that using a uniform alignment objective for all anomaly shapes is not sufficient.

This supports the motivation for the next phase: shape-conditioned alignment.

---

## Conclusion

Full Co-TSFA-style training and corrupted evaluation were successfully completed.

The results provide a meaningful baseline for Phase 4.

The key observation is that Co-TSFA-style training may improve robustness to some forecast-relevant anomaly shapes, but it does not uniformly improve all anomaly types.

Therefore, Phase 4 will focus on shape-conditioned weighting of the alignment loss.