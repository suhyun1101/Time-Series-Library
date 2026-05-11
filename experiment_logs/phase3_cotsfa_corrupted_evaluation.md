# Phase 3-4B Co-TSFA Pilot Corrupted Evaluation

## Purpose

This document records the corrupted evaluation results of the Co-TSFA-style pilot checkpoint.

The goal of this step is not to claim final performance improvement.

The goal is to verify that the Co-TSFA-style checkpoint can be loaded and evaluated under the same clean and corrupted test conditions used in Phase 2.

---

## Evaluated Model

| Item | Value |
|---|---|
| Model | CoTSFATimesNet |
| Dataset | ETTh1 |
| Seq Len | 96 |
| Label Len | 48 |
| Pred Len | 96 |
| Training Setting | pilot_2epochs_80batches_lambda0.1 |
| Device | CPU |
| Checkpoint | `checkpoints_cotsfa/cotsfa_etth1_96/checkpoint.pth` |

The checkpoint is stored locally and is not committed to GitHub.

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

## Co-TSFA Pilot Evaluation Result

| Condition | MSE | MAE |
|---|---:|---:|
| clean | 0.492361 | 0.476734 |
| continuous_input_only | 0.493174 | 0.477178 |
| gaussian_pointwise_input_only | 0.495861 | 0.478759 |
| missing_observation_input_only | 0.491029 | 0.474606 |
| pointwise_spike_input_only | 0.560616 | 0.511778 |
| level_shift_input_output | 0.734118 | 0.596749 |
| variance_burst_input_output | 1.856144 | 0.909026 |

Detailed CSV result is saved in:

- `experiment_logs/phase3_cotsfa_corrupted_eval_results.csv`

---

## Comparison with Base TimesNet

| Condition | Base TimesNet MSE | Co-TSFA Pilot MSE | Difference |
|---|---:|---:|---:|
| clean | 0.388676 | 0.492361 | +0.103685 |
| continuous_input_only | 0.389327 | 0.493174 | +0.103847 |
| gaussian_pointwise_input_only | 0.390497 | 0.495861 | +0.105364 |
| missing_observation_input_only | 0.391175 | 0.491029 | +0.099854 |
| pointwise_spike_input_only | 0.447148 | 0.560616 | +0.113468 |
| level_shift_input_output | 0.615161 | 0.734118 | +0.118957 |
| variance_burst_input_output | 1.552697 | 1.856144 | +0.303447 |

---

## Interpretation

The Co-TSFA-style pilot checkpoint was successfully loaded and evaluated under all clean and corrupted test conditions.

The Co-TSFA pilot model performs worse than the fully trained Base TimesNet model across all conditions. This is expected because the pilot model was trained only for 2 epochs with 80 batches per epoch, while the Phase 1 Base TimesNet was trained with early stopping until epoch 7.

Therefore, this result should not be interpreted as final Co-TSFA performance.

The important conclusion is that the full pipeline works:

- Co-TSFA-style training
- checkpoint saving
- checkpoint loading
- clean evaluation
- corrupted evaluation
- comparison with Base TimesNet

The anomaly-shape pattern remains consistent with Phase 2. Small input-only perturbations cause minor changes, while pointwise spike, level shift, and especially variance burst produce larger degradation.

---

## Conclusion

Phase 3-4B corrupted evaluation is successful.

The next step is to move from pilot training to a fuller Co-TSFA-style training run, or to begin tuning the Co-TSFA settings such as training epochs, augmentation mix, and lambda alignment weight.