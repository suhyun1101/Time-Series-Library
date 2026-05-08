# Phase 2-2 Corrupted Evaluation

## Purpose

This experiment evaluates the clean TimesNet checkpoint under different corrupted test-input conditions.

The goal is to examine how the base TimesNet model behaves when different anomaly shapes are injected into the test sequence.

This experiment is part of Phase 2.  
Phase 2 focuses on building and validating anomaly augmentation functions before implementing Co-TSFA-style or shape-conditioned training objectives.

---

## Base Model

| Item | Value |
|---|---|
| Model | TimesNet |
| Dataset | ETTh1 |
| Task | Long-term forecasting |
| Seq Len | 96 |
| Label Len | 48 |
| Pred Len | 96 |
| Features | M |
| Checkpoint | Phase 1 clean baseline checkpoint |
| Device | CPU |

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

## Important Design Decision

For input-only anomalies, only the input sequence x is corrupted and the forecasting target y is kept clean.

For input-output anomalies, both x and the full decoder-side y sequence are corrupted. This includes both the label_len context and the pred_len target.

This is intentional because persistent forecast-relevant anomalies should remain consistent across the encoder input, decoder context, and future target.

Although the full decoder-side y sequence is corrupted for input-output anomalies, evaluation metrics are computed only on the pred_len forecasting target.

---

## Sanity Check

The clean evaluation result matched the Phase 1 baseline result.

| Condition | MSE | MAE |
|---|---:|---:|
| Phase 1 clean baseline | 0.388676 | 0.411777 |
| Phase 2 clean evaluation | 0.388676 | 0.411777 |

This confirms that the checkpoint was loaded correctly and that the corrupted evaluation pipeline is consistent with the Phase 1 clean baseline.

---

## Result

| Condition | MSE | MAE |
|---|---:|---:|
| clean | 0.388676 | 0.411777 |
| continuous_input_only | 0.389327 | 0.412324 |
| gaussian_pointwise_input_only | 0.390497 | 0.413033 |
| missing_observation_input_only | 0.391175 | 0.413599 |
| pointwise_spike_input_only | 0.447148 | 0.445215 |
| level_shift_input_output | 0.615161 | 0.551453 |
| variance_burst_input_output | 1.552697 | 0.819610 |

---

## Performance Change from Clean Baseline

| Condition | Delta MSE | Delta MAE |
|---|---:|---:|
| continuous_input_only | +0.000651 | +0.000547 |
| gaussian_pointwise_input_only | +0.001821 | +0.001256 |
| missing_observation_input_only | +0.002499 | +0.001822 |
| pointwise_spike_input_only | +0.058472 | +0.033438 |
| level_shift_input_output | +0.226485 | +0.139676 |
| variance_burst_input_output | +1.164021 | +0.407833 |

---

## Interpretation

Small input-only perturbations, such as continuous perturbation, Gaussian pointwise noise, and missing observation, caused only minor degradation compared to the clean baseline.

Pointwise spike caused a clearer performance drop, even though it was applied only to the input sequence. This suggests that large local perturbations can still affect the forecast.

Input-output anomalies caused much larger degradation. Level shift substantially increased both MSE and MAE, while variance burst caused the largest performance degradation.

The results indicate that anomaly impact differs strongly by anomaly shape. Therefore, anomalies should not be treated as a single uniform noise type.

This supports the motivation for shape-aware or shape-conditioned anomaly-aware forecasting methods.

---

## Conclusion

The corrupted evaluation pipeline runs successfully.

The clean condition reproduced the Phase 1 baseline result, confirming that the checkpoint loading and evaluation procedure are correct.

The results show that TimesNet is relatively stable under small input-only perturbations but is much more vulnerable to large local spikes and persistent input-output anomalies.

This result supports the next phase of the project: implementing a Co-TSFA-style baseline and later extending it with shape-conditioned contrastive regularization.

---

## Next Step

The next step is Phase 3: Co-TSFA-style baseline implementation.

The key idea is to compare the change in latent representations with the change in forecasting targets under different augmented inputs.

This Phase 2 result will be used as the base corrupted evaluation reference.