# Phase 3B Original Co-TSFA Augmentation Check

## Purpose

This document records the visual and functional check of the original Co-TSFA anomaly curve implementation.

The goal is to verify that the paper-style anomaly curve works correctly before implementing the original Co-TSFA contrastive alignment loss.

This step is part of Phase 3B, whose goal is to move from the earlier Co-TSFA-style prototype toward a more faithful reproduction of the original Co-TSFA method.

---

## Implemented Module

The original Co-TSFA anomaly curve was implemented in:

- `custom/cotsfa_original/augmentations.py`

The module includes:

- paper-style anomaly curve
- input-only augmentation
- input-output augmentation
- multiple augmentations per sequence
- raw curve validation
- optional scale diagnostic mode

The original Co-TSFA reproduction code is separated from the earlier prototype code.

- `custom/augmentations/`: previous multi-shape prototype augmentations
- `custom/cotsfa_original/`: original Co-TSFA reproduction modules

---

## Curve Formula

The implemented anomaly curve is:

`a(t) = A * t * exp(-B * t^C) / Z`

Used parameters:

| Parameter | Value |
|---|---:|
| B | 0.385 |
| Z | 90409 |
| A mean | 74120 |
| A std | 20000 |
| C mean | 0.806 |
| C std | 0.2 |

The default reproduction setting is:

| Setting | Value |
|---|---|
| normalize_curve | False |
| scaling_mode | raw |
| strength | 1.0 |
| random sign flipping | disabled |
| number of augmentations | 5 |

Random sign flipping is disabled because the original curve is constrained to be non-negative.

---

## Curve Validity Conditions

The raw anomaly curve is checked using the following constraints:

| Condition | Requirement |
|---|---|
| Non-negativity | curve values should be greater than or equal to 0 |
| Maximum value | max curve value should be less than 2.0 |
| t=30 value | curve value at t=30 should be less than 0.4 |

If a sampled curve does not satisfy the constraints, the curve parameters are resampled.

---

## Raw Curve Check

Observed raw curve statistics:

| Metric | Value |
|---|---:|
| curve min | 0.0000188 |
| curve max | 1.010845 |
| curve t=30 value | 0.062781 |
| max < 2.0 | True |
| t30 < 0.4 | True |

The raw curve satisfies the paper-style validity constraints.

---

## ETTh1 Scale Check

ETTh1 sample statistics:

| Metric | Value |
|---|---:|
| x min | 0.355 |
| x max | 33.133 |
| x mean | 7.055485 |
| x std | 7.644119 |

The raw Co-TSFA curve is valid, but its magnitude is relatively small compared with the ETTh1 time-series scale.

Therefore, the raw setting is kept as the faithful reproduction baseline, while additional strength values are checked as ETTh1-calibrated variants.

---

## Input-only Check

Input-only augmentation was applied using the raw setting.

Result:

| Metric | Value |
|---|---:|
| x changed | 0.048823 |
| y changed | 0.000000 |
| curve valid | True |
| scaling mode | raw |

Metadata:

| Item | Value |
|---|---:|
| start_idx | 4 |
| curve_len | 92 |
| raw_curve_max | 0.588593 |
| raw_curve_t30 | 0.000846 |
| applied_curve_max | 0.588593 |

This confirms that input-only augmentation changes only the input sequence while keeping the forecasting target unchanged.

---

## Input-output Check

Input-output augmentation was applied using the raw setting.

Result:

| Metric | Value |
|---|---:|
| x changed | 0.111359 |
| y changed | 0.071334 |
| curve valid | True |
| scaling mode | raw |

Metadata:

| Item | Value |
|---|---:|
| start_idx | 86 |
| curve_len | 106 |
| raw_curve_max | 1.268932 |
| raw_curve_t30 | 0.074029 |
| applied_curve_max | 1.268932 |

This confirms that input-output augmentation changes both the input and the forecasting target.

---

## Raw vs Std-scaled Diagnostic Check

The raw setting was compared with std-scaled diagnostic mode.

| Setting | Applied Curve Max | x changed | y changed |
|---|---:|---:|---:|
| raw | 1.268932 | 0.111359 | 0.071334 |
| std-scaled | 5.390352 | 0.192006 | 0.122996 |

The std-scaled curve is much stronger than the raw curve and may dominate the original time series.

Therefore, std scaling is not used as the default reproduction setting.

---

## Multiple Augmentation Check

The paper uses multiple augmentations per sequence.

A=5 augmentation result:

| Output | Shape |
|---|---|
| x_augs | `(5, 96, 7)` |
| y_augs | `(5, 96, 7)` |
| metadata count | 5 |

Example metadata summary:

| Augmentation Index | Start Index | Curve Length | Curve Valid | Raw Curve Max | Raw Curve t30 | Scaling Mode |
|---:|---:|---:|---|---:|---:|---|
| 0 | 85 | 107 | True | 0.860503 | 0.000145 | raw |
| 1 | 87 | 105 | True | 0.628351 | 0.000000 | raw |
| 2 | 88 | 104 | True | 0.673690 | 0.000000 | raw |
| 3 | 87 | 105 | True | 0.688991 | 0.000038 | raw |
| 4 | 88 | 104 | True | 0.661713 | 0.069347 | raw |

This confirms that five augmentations per sequence are generated correctly.

---

## Strength Sensitivity Check on ETTh1

Because the original curve parameters were estimated from the paper's data scale, the raw curve magnitude may not perfectly match ETTh1.

Therefore, raw Co-TSFA curves with different strength values were compared.

| Strength | x changed | y changed | Applied Curve Max |
|---:|---:|---:|---:|
| 1.0 | 0.111359 | 0.071334 | 1.268932 |
| 3.0 | 0.334076 | 0.214003 | 3.806795 |
| 5.0 | 0.556794 | 0.356672 | 6.344658 |

The OT-channel scale comparison was:

| Strength | Mean Abs Diff | Max Abs Diff | Mean Diff / OT Std | Max Diff / OT Std | Max Diff / OT Range |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.091347 | 1.268932 | 0.019238 | 0.267241 | 0.059926 |
| 3.0 | 0.274040 | 3.806795 | 0.057714 | 0.801724 | 0.179778 |
| 5.0 | 0.456733 | 6.344660 | 0.096189 | 1.336206 | 0.299630 |

---

## Strength Interpretation

### Strength 1.0

`strength=1.0` is the most faithful raw reproduction setting.

However, on ETTh1, the anomaly is relatively weak.

Observed values:

| Metric | Value |
|---|---:|
| mean diff / OT std | 0.019238 |
| max diff / OT std | 0.267241 |
| max diff / OT range | 0.059926 |

This setting should be used as the default raw Co-TSFA reproduction setting.

### Strength 3.0

`strength=3.0` produces a visible anomaly without completely dominating the original time series.

Observed values:

| Metric | Value |
|---|---:|
| mean diff / OT std | 0.057714 |
| max diff / OT std | 0.801724 |
| max diff / OT range | 0.179778 |

This setting is a reasonable ETTh1-calibrated variant.

### Strength 5.0

`strength=5.0` produces a strong anomaly.

Observed values:

| Metric | Value |
|---|---:|
| mean diff / OT std | 0.096189 |
| max diff / OT std | 1.336206 |
| max diff / OT range | 0.299630 |

This setting is too strong for the default reproduction setting, but it may be useful as a stress-test variant.

---

## Final Decision

The default Co-TSFA reproduction setting will remain:

- `scaling_mode="raw"`
- `strength=1.0`
- `normalize_curve=False`
- no random sign flipping
- A=5 augmentations per sequence

This setting is the most faithful raw reproduction setting.

However, on ETTh1, the anomaly magnitude with `strength=1.0` may be relatively weak compared with the OT-channel scale. In particular, the maximum perturbation is about 26.7% of the OT standard deviation.

Therefore, if the alignment loss signal is too weak during training with `strength=1.0`, the ETTh1-calibrated setting will be used:

- `scaling_mode="raw"`
- `strength=3.0`

The `strength=3.0` setting produces a visible anomaly without completely dominating the original time series, so it will be kept as the main ETTh1-calibrated variant.

The `strength=5.0` setting will be treated only as a possible stress-test setting.

The std-scaled setting is not used as the default because it is too strong and introduces dataset-dependent scaling.

---

## Conclusion

The original Co-TSFA anomaly curve implementation works as expected.

The implementation now supports:

- valid raw curve generation
- input-only augmentation
- input-output augmentation
- A=5 multiple augmentations
- strength sensitivity checks
- raw reproduction and ETTh1-calibrated variants

The next step is to implement the original Co-TSFA InfoNCE-style latent-output alignment loss in:

- `custom/cotsfa_original/losses.py`