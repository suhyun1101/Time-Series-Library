# Phase 3B Original Co-TSFA Training Sanity Check

## Purpose

This document records the first training-loop sanity check for the original Co-TSFA reproduction.

The goal is not to train a full model yet.

The goal is to verify that the original Co-TSFA components can be connected in a training loop:

- original Co-TSFA anomaly curve
- A=5 augmentations per sequence
- clean forward pass
- augmented forward passes
- clean-only forecast loss
- InfoNCE-style latent-output alignment loss
- total loss
- backward propagation
- optimizer step

---

## Implemented Script

The sanity check script was implemented in:

- `custom/training/train_cotsfa_original_etth1_sanity.py`

---

## Loss Definition

The original Co-TSFA training sanity check uses:

`L_total = L_forecast + lambda_align * L_align`

where:

- `L_forecast` is computed only on the clean branch.
- augmented samples are used only for the alignment loss.
- `L_align` is computed using the original Co-TSFA InfoNCE-style latent-output alignment loss.

This corrects the earlier prototype-style implementation where augmented forecast loss was also included.

---

## Configuration

| Item | Value |
|---|---:|
| Dataset | ETTh1 |
| Seq Len | 96 |
| Label Len | 48 |
| Pred Len | 96 |
| Batch Size | 8 |
| Number of augmentations | 5 |
| Scaling mode | raw |
| Strength | 1.0 |
| Lambda align | 0.1 |
| Forecast loss | clean branch only |
| Device | CPU |

---

## Input-only Result

Input-only augmentation changes only the input sequence while keeping the forecasting target unchanged.

### Shapes

| Tensor | Shape |
|---|---|
| batch_x | `(8, 96, 7)` |
| batch_y | `(8, 144, 7)` |
| batch_x_aug | `(5, 8, 96, 7)` |
| batch_y_aug | `(5, 8, 144, 7)` |
| y_hat | `(8, 96, 7)` |
| y_hat_aug | `(5, 8, 96, 7)` |
| z | `(8, 96, 16)` |
| z_aug | `(5, 8, 96, 16)` |
| y_true | `(8, 96, 7)` |
| y_aug_true | `(5, 8, 96, 7)` |

### Augmentation Check

| Metric | Value |
|---|---:|
| x_changed | 0.105418 |
| y_changed | 0.000000 |
| curve_valid_rate | 1.000000 |
| raw_curve_max_mean | 1.036500 |
| applied_curve_max_mean | 1.036500 |

### Loss Check

| Metric | Value |
|---|---:|
| loss_forecast_clean_only | 0.711100 |
| loss_align | 1.762847 |
| latent_score_mean | 2.955184 |
| target_score_mean | 2.042876 |
| latent_score_std | 2.731513 |
| target_score_std | 0.602156 |
| loss_total | 0.887385 |

---

## Input-output Result

Input-output augmentation changes both the input sequence and the forecasting target.

### Shapes

| Tensor | Shape |
|---|---|
| batch_x | `(8, 96, 7)` |
| batch_y | `(8, 144, 7)` |
| batch_x_aug | `(5, 8, 96, 7)` |
| batch_y_aug | `(5, 8, 144, 7)` |
| y_hat | `(8, 96, 7)` |
| y_hat_aug | `(5, 8, 96, 7)` |
| z | `(8, 96, 16)` |
| z_aug | `(5, 8, 96, 16)` |
| y_true | `(8, 96, 7)` |
| y_aug_true | `(5, 8, 96, 7)` |

### Augmentation Check

| Metric | Value |
|---|---:|
| x_changed | 0.051695 |
| y_changed | 0.030528 |
| curve_valid_rate | 1.000000 |
| raw_curve_max_mean | 0.830347 |
| applied_curve_max_mean | 0.830347 |

### Loss Check

| Metric | Value |
|---|---:|
| loss_forecast_clean_only | 0.695013 |
| loss_align | 1.186871 |
| latent_score_mean | 2.386704 |
| target_score_mean | 2.043725 |
| latent_score_std | 1.673981 |
| target_score_std | 0.603773 |
| loss_total | 0.813700 |

---

## Interpretation

The sanity check completed successfully.

The main purpose of this step was to verify whether the original Co-TSFA reproduction components can be connected in a training loop.

The following were confirmed:

- ETTh1 train batch can be loaded.
- A=5 original Co-TSFA augmentations can be generated.
- clean and augmented branches can be forwarded through CoTSFATimesNet.
- latent representations `z` and `z_aug` can be extracted.
- clean-only forecast loss can be computed.
- original InfoNCE-style alignment loss can be computed.
- total loss can be computed.
- backward propagation and optimizer update run without errors.

The input-only alignment loss is larger than the input-output alignment loss in this one-batch sanity check. This should not be interpreted as a performance result because the model is randomly initialized and only one batch was used.

The important conclusion is that the original Co-TSFA training structure is now technically connected.

---

## Conclusion

The original Co-TSFA training sanity check is successful.

The next step is to implement a short pilot training run using the original Co-TSFA loss and augmentation pipeline.