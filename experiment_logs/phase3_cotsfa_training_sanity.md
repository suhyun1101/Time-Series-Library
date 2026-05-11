# Phase 3-3 Co-TSFA Training Sanity Check

## Purpose

This document records the first sanity check for connecting Co-TSFA-style alignment loss to the training loop.

The goal is not to train a full model yet.  
The goal is to verify that the following steps work correctly:

- Load ETTh1 train data
- Build CoTSFATimesNet
- Create augmented input
- Forward clean and augmented inputs
- Compute forecast loss
- Compute Co-TSFA-style alignment loss
- Combine losses
- Run backward propagation and optimizer step

---

## Implemented Script

The sanity check script was implemented in:

- `custom/training/train_cotsfa_sanity.py`

---

## Augmentation Setting

The sanity check used an input-only Gaussian pointwise noise augmentation.

| Item | Value |
|---|---|
| Augmentation type | Gaussian pointwise noise |
| Mode | input-only |
| Ratio | 0.10 |
| Std scale | 0.70 |
| Independent channels | False |

Because this is an input-only anomaly, the target sequence is kept unchanged.

---

## Shape Check

Observed shapes:

| Tensor | Shape |
|---|---|
| batch_x | `[8, 96, 7]` |
| batch_x_aug | `[8, 96, 7]` |
| y_hat | `[8, 96, 7]` |
| y_hat_aug | `[8, 96, 7]` |
| z | `[8, 96, 16]` |
| z_aug | `[8, 96, 16]` |

All shapes are correct.

---

## Loss Check

Observed values:

| Metric | Value |
|---|---:|
| loss_forecast_clean | 0.697380 |
| loss_forecast_aug | 0.700192 |
| loss_forecast | 0.698786 |
| loss_align | 0.154262 |
| sim_latent | 0.845738 |
| sim_output | 1.000000 |
| loss_total | 0.714212 |

---

## Interpretation

The input-only setting keeps the target unchanged, so the output-target similarity is 1.0.

The latent similarity is lower because the model is newly initialized and has not yet been trained with the alignment objective.

The important result is that the total loss was successfully computed and backpropagation ran without errors.

This confirms that the Co-TSFA-style training components can be connected in a training loop.

---

## Conclusion

Phase 3-3 sanity check is successful.

The next step is to implement a full Co-TSFA-style training script that trains the model for multiple epochs and compares clean and corrupted evaluation results.