# Phase 3-2 TimesNet Latent Representation Check

## Purpose

This document records the check for extracting latent representations from TimesNet for Co-TSFA-style training.

The goal of Phase 3-2 is to verify that the model can return both forecasting outputs and latent representations.

---

## Implemented Module

The TimesNet latent extraction wrapper was implemented in:

- `custom/models/timesnet_cotsfa.py`

The wrapper preserves the original TimesNet behavior and adds:

- `forecast_with_latent`
- `forward_with_latent`
- `forward_cotsfa`

---

## Latent Representation

The latent representation is extracted from `enc_out` after the TimesBlock layers and before the final projection layer.

For this check, the prediction-horizon latent representation was used.

Expected shape:

- forecast output: `(B, pred_len, C)`
- latent representation: `(B, pred_len, d_model)`

---

## Shape Check

Dummy input setting:

| Item | Value |
|---|---:|
| Batch size | 4 |
| Seq Len | 96 |
| Label Len | 48 |
| Pred Len | 96 |
| Number of variables | 7 |
| d_model | 16 |

Observed output shapes:

| Output | Shape |
|---|---|
| standard y_hat | `[4, 96, 7]` |
| y_hat_latent | `[4, 96, 7]` |
| z | `[4, 96, 16]` |
| y_hat_aug | `[4, 96, 7]` |
| z_aug | `[4, 96, 16]` |

The shapes are correct.

---

## Latent Similarity Check

The similarity between `z` and `z_aug` was computed using batch-wise cosine similarity.

Result:

| Metric | Value |
|---|---:|
| sim_z mean | 0.998778 |

This confirms that the latent representations can be compared with the Co-TSFA-style alignment loss.

---

## Alignment Loss Check

### Input-only Case

In the input-only case, `y_aug = y`.

| Metric | Value |
|---|---:|
| loss_align | 0.001221 |
| sim_latent_mean | 0.998778 |
| sim_output_mean | 1.000000 |

The alignment loss is small, as expected.

### Input-output Changed Case

In the input-output changed case, `y_aug` was perturbed.

| Metric | Value |
|---|---:|
| loss_align | 0.103537 |
| sim_latent_mean | 0.998778 |
| sim_output_mean | 0.895242 |

The output similarity decreased, and the alignment loss increased. This confirms that the loss reacts to target changes.

---

## Conclusion

The CoTSFATimesNet wrapper successfully returns both forecast outputs and latent representations.

Phase 3-2 is complete.

The next step is Phase 3-3: connecting anomaly augmentation and Co-TSFA-style alignment loss to the training loop.