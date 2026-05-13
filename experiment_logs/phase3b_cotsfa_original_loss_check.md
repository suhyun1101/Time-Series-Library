# Phase 3B Original Co-TSFA Alignment Loss Check

## Purpose

This document records the sanity check of the original Co-TSFA InfoNCE-style latent-output alignment loss.

The goal is to verify that the original Co-TSFA-style contrastive score and alignment loss work correctly before connecting them to the training loop.

---

## Implemented Module

The original Co-TSFA alignment loss was implemented in:

- `custom/cotsfa_original/losses.py`

The module includes:

- `paper_contrastive_score`
- `CoTSFAOriginalAlignmentLoss`
- `CoTSFAOriginalLossOutput`

This loss is separated from the earlier prototype loss.

- Prototype loss: cosine-similarity-based simplified alignment
- Original Co-TSFA loss: InfoNCE-style dot-product alignment with batch negatives

---

## Input Tensor Shapes

The sanity check used dummy tensors with the following shapes:

| Tensor | Shape |
|---|---|
| z | `[8, 96, 16]` |
| z_aug | `[5, 8, 96, 16]` |
| y | `[8, 96, 7]` |
| y_aug_input_only | `[5, 8, 96, 7]` |

where:

| Symbol | Meaning | Value |
|---|---|---:|
| A | number of augmentations | 5 |
| B | batch size | 8 |
| T | prediction length | 96 |
| Dz | latent dimension | 16 |
| Dy | target dimension | 7 |

---

## Contrastive Score Shape Check

The InfoNCE-style score function returned the following shapes:

| Score | Shape |
|---|---|
| latent_scores | `[5, 8, 96]` |
| target_scores | `[5, 8, 96]` |

This confirms that the score is computed for each augmentation, batch sample, and prediction time step.

Observed score statistics:

| Metric | Value |
|---|---:|
| latent_scores mean | 1.631487 |
| target_scores mean | 1.755358 |
| latent_scores std | 0.187020 |
| target_scores std | 0.274563 |

---

## Input-only Case

In the input-only case, the target was unchanged:

- `y_aug = y`

Result:

| Metric | Value |
|---|---:|
| loss_align | 0.232351 |
| latent_score_mean | 1.631487 |
| target_score_mean | 1.755358 |

The loss is not zero because the latent representation was perturbed while the target remained unchanged.

This is expected in the dummy sanity check.

---

## Input-output Changed Case

In the input-output changed case, the target was perturbed.

Result:

| Metric | Value |
|---|---:|
| loss_align | 1.152070 |
| latent_score_mean | 1.631487 |
| target_score_mean | 2.372596 |
| latent_score_std | 0.187020 |
| target_score_std | 1.316339 |

Compared with the input-only case, the target score changed substantially and the alignment loss increased.

This confirms that the loss reacts to target-space changes.

---

## Backward Check

Backward propagation was also tested.

| Metric | Value |
|---|---|
| z grad is None | False |
| z_aug grad is None | False |
| z grad mean abs | 0.0000216 |
| z_aug grad mean abs | 0.0001599 |

This confirms that the alignment loss can propagate gradients to latent representations and augmented latent representations.

---

## Conclusion

The original Co-TSFA InfoNCE-style alignment loss works as expected in the sanity check.

The implementation supports:

- A=5 augmentations
- batch-wise negative pairs
- augmentation-specific denominators
- original batch negatives
- same-sample multiple augmentation terms
- latent-output score alignment
- gradient backpropagation

The next step is to connect this loss to the original Co-TSFA training loop.