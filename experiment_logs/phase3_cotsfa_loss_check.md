# Phase 3-1 Co-TSFA Alignment Loss Check

## Purpose

This document records the initial implementation and sanity check of the Co-TSFA-style alignment loss.

Phase 3 aims to implement a Co-TSFA-style baseline by adding an alignment objective between latent representation changes and forecasting target changes.

---

## Implemented Module

The Co-TSFA-style alignment loss was implemented in:

- `custom/losses/cotsfa_loss.py`

The module includes:

- `flatten_for_similarity`
- `cosine_similarity_batch`
- `CoTSFAAlignmentLoss`

---

## Loss Definition

The alignment loss compares latent similarity and output-target similarity.

- `sim_latent`: cosine similarity between `z` and `z_aug`
- `sim_output`: cosine similarity between `y` and `y_aug`

The loss is defined as:

`L_align = mean(|sim_latent - sim_output|)`

For input-only anomalies, `y_aug = y`, so `sim_output` should be close to 1.

In this case, the latent representation should also remain similar.

For input-output anomalies, `y_aug` differs from `y`, so `sim_output` becomes smaller.

In this case, the latent representation is allowed to change.

---

## Sanity Check 1: Input-only Case

In this case, `y_aug = y`.

### Result

| Metric | Value |
|---|---:|
| loss | 0.0000468 |
| sim_latent | 0.999953 |
| sim_output | 1.000000 |

### Interpretation

The output target did not change, and the latent representation was also nearly identical.

The alignment loss was close to zero, which is the expected behavior.

---

## Sanity Check 2: Input-output Changed Case

In this case, `y_aug` was changed by adding random perturbation.

### Result

| Metric | Value |
|---|---:|
| loss | 0.097633 |
| sim_latent | 0.999953 |
| sim_output | 0.902321 |

### Interpretation

The output target changed, but the latent representation remained almost identical.

Therefore, the gap between `sim_latent` and `sim_output` increased, producing a larger alignment loss.

This confirms that the loss reacts to mismatch between latent change and target change.

---

## Conclusion

The Co-TSFA-style alignment loss behaves as expected in toy sanity checks.

Phase 3-1 is complete.

The next step is Phase 3-2: modifying TimesNet to expose latent representations during forward propagation.