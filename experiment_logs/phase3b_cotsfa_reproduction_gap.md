# Phase 3B Co-TSFA Reproduction Gap Analysis

## Purpose

This document separates the current Co-TSFA-style prototype from the original Co-TSFA reproduction target.

The experiments completed so far are useful for implementation practice and pipeline validation, but they should not be reported as an official reproduction of the original Co-TSFA paper.

The goal of Phase 3B is to reproduce the original Co-TSFA method as faithfully as possible before developing the proposed extension.

---

## Current Implementation Status

The current implementation includes:

- anomaly augmentation functions
- input-only and input-output anomaly concepts
- TimesNet latent representation extraction
- Co-TSFA-style latent-output alignment loss
- Co-TSFA-style training loop
- corrupted evaluation pipeline
- pilot and full prototype training on ETTh1

This confirms that the overall training and evaluation pipeline works.

However, the current implementation is a simplified Co-TSFA-style prototype, not a faithful reproduction of the original Co-TSFA paper.

---

## Why This Distinction Matters

The proposed method should be developed only after establishing a reliable baseline.

If the existing Co-TSFA method is not reproduced or implemented faithfully, it is difficult to claim that the proposed method improves upon Co-TSFA.

Therefore, before moving to shape-conditioned alignment, we need to reproduce the original Co-TSFA method more carefully.

---

## Main Gaps from Original Co-TSFA

| Component | Original Co-TSFA | Current Prototype |
|---|---|---|
| Anomaly generation | Paper-specific anomaly curve | Multi-shape anomaly functions |
| Anomaly curve | Continuous curve defined by the paper | Continuous, Gaussian, missing, spike, level shift, variance burst |
| Augmentation count | Multiple augmentations per sequence, A = 5 | One augmentation per batch |
| Input-only anomaly | Starts early and ends within input window | Custom input-only corruption |
| Input-output anomaly | Starts near the end of input and continues into output | Custom level shift / variance burst |
| Similarity function | Softmax dot-product contrastive similarity | Batch-wise cosine similarity |
| Negative pairs | Uses batch negatives | Not implemented |
| Dataset setting | Paper-specific Traffic/Electricity/Cash Demand/ETTh1 settings | ETTh1 96/96 prototype |
| Batch size | 128 | 32 |
| Learning rate schedule | Step decay | Fixed learning rate |
| Status | Original method | Prototype only |

---

## Reproduction Target

The next goal is to implement a faithful Co-TSFA reproduction.

Required components:

1. Implement the paper-specific anomaly curve.
2. Generate multiple augmentations per sequence.
3. Implement input-only anomaly timing.
4. Implement input-output anomaly timing.
5. Implement softmax dot-product contrastive similarity.
6. Include batch negative pairs.
7. Match the original training hyperparameters as closely as possible.
8. Match the original dataset settings as closely as possible.
9. Compare reproduced results with the reported paper results.

---

## Immediate Next Step

The first implementation target is the original Co-TSFA anomaly generation module.

We will create a separate folder for original Co-TSFA reproduction code:

- `custom/cotsfa_original/`

This keeps the original Co-TSFA reproduction code separate from the earlier prototype code.

---

## Naming Rule

The previous implementation should be called:

- Co-TSFA-style prototype

The new implementation should be called:

- Co-TSFA reproduction
- original Co-TSFA reimplementation
- faithful Co-TSFA implementation

Only after the reproduction is verified should we move to the proposed shape-conditioned extension.