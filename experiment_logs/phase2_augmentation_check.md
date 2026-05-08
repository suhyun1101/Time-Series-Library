# Phase 2 Augmentation Visual Check

## Purpose

This document records the first visual and functional check of anomaly augmentation functions.

## Implemented Augmentation Types

- Continuous perturbation
- Gaussian pointwise noise
- Missing observation
- Pointwise spike
- Level shift
- Variance burst

## Input-only Check

Gaussian pointwise noise was applied only to the input sequence.

Result:

- x changed: 0.06246786
- y changed: 0.0

This confirms that the input-only setting keeps the forecasting target unchanged.

## Input-output Check

Level shift was applied to both input and output sequences.

Result:

- x changed: 0.5847247
- y changed: 1.4393223864691598
- shift shape: (1, 1, 7)

This confirms that the input-output setting applies a persistent shift to both the input and target sequences.

## Conclusion

The first implementation of anomaly augmentation functions works as expected.

The next step is to connect these augmentation functions to the evaluation or training pipeline.