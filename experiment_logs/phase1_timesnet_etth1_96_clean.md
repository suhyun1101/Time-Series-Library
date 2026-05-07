# Phase 1 Clean Baseline: TimesNet on ETTh1

## Purpose

This experiment checks whether the Time-Series-Library environment is correctly set up by running TimesNet on ETTh1 with prediction length 96.

## Setting

| Item | Value |
|---|---|
| Model | TimesNet |
| Dataset | ETTh1 |
| Task | Long-term forecasting |
| Seq Len | 96 |
| Label Len | 48 |
| Pred Len | 96 |
| Features | M |
| Device | CPU |

## Data Split

| Split | Samples |
|---|---:|
| Train | 8449 |
| Validation | 2785 |
| Test | 2785 |

## Result

| Metric | Value |
|---|---:|
| MSE | 0.3887 |
| MAE | 0.4118 |

## Notes

Training stopped at epoch 7 due to early stopping.

The clean baseline was successfully reproduced. This completes Phase 1 environment setup.