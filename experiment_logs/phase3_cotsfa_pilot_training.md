# Phase 3-4A Co-TSFA-style Pilot Training

## Purpose

This document records the first multi-batch pilot training run for the Co-TSFA-style baseline.

The purpose of this run is not to produce final performance results.

The goal is to verify that the following components can be connected in a real training loop:

- CoTSFATimesNet
- anomaly augmentation
- clean and augmented forward passes
- forecast loss
- Co-TSFA-style alignment loss
- total loss
- backward propagation
- optimizer update
- checkpoint saving
- training log saving

---

## Implemented Script

The pilot training script was implemented in:

- `custom/training/train_cotsfa_etth1.py`

The training log was saved to:

- `experiment_logs/phase3_cotsfa_training_log.csv`

The best checkpoint was saved locally to:

- `checkpoints_cotsfa/cotsfa_etth1_96/checkpoint.pth`

The checkpoint directory is not intended to be committed to GitHub.

---

## Training Setting

| Item | Value |
|---|---:|
| Model | CoTSFATimesNet |
| Dataset | ETTh1 |
| Seq Len | 96 |
| Label Len | 48 |
| Pred Len | 96 |
| Batch Size | 16 |
| Train Epochs | 2 |
| Max Train Batches | 80 |
| Max Validation Batches | 30 |
| Lambda Align | 0.1 |
| Device | CPU |

---

## Objective

The training objective is:

`L_total = L_forecast + lambda_align * L_align`

where:

- `L_forecast` is the average of clean forecast loss and augmented forecast loss.
- `L_align` is the Co-TSFA-style latent-output alignment loss.
- `lambda_align = 0.1`.

---

## Training Result

| Epoch | Train Total Loss | Train Forecast Loss | Train Align Loss | Clean Validation Loss | Time Sec |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.752790 | 0.739345 | 0.134448 | 1.048099 | 60.10 |
| 2 | 0.541052 | 0.535911 | 0.051414 | 0.803311 | 57.78 |

Best validation loss:

| Metric | Value |
|---|---:|
| Best clean validation loss | 0.803311 |

---

## Interpretation

The pilot training run completed successfully.

The total training loss decreased from epoch 1 to epoch 2.

The forecast loss also decreased, indicating that the model is learning the forecasting task during the pilot run.

The alignment loss decreased from 0.134448 to 0.051414, suggesting that the latent-output alignment objective is being optimized.

The clean validation loss also decreased from 1.048099 to 0.803311.

Because this was a short pilot run with only 2 epochs and 80 training batches per epoch, the result should not be compared directly with the full Phase 1 clean baseline.

The main conclusion is that Co-TSFA-style multi-batch training works without runtime errors.

---

## Conclusion

Phase 3-4A pilot training is successful.

The next step is to evaluate the saved Co-TSFA-style checkpoint under clean and corrupted test conditions, then compare it with the base TimesNet corrupted evaluation results.