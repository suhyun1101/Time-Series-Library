# Informs Research Project

This repository is based on [Time-Series-Library](https://github.com/thuml/Time-Series-Library) and is used for developing an anomaly-aware robust time-series forecasting method.

## Project Goal

The goal of this project is to develop and evaluate a shape-conditioned contrastive regularization framework for robust time-series forecasting under diverse anomaly types.

The project focuses on the question of whether different anomaly shapes should be treated differently during representation learning and forecasting.

---

## Base Codebase

- Base repository: Time-Series-Library
- Main baseline model: TimesNet
- Initial dataset: ETTh1
- Initial forecasting horizon: 96

This repository uses Time-Series-Library as the implementation backbone.

The original codebase is preserved as much as possible, and project-specific implementations will be added separately.

---

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| Phase 1 | Environment setup and clean baseline reproduction | Done |
| Phase 2 | Anomaly augmentation implementation | Done |
| Phase 3 | Co-TSFA-style baseline implementation | Not started |
| Phase 4 | Shape-conditioned proposed method | Not started |
| Phase 5 | Ablation studies and paper writing | Not started |

---

## Phase 1 Summary

Phase 1 checked whether the Time-Series-Library codebase can successfully run a clean TimesNet baseline on ETTh1.

### Clean Baseline Result

| Model | Dataset | Prediction Length | MSE | MAE |
|---|---|---:|---:|---:|
| TimesNet | ETTh1 | 96 | 0.3887 | 0.4118 |

The clean baseline was successfully reproduced.

This confirms that the codebase, dataset path, conda environment, and execution pipeline are correctly set up.

Detailed logs are stored in:

- `experiment_logs/phase1_timesnet_etth1_96_clean.md`
- `experiment_logs/results_summary.csv`

---

## Phase 2 Summary

Phase 2 implemented anomaly augmentation functions and evaluated the clean TimesNet checkpoint under corrupted test conditions.

Implemented anomaly types:

- Continuous perturbation
- Gaussian pointwise noise
- Missing observation
- Pointwise spike
- Level shift
- Variance burst

The corrupted evaluation showed that small input-only perturbations caused minor degradation, while input-output anomalies such as level shift and variance burst caused much larger performance drops.

Key corrupted evaluation results:

| Condition | MSE | MAE |
|---|---:|---:|
| clean | 0.388676 | 0.411777 |
| continuous_input_only | 0.389327 | 0.412324 |
| gaussian_pointwise_input_only | 0.390497 | 0.413033 |
| missing_observation_input_only | 0.391175 | 0.413599 |
| pointwise_spike_input_only | 0.447148 | 0.445215 |
| level_shift_input_output | 0.615161 | 0.551453 |
| variance_burst_input_output | 1.552697 | 0.819610 |

Phase 2 is considered complete.