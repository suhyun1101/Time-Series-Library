"""
Phase 3B: Original Co-TSFA training sanity check on ETTh1.

This script checks whether the original Co-TSFA reproduction components can be
connected to a training loop.

It verifies:

1. Load ETTh1 train data
2. Build CoTSFATimesNet
3. Generate A=5 original Co-TSFA augmentations
4. Forward clean input
5. Forward augmented inputs
6. Compute clean-only forecast loss
7. Compute original InfoNCE-style alignment loss
8. Compute total loss
9. Run backward and optimizer step

Run from repository root:

    python custom/training/train_cotsfa_original_etth1_sanity.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data_provider.data_factory import data_provider
from custom.models import CoTSFATimesNet
from custom.evaluation.evaluate_corrupted_etth1 import build_args
from custom.cotsfa_original import (
    CoTSFAAugmentConfig,
    generate_cotsfa_augmentations,
    CoTSFAOriginalAlignmentLoss,
)


def make_decoder_input(
    batch_y: torch.Tensor,
    label_len: int,
    pred_len: int,
) -> torch.Tensor:
    """
    Create decoder input used by Time-Series-Library forecasting models.

    batch_y shape:
        (B, label_len + pred_len, C)

    output shape:
        (B, label_len + pred_len, C)
    """
    dec_zeros = torch.zeros_like(batch_y[:, -pred_len:, :])
    dec_inp = torch.cat([batch_y[:, :label_len, :], dec_zeros], dim=1)
    return dec_inp.float()


def forward_augmented_branches(
    model: CoTSFATimesNet,
    batch_x_aug: torch.Tensor,
    batch_y_aug: torch.Tensor,
    batch_x_mark: torch.Tensor,
    batch_y_mark: torch.Tensor,
    label_len: int,
    pred_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward A augmented branches.

    Parameters
    ----------
    batch_x_aug:
        Tensor with shape (A, B, seq_len, C)

    batch_y_aug:
        Tensor with shape (A, B, label_len + pred_len, C)

    Returns
    -------
    y_hat_aug_stack:
        Tensor with shape (A, B, pred_len, C)

        This is returned for shape/debug checks.
        It is not used in the original Co-TSFA forecast loss.

    z_aug_stack:
        Tensor with shape (A, B, pred_len, d_model)

        This is used in the alignment loss.
    """
    aug_count = batch_x_aug.shape[0]

    y_hat_aug_list = []
    z_aug_list = []

    for aug_idx in range(aug_count):
        dec_inp_aug = make_decoder_input(
            batch_y=batch_y_aug[aug_idx],
            label_len=label_len,
            pred_len=pred_len,
        )

        y_hat_aug, z_aug = model.forward_with_latent(
            x_enc=batch_x_aug[aug_idx],
            x_mark_enc=batch_x_mark,
            x_dec=dec_inp_aug,
            x_mark_dec=batch_y_mark,
            latent_slice="pred",
        )

        y_hat_aug_list.append(y_hat_aug)
        z_aug_list.append(z_aug)

    y_hat_aug_stack = torch.stack(y_hat_aug_list, dim=0)
    z_aug_stack = torch.stack(z_aug_list, dim=0)

    return y_hat_aug_stack, z_aug_stack


def run_one_mode(
    mode: str,
    model: CoTSFATimesNet,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    batch_x_mark: torch.Tensor,
    batch_y_mark: torch.Tensor,
    args,
    forecast_criterion,
    align_criterion,
    optimizer,
    device: torch.device,
    config: CoTSFAAugmentConfig,
    seed: int,
    lambda_align: float = 0.1,
) -> Dict[str, float]:
    """
    Run one training sanity step for either input_only or input_output mode.

    Important:
        Original Co-TSFA uses clean forecast loss only.

        Augmented samples are used for the alignment loss, not for an
        additional augmented forecast loss.
    """
    model.train()

    # ------------------------------------------------------------
    # Generate A=5 original Co-TSFA augmentations
    # ------------------------------------------------------------
    x_np = batch_x.float().numpy()
    y_np = batch_y.float().numpy()

    x_aug_np, y_aug_np, metadata_list = generate_cotsfa_augmentations(
        x=x_np,
        y=y_np,
        mode=mode,
        config=config,
        seed=seed,
    )

    # Convert to tensors
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    batch_x_aug = torch.from_numpy(x_aug_np).float().to(device)
    batch_y_aug = torch.from_numpy(y_aug_np).float().to(device)

    # ------------------------------------------------------------
    # Clean forward
    # ------------------------------------------------------------
    dec_inp = make_decoder_input(
        batch_y=batch_y,
        label_len=args.label_len,
        pred_len=args.pred_len,
    )

    y_hat, z = model.forward_with_latent(
        x_enc=batch_x,
        x_mark_enc=batch_x_mark,
        x_dec=dec_inp,
        x_mark_dec=batch_y_mark,
        latent_slice="pred",
    )

    # ------------------------------------------------------------
    # Augmented forward, A branches
    # ------------------------------------------------------------
    # y_hat_aug is kept only for sanity/shape checking.
    # Original Co-TSFA forecast loss is computed only on the clean branch.
    y_hat_aug, z_aug = forward_augmented_branches(
        model=model,
        batch_x_aug=batch_x_aug,
        batch_y_aug=batch_y_aug,
        batch_x_mark=batch_x_mark,
        batch_y_mark=batch_y_mark,
        label_len=args.label_len,
        pred_len=args.pred_len,
    )

    # ------------------------------------------------------------
    # Target tensors
    # ------------------------------------------------------------
    y_true = batch_y[:, -args.pred_len:, :]
    y_aug_true = batch_y_aug[:, :, -args.pred_len:, :]

    # Shapes:
    # y_hat:       (B, T, C)
    # y_hat_aug:   (A, B, T, C)  # debug only
    # z:           (B, T, D)
    # z_aug:       (A, B, T, D)
    # y_true:      (B, T, C)
    # y_aug_true:  (A, B, T, C)

    # ------------------------------------------------------------
    # Forecast loss: clean branch only
    # ------------------------------------------------------------
    loss_forecast = forecast_criterion(y_hat, y_true)

    # ------------------------------------------------------------
    # Original Co-TSFA alignment loss
    # ------------------------------------------------------------
    details = align_criterion(
        z=z,
        z_aug=z_aug,
        y=y_true,
        y_aug=y_aug_true,
        return_details=True,
    )

    loss_align = details.loss_align

    # ------------------------------------------------------------
    # Total loss
    # ------------------------------------------------------------
    loss_total = loss_forecast + lambda_align * loss_align

    # ------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------
    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()

    # ------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------
    x_changed = float(np.mean(np.abs(x_aug_np - x_np)))
    y_changed = float(np.mean(np.abs(y_aug_np - y_np)))

    curve_valid_rate = float(np.mean([m["curve_valid"] for m in metadata_list]))
    raw_curve_max_mean = float(np.mean([m["raw_curve_max"] for m in metadata_list]))
    applied_curve_max_mean = float(np.mean([m["applied_curve_max"] for m in metadata_list]))

    result = {
        "mode": mode,
        "loss_total": float(loss_total.item()),
        "loss_forecast": float(loss_forecast.item()),
        "loss_align": float(loss_align.item()),
        "latent_score_mean": float(details.latent_score_mean.item()),
        "target_score_mean": float(details.target_score_mean.item()),
        "latent_score_std": float(details.latent_score_std.item()),
        "target_score_std": float(details.target_score_std.item()),
        "x_changed": x_changed,
        "y_changed": y_changed,
        "curve_valid_rate": curve_valid_rate,
        "raw_curve_max_mean": raw_curve_max_mean,
        "applied_curve_max_mean": applied_curve_max_mean,
    }

    print(f"\n===== Mode: {mode} =====")
    print("batch_x shape:", tuple(batch_x.shape))
    print("batch_y shape:", tuple(batch_y.shape))
    print("batch_x_aug shape:", tuple(batch_x_aug.shape))
    print("batch_y_aug shape:", tuple(batch_y_aug.shape))
    print("y_hat shape:", tuple(y_hat.shape))
    print("y_hat_aug shape:", tuple(y_hat_aug.shape), "(debug only; not used in forecast loss)")
    print("z shape:", tuple(z.shape))
    print("z_aug shape:", tuple(z_aug.shape))
    print("y_true shape:", tuple(y_true.shape))
    print("y_aug_true shape:", tuple(y_aug_true.shape))
    print("x_changed:", result["x_changed"])
    print("y_changed:", result["y_changed"])
    print("curve_valid_rate:", result["curve_valid_rate"])
    print("raw_curve_max_mean:", result["raw_curve_max_mean"])
    print("applied_curve_max_mean:", result["applied_curve_max_mean"])
    print("loss_forecast_clean_only:", result["loss_forecast"])
    print("loss_align:", result["loss_align"])
    print("latent_score_mean:", result["latent_score_mean"])
    print("target_score_mean:", result["target_score_mean"])
    print("loss_total:", result["loss_total"])

    return result


def main() -> None:
    args = build_args()

    # Make paths robust on Windows.
    args.root_path = str(ROOT / "dataset" / "ETT-small") + "/"
    args.checkpoints = str(ROOT / "checkpoints") + "/"

    # CPU-safe sanity setting.
    args.batch_size = 8
    args.num_workers = 0
    args.use_gpu = False

    device = torch.device("cpu")

    print("Loading train data...")
    _, train_loader = data_provider(args, flag="train")

    print("Building CoTSFATimesNet...")
    model = CoTSFATimesNet(args).to(device)

    forecast_criterion = nn.MSELoss()

    align_criterion = CoTSFAOriginalAlignmentLoss(
        temperature=1.0,
        include_original_negatives=True,
        include_same_sample_augmentations=True,
        normalize=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Faithful raw reproduction setting.
    # If the alignment signal is too weak during full training,
    # strength=3.0 can be used as an ETTh1-calibrated variant.
    config = CoTSFAAugmentConfig(
        num_augmentations=5,
        normalize_curve=False,
        scaling_mode="raw",
        strength=1.0,
    )

    lambda_align = 0.1

    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(train_loader))

    print("\nSanity check config:")
    print("num_augmentations:", config.num_augmentations)
    print("scaling_mode:", config.scaling_mode)
    print("strength:", config.strength)
    print("lambda_align:", lambda_align)
    print("forecast loss:", "clean branch only")

    # Run input-only and input-output sanity checks.
    result_input_only = run_one_mode(
        mode="input_only",
        model=model,
        batch_x=batch_x,
        batch_y=batch_y,
        batch_x_mark=batch_x_mark,
        batch_y_mark=batch_y_mark,
        args=args,
        forecast_criterion=forecast_criterion,
        align_criterion=align_criterion,
        optimizer=optimizer,
        device=device,
        config=config,
        seed=args.seed,
        lambda_align=lambda_align,
    )

    result_input_output = run_one_mode(
        mode="input_output",
        model=model,
        batch_x=batch_x,
        batch_y=batch_y,
        batch_x_mark=batch_x_mark,
        batch_y_mark=batch_y_mark,
        args=args,
        forecast_criterion=forecast_criterion,
        align_criterion=align_criterion,
        optimizer=optimizer,
        device=device,
        config=config,
        seed=args.seed + 1,
        lambda_align=lambda_align,
    )

    print("\nOriginal Co-TSFA training sanity check finished successfully.")

    print("\nSummary:")
    print(result_input_only)
    print(result_input_output)


if __name__ == "__main__":
    main()