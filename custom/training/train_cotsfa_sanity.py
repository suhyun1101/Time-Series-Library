"""
Phase 3-3 sanity check for Co-TSFA-style training.

This script checks whether we can:

1. Load ETTh1 train data
2. Build CoTSFATimesNet
3. Create augmented input x_aug
4. Forward clean and augmented inputs
5. Compute forecast loss
6. Compute Co-TSFA-style alignment loss
7. Combine them into total loss
8. Run backward and optimizer step

Run from repository root:

    python custom/training/train_cotsfa_sanity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data_provider.data_factory import data_provider
from custom.models import CoTSFATimesNet
from custom.losses import CoTSFAAlignmentLoss
from custom.augmentations import gaussian_pointwise_noise, input_only_pair
from custom.evaluation.evaluate_corrupted_etth1 import build_args


def make_decoder_input(batch_y: torch.Tensor, label_len: int, pred_len: int) -> torch.Tensor:
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


def main() -> None:
    args = build_args()

    # Keep this sanity check small and CPU-safe.
    args.batch_size = 8
    args.num_workers = 0
    args.use_gpu = False

    device = torch.device("cpu")

    print("Loading train data...")
    _, train_loader = data_provider(args, flag="train")

    print("Building CoTSFATimesNet...")
    model = CoTSFATimesNet(args).to(device)
    model.train()

    forecast_criterion = nn.MSELoss()
    align_criterion = CoTSFAAlignmentLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    lambda_align = 0.1

    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(train_loader))

    batch_x = batch_x.float()
    batch_y = batch_y.float()
    batch_x_mark = batch_x_mark.float()
    batch_y_mark = batch_y_mark.float()

    # ------------------------------------------------------------
    # Create input-only augmented pair
    # ------------------------------------------------------------
    x_np = batch_x.numpy()
    y_np = batch_y.numpy()

    x_aug_np, y_aug_np, metadata = input_only_pair(
        x_np,
        y_np,
        gaussian_pointwise_noise,
        ratio=0.10,
        std_scale=0.70,
        seed=args.seed,
        independent_channels=False,
    )

    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    batch_x_aug = torch.from_numpy(x_aug_np).float().to(device)
    batch_y_aug = torch.from_numpy(y_aug_np).float().to(device)

    batch_x_mark = batch_x_mark.to(device)
    batch_y_mark = batch_y_mark.to(device)

    # Decoder inputs for clean and augmented branches.
    # For input-only augmentation, batch_y_aug is the same as batch_y.
    dec_inp = make_decoder_input(batch_y, args.label_len, args.pred_len).to(device)
    dec_inp_aug = make_decoder_input(batch_y_aug, args.label_len, args.pred_len).to(device)

    # ------------------------------------------------------------
    # Forward clean branch
    # ------------------------------------------------------------
    y_hat, z = model.forward_with_latent(
        x_enc=batch_x,
        x_mark_enc=batch_x_mark,
        x_dec=dec_inp,
        x_mark_dec=batch_y_mark,
        latent_slice="pred",
    )

    # ------------------------------------------------------------
    # Forward augmented branch
    # ------------------------------------------------------------
    y_hat_aug, z_aug = model.forward_with_latent(
        x_enc=batch_x_aug,
        x_mark_enc=batch_x_mark,
        x_dec=dec_inp_aug,
        x_mark_dec=batch_y_mark,
        latent_slice="pred",
    )

    # Target uses only pred_len part.
    y_true = batch_y[:, -args.pred_len:, :]
    y_aug_true = batch_y_aug[:, -args.pred_len:, :]

    loss_forecast_clean = forecast_criterion(y_hat, y_true)
    loss_forecast_aug = forecast_criterion(y_hat_aug, y_aug_true)
    loss_forecast = 0.5 * (loss_forecast_clean + loss_forecast_aug)

    loss_align, sim_latent, sim_output = align_criterion(
        z=z,
        z_aug=z_aug,
        y=y_true,
        y_aug=y_aug_true,
    )

    loss_total = loss_forecast + lambda_align * loss_align

    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()

    print("\nSanity check finished successfully.")
    print("augmentation:", metadata)
    print("batch_x shape:", tuple(batch_x.shape))
    print("batch_x_aug shape:", tuple(batch_x_aug.shape))
    print("y_hat shape:", tuple(y_hat.shape))
    print("y_hat_aug shape:", tuple(y_hat_aug.shape))
    print("z shape:", tuple(z.shape))
    print("z_aug shape:", tuple(z_aug.shape))
    print(f"loss_forecast_clean: {loss_forecast_clean.item():.6f}")
    print(f"loss_forecast_aug:   {loss_forecast_aug.item():.6f}")
    print(f"loss_forecast:       {loss_forecast.item():.6f}")
    print(f"loss_align:          {loss_align.item():.6f}")
    print(f"sim_latent:          {sim_latent.item():.6f}")
    print(f"sim_output:          {sim_output.item():.6f}")
    print(f"loss_total:          {loss_total.item():.6f}")


if __name__ == "__main__":
    main()