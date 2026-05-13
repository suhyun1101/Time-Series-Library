"""
Phase 3-5: Full Co-TSFA-style training run on ETTh1.

This script trains CoTSFATimesNet with:

    L_total = L_forecast + lambda_align * L_align

Unlike the pilot script, this version uses the full train loader for each epoch
and applies early stopping based on clean validation loss.

Run from repository root:

    python custom/training/train_cotsfa_etth1_full.py

Outputs:

    checkpoints_cotsfa/cotsfa_etth1_96_full/checkpoint.pth
    experiment_logs/phase3_cotsfa_full_training_log.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data_provider.data_factory import data_provider
from custom.models import CoTSFATimesNet
from custom.losses import CoTSFAAlignmentLoss
from custom.evaluation.evaluate_corrupted_etth1 import build_args, apply_condition


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


def evaluate_clean(
    model: CoTSFATimesNet,
    data_loader,
    args,
    criterion,
    device: torch.device,
) -> float:
    """
    Evaluate clean validation loss using the full validation loader.
    """
    model.eval()
    losses: List[float] = []

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = make_decoder_input(
                batch_y=batch_y,
                label_len=args.label_len,
                pred_len=args.pred_len,
            ).to(device)

            y_hat, _ = model.forward_with_latent(
                x_enc=batch_x,
                x_mark_enc=batch_x_mark,
                x_dec=dec_inp,
                x_mark_dec=batch_y_mark,
                latent_slice="pred",
            )

            y_true = batch_y[:, -args.pred_len:, :]
            loss = criterion(y_hat, y_true)
            losses.append(loss.item())

    model.train()
    return float(np.mean(losses))


def train_one_epoch(
    model: CoTSFATimesNet,
    train_loader,
    args,
    forecast_criterion,
    align_criterion,
    optimizer,
    device: torch.device,
    rng: np.random.Generator,
    epoch: int,
    lambda_align: float,
) -> Dict[str, float]:
    """
    Train one full epoch with Co-TSFA-style objective.
    """
    model.train()

    train_conditions = [
        "continuous_input_only",
        "gaussian_pointwise_input_only",
        "missing_observation_input_only",
        "pointwise_spike_input_only",
        "level_shift_input_output",
        "variance_burst_input_output",
    ]

    condition_counts = {condition: 0 for condition in train_conditions}

    losses_total: List[float] = []
    losses_forecast: List[float] = []
    losses_align: List[float] = []
    sims_latent: List[float] = []
    sims_output: List[float] = []

    start_time = time.time()

    for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        condition = str(rng.choice(train_conditions))
        condition_counts[condition] += 1

        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        x_np = batch_x.numpy()
        y_np = batch_y.numpy()

        x_aug_np, y_aug_np, _ = apply_condition(
            x_np=x_np,
            y_np=y_np,
            condition=condition,
            seed=args.seed + epoch * 10000 + batch_idx,
        )

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_x_aug = torch.from_numpy(x_aug_np).float().to(device)
        batch_y_aug = torch.from_numpy(y_aug_np).float().to(device)

        batch_x_mark = batch_x_mark.to(device)
        batch_y_mark = batch_y_mark.to(device)

        dec_inp = make_decoder_input(
            batch_y=batch_y,
            label_len=args.label_len,
            pred_len=args.pred_len,
        ).to(device)

        dec_inp_aug = make_decoder_input(
            batch_y=batch_y_aug,
            label_len=args.label_len,
            pred_len=args.pred_len,
        ).to(device)

        y_hat, z = model.forward_with_latent(
            x_enc=batch_x,
            x_mark_enc=batch_x_mark,
            x_dec=dec_inp,
            x_mark_dec=batch_y_mark,
            latent_slice="pred",
        )

        y_hat_aug, z_aug = model.forward_with_latent(
            x_enc=batch_x_aug,
            x_mark_enc=batch_x_mark,
            x_dec=dec_inp_aug,
            x_mark_dec=batch_y_mark,
            latent_slice="pred",
        )

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

        losses_total.append(loss_total.item())
        losses_forecast.append(loss_forecast.item())
        losses_align.append(loss_align.item())
        sims_latent.append(sim_latent.item())
        sims_output.append(sim_output.item())

        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(
                f"epoch {epoch} | batch {batch_idx + 1} | "
                f"condition {condition} | "
                f"loss_total {np.mean(losses_total):.6f} | "
                f"loss_forecast {np.mean(losses_forecast):.6f} | "
                f"loss_align {np.mean(losses_align):.6f} | "
                f"elapsed {elapsed:.1f}s"
            )

    condition_counts_str = ";".join(
        [f"{condition}:{count}" for condition, count in condition_counts.items()]
    )

    return {
        "train_loss_total": float(np.mean(losses_total)),
        "train_loss_forecast": float(np.mean(losses_forecast)),
        "train_loss_align": float(np.mean(losses_align)),
        "train_sim_latent": float(np.mean(sims_latent)),
        "train_sim_output": float(np.mean(sims_output)),
        "condition_counts": condition_counts_str,
    }


def main() -> None:
    args = build_args()

    # Make paths robust on Windows.
    args.root_path = str(ROOT / "dataset" / "ETT-small") + "/"
    args.checkpoints = str(ROOT / "checkpoints") + "/"

    # Full training setting.
    args.batch_size = 32
    args.num_workers = 0
    args.use_gpu = False

    TRAIN_EPOCHS = 10
    PATIENCE = 3
    LAMBDA_ALIGN = 0.1

    device = torch.device("cpu")
    rng = np.random.default_rng(args.seed)

    print("Loading data...")
    _, train_loader = data_provider(args, flag="train")
    _, vali_loader = data_provider(args, flag="val")

    print("Building CoTSFATimesNet...")
    model = CoTSFATimesNet(args).to(device)

    forecast_criterion = nn.MSELoss()
    align_criterion = CoTSFAAlignmentLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    save_dir = ROOT / "checkpoints_cotsfa" / "cotsfa_etth1_96_full"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "checkpoint.pth"

    logs = []
    best_val_loss = float("inf")
    early_stop_counter = 0

    print("\nStarting full Co-TSFA-style training...")
    print(f"TRAIN_EPOCHS: {TRAIN_EPOCHS}")
    print(f"PATIENCE: {PATIENCE}")
    print(f"LAMBDA_ALIGN: {LAMBDA_ALIGN}")
    print(f"Batch size: {args.batch_size}")
    print("Device: CPU")

    for epoch in range(1, TRAIN_EPOCHS + 1):
        epoch_start = time.time()

        train_stats = train_one_epoch(
            model=model,
            train_loader=train_loader,
            args=args,
            forecast_criterion=forecast_criterion,
            align_criterion=align_criterion,
            optimizer=optimizer,
            device=device,
            rng=rng,
            epoch=epoch,
            lambda_align=LAMBDA_ALIGN,
        )

        val_loss = evaluate_clean(
            model=model,
            data_loader=vali_loader,
            args=args,
            criterion=forecast_criterion,
            device=device,
        )

        epoch_time = time.time() - epoch_start

        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Validation improved. Saved best checkpoint to: {save_path}")
        else:
            early_stop_counter += 1
            print(f"No validation improvement. EarlyStopping counter: {early_stop_counter}/{PATIENCE}")

        row = {
            "epoch": epoch,
            **train_stats,
            "val_loss_clean": val_loss,
            "best_val_loss": best_val_loss,
            "lambda_align": LAMBDA_ALIGN,
            "batch_size": args.batch_size,
            "train_epochs_max": TRAIN_EPOCHS,
            "patience": PATIENCE,
            "early_stop_counter": early_stop_counter,
            "improved": improved,
            "epoch_time_sec": epoch_time,
        }
        logs.append(row)

        print(
            f"\nEpoch {epoch} summary | "
            f"train_total {row['train_loss_total']:.6f} | "
            f"train_forecast {row['train_loss_forecast']:.6f} | "
            f"train_align {row['train_loss_align']:.6f} | "
            f"val_clean {val_loss:.6f} | "
            f"best_val {best_val_loss:.6f} | "
            f"time {epoch_time:.1f}s"
        )

        if early_stop_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    log_df = pd.DataFrame(logs)

    out_log = ROOT / "experiment_logs" / "phase3_cotsfa_full_training_log.csv"
    out_log.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(out_log, index=False)

    print("\nFull training finished.")
    print(log_df)
    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Training log saved to: {out_log}")
    print(f"Best checkpoint saved to: {save_path}")


if __name__ == "__main__":
    main()