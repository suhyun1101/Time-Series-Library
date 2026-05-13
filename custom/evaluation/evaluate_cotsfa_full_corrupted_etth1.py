"""
Phase 3-5B: Corrupted evaluation for full Co-TSFA-style TimesNet on ETTh1.

This script loads the full Co-TSFA-style checkpoint and evaluates it under
the same clean and corrupted test conditions used in Phase 2.

Run from repository root:

    python custom/evaluation/evaluate_cotsfa_full_corrupted_etth1.py

Outputs:

    experiment_logs/phase3_cotsfa_full_corrupted_eval_results.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data_provider.data_factory import data_provider
from custom.models import CoTSFATimesNet
from custom.evaluation.evaluate_corrupted_etth1 import build_args, apply_condition


def make_decoder_input(batch_y: torch.Tensor, label_len: int, pred_len: int) -> torch.Tensor:
    """
    Create decoder input used by Time-Series-Library forecasting models.
    """
    dec_zeros = torch.zeros_like(batch_y[:, -pred_len:, :])
    dec_inp = torch.cat([batch_y[:, :label_len, :], dec_zeros], dim=1)
    return dec_inp.float()


def find_cotsfa_checkpoint() -> Path:
    """
    Find the full Co-TSFA-style checkpoint.
    """
    checkpoint_path = (
        ROOT
        / "checkpoints_cotsfa"
        / "cotsfa_etth1_96_full"
        / "checkpoint.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Full Co-TSFA checkpoint not found: {checkpoint_path}\n"
            "Run custom/training/train_cotsfa_etth1_full.py first."
        )

    return checkpoint_path


def load_checkpoint_once(
    model: CoTSFATimesNet,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    """
    Load model weights once before evaluating all conditions.
    """
    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(
            checkpoint_path,
            map_location=device,
        )

    model.load_state_dict(state_dict)
    model.eval()


def evaluate_condition(
    model: CoTSFATimesNet,
    args,
    test_loader,
    condition: str,
    device: torch.device,
    base_seed: int = 42,
) -> Dict[str, float | str | int]:
    """
    Evaluate one corruption condition.
    """
    model.eval()

    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            x_np = batch_x.float().numpy()
            y_np = batch_y.float().numpy()

            x_aug_np, y_aug_np, _ = apply_condition(
                x_np=x_np,
                y_np=y_np,
                condition=condition,
                seed=base_seed + batch_idx,
            )

            batch_x = torch.from_numpy(x_aug_np).float().to(device)
            batch_y = torch.from_numpy(y_aug_np).float().to(device)
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

            preds.append(y_hat.detach().cpu().numpy())
            trues.append(y_true.detach().cpu().numpy())

    preds_np = np.concatenate(preds, axis=0)
    trues_np = np.concatenate(trues, axis=0)

    mse = np.mean((preds_np - trues_np) ** 2)
    mae = np.mean(np.abs(preds_np - trues_np))

    return {
        "condition": condition,
        "model": "CoTSFATimesNet",
        "dataset": args.data,
        "seq_len": args.seq_len,
        "label_len": args.label_len,
        "pred_len": args.pred_len,
        "mse": float(mse),
        "mae": float(mae),
    }


def main() -> None:
    args = build_args()

    # Make paths robust on Windows.
    args.root_path = str(ROOT / "dataset" / "ETT-small") + "/"
    args.checkpoints = str(ROOT / "checkpoints") + "/"

    args.batch_size = 32
    args.num_workers = 0
    args.use_gpu = False

    device = torch.device("cpu")

    print("Loading test data...")
    _, test_loader = data_provider(args, flag="test")

    print("Building CoTSFATimesNet...")
    model = CoTSFATimesNet(args).to(device)

    checkpoint_path = find_cotsfa_checkpoint()
    print(f"Loaded full Co-TSFA checkpoint path: {checkpoint_path}")

    load_checkpoint_once(model, checkpoint_path, device)
    print("Checkpoint loaded successfully.")

    conditions = [
        "clean",
        "continuous_input_only",
        "gaussian_pointwise_input_only",
        "missing_observation_input_only",
        "pointwise_spike_input_only",
        "level_shift_input_output",
        "variance_burst_input_output",
    ]

    results = []

    for condition in conditions:
        print(f"\nEvaluating condition: {condition}")

        result = evaluate_condition(
            model=model,
            args=args,
            test_loader=test_loader,
            condition=condition,
            device=device,
            base_seed=args.seed,
        )

        result["checkpoint"] = str(checkpoint_path)
        result["training_setting"] = "full_10epochs_best_epoch9_lambda0.1"

        print(f"MSE: {result['mse']:.6f}, MAE: {result['mae']:.6f}")
        results.append(result)

    result_df = pd.DataFrame(results)

    out_path = ROOT / "experiment_logs" / "phase3_cotsfa_full_corrupted_eval_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)

    print("\nFinal results:")
    print(result_df[["condition", "mse", "mae"]])
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()