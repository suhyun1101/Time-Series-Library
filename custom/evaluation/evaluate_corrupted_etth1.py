"""
Phase 2-2: Corrupted evaluation for TimesNet on ETTh1.

This script loads the clean TimesNet checkpoint from Phase 1 and evaluates
the trained model on clean and corrupted test inputs.

Run from the repository root:

    python custom/evaluation/evaluate_corrupted_etth1.py

Output:

    experiment_logs/phase2_corrupted_eval_results.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from argparse import Namespace
from typing import Tuple, Dict, List, Any

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from custom.augmentations import (
    continuous_perturbation,
    gaussian_pointwise_noise,
    missing_observation,
    pointwise_spike,
    level_shift_input_output_pair,
    variance_burst_input_output_pair,
    input_only_pair,
)


# ---------------------------------------------------------------------
# Experiment arguments
# These match the Phase 1 clean baseline setting.
# ---------------------------------------------------------------------

def build_args() -> Namespace:
    return Namespace(
        # basic config
        task_name="long_term_forecast",
        is_training=0,
        model_id="ETTh1_96_96",
        model="TimesNet",
        data="ETTh1",

        # data loader
        root_path=str(ROOT / "dataset" / "ETT-small") + "/",
        data_path="ETTh1.csv",
        features="M",
        target="OT",
        freq="h",
        checkpoints=str(ROOT / "checkpoints") + "/",

        # forecasting task
        seq_len=96,
        label_len=48,
        pred_len=96,
        seasonal_patterns="Monthly",
        inverse=False,

        # model parameters
        top_k=5,
        num_kernels=6,
        enc_in=7,
        dec_in=7,
        c_out=7,
        d_model=16,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=32,
        moving_avg=25,
        factor=3,
        distil=True,
        dropout=0.1,
        embed="timeF",
        activation="gelu",
        output_attention=False,

        # compatibility args used by the TSLib codebase
        expand=2,
        d_conv=4,
        tv_dt=False,
        tv_B=False,
        tv_C=False,
        use_D=True,
        channel_independence=0,
        decomp_method="moving_avg",
        use_norm=1,
        down_sampling_layers=0,
        down_sampling_window=1,
        down_sampling_method=None,
        seg_len=48,

        # run parameters
        num_workers=0,
        itr=1,
        train_epochs=10,
        batch_size=32,
        patience=3,
        learning_rate=0.0001,
        des="Exp",
        loss="MSE",
        lradj="type1",
        use_amp=False,

        # force CPU because the current torch installation is not CUDA-enabled
        use_gpu=False,
        gpu=0,
        gpu_type="cuda",
        use_multi_gpu=False,
        devices="0,1,2,3",

        # de-stationary projector params
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,

        # augmentation compatibility args
        use_dtw=False,
        augmentation_ratio=0,
        seed=42,
        jitter=False,
        scaling=False,
        permutation=False,
        randompermutation=False,
        magwarp=False,
        timewarp=False,
        windowslice=False,
        windowwarp=False,
        rotation=False,
        spawner=False,
        dtwwarp=False,
        shapedtwwarp=False,
        wdba=False,
        discdtw=False,
        discsdtw=False,
        extra_tag="",

        # other model compatibility args
        patch_len=16,
        node_dim=10,
        gcn_depth=2,
        gcn_dropout=0.3,
        propalpha=0.3,
        conv_channel=32,
        skip_channel=32,
        individual=False,
        alpha=0.5,
        top_p=0.5,
        pos=0,
    )


# ---------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------

def find_checkpoint(args: Namespace) -> Path:
    """
    Find the checkpoint produced by the Phase 1 clean baseline run.
    """
    ckpt_root = Path(args.checkpoints)

    if not ckpt_root.exists():
        raise FileNotFoundError(
            f"Checkpoint folder does not exist: {ckpt_root}\n"
            "Run the Phase 1 clean baseline first."
        )

    candidates = []

    for path in ckpt_root.rglob("checkpoint.pth"):
        path_str = str(path)
        if args.model_id in path_str and args.model in path_str and args.data in path_str:
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "No matching checkpoint found.\n"
            f"Searched under: {ckpt_root}\n"
            f"Expected model_id: {args.model_id}, model: {args.model}, data: {args.data}"
        )

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_checkpoint_once(
    exp: Exp_Long_Term_Forecast,
    checkpoint_path: Path,
) -> None:
    """
    Load model weights once before evaluating all corruption conditions.
    """
    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location=exp.device,
            weights_only=True,
        )
    except TypeError:
        # For compatibility with older PyTorch versions.
        state_dict = torch.load(
            checkpoint_path,
            map_location=exp.device,
        )

    exp.model.load_state_dict(state_dict)
    exp.model.eval()


# ---------------------------------------------------------------------
# Corruption functions
# ---------------------------------------------------------------------

def apply_condition(
    x_np: np.ndarray,
    y_np: np.ndarray,
    condition: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Apply anomaly condition to x and possibly y.

    x_np shape:
        (B, seq_len, C)

    y_np shape:
        (B, label_len + pred_len, C)

    For input-only anomalies, y_np is unchanged.

    For input-output anomalies, y_np includes both the decoder label part
    and the prediction target part. We intentionally corrupt the full y_np
    sequence so that the decoder context is consistent with the shifted or
    bursty input sequence.

    Final evaluation metrics are still computed only on the pred_len part.
    """
    if condition == "clean":
        return x_np.copy(), y_np.copy(), {
            "type": "clean",
            "mode": "clean",
        }

    if condition == "continuous_input_only":
        return input_only_pair(
            x_np,
            y_np,
            continuous_perturbation,
            strength=0.08,
            n_knots=8,
            seed=seed,
        )

    if condition == "gaussian_pointwise_input_only":
        return input_only_pair(
            x_np,
            y_np,
            gaussian_pointwise_noise,
            ratio=0.10,
            std_scale=0.70,
            seed=seed,
            independent_channels=False,
        )

    if condition == "missing_observation_input_only":
        return input_only_pair(
            x_np,
            y_np,
            missing_observation,
            ratio=0.10,
            fill="mean",
            seed=seed,
            independent_channels=False,
        )

    if condition == "pointwise_spike_input_only":
        return input_only_pair(
            x_np,
            y_np,
            pointwise_spike,
            ratio=0.05,
            magnitude=4.0,
            seed=seed,
            independent_channels=False,
        )

    if condition == "level_shift_input_output":
        return level_shift_input_output_pair(
            x_np,
            y_np,
            start_ratio=0.60,
            shift_scale=1.0,
            seed=seed,
        )

    if condition == "variance_burst_input_output":
        return variance_burst_input_output_pair(
            x_np,
            y_np,
            start_ratio=0.45,
            length_ratio=0.25,
            variance_scale=2.0,
            seed=seed,
        )

    raise ValueError(f"Unknown condition: {condition}")


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_condition(
    exp: Exp_Long_Term_Forecast,
    args: Namespace,
    condition: str,
    base_seed: int = 42,
) -> Dict[str, float | str | int]:
    """
    Evaluate one corruption condition.

    The model checkpoint is loaded once in main().
    This function only applies the selected corruption condition and computes
    MSE/MAE on the pred_len forecasting target.
    """
    device = exp.device
    exp.model.eval()

    _, test_loader = exp._get_data(flag="test")

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

            # Decoder input:
            # first label_len observations are given,
            # future pred_len positions are initialized with zeros.
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat(
                [batch_y[:, :args.label_len, :], dec_inp],
                dim=1,
            ).float().to(device)

            outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            f_dim = -1 if args.features == "MS" else 0

            outputs = outputs[:, -args.pred_len:, f_dim:]
            true_y = batch_y[:, -args.pred_len:, f_dim:]

            preds.append(outputs.detach().cpu().numpy())
            trues.append(true_y.detach().cpu().numpy())

    preds_np = np.concatenate(preds, axis=0)
    trues_np = np.concatenate(trues, axis=0)

    mse = np.mean((preds_np - trues_np) ** 2)
    mae = np.mean(np.abs(preds_np - trues_np))

    return {
        "condition": condition,
        "model": args.model,
        "dataset": args.data,
        "seq_len": args.seq_len,
        "label_len": args.label_len,
        "pred_len": args.pred_len,
        "mse": float(mse),
        "mae": float(mae),
    }


def main() -> None:
    args = build_args()

    print("Building experiment...")
    exp = Exp_Long_Term_Forecast(args)

    checkpoint_path = find_checkpoint(args)
    print(f"Loaded checkpoint path: {checkpoint_path}")

    load_checkpoint_once(exp, checkpoint_path)
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
            exp=exp,
            args=args,
            condition=condition,
            base_seed=args.seed,
        )

        result["checkpoint"] = str(checkpoint_path)

        print(f"MSE: {result['mse']:.6f}, MAE: {result['mae']:.6f}")
        results.append(result)

    result_df = pd.DataFrame(results)

    out_path = ROOT / "experiment_logs" / "phase2_corrupted_eval_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)

    print("\nFinal results:")
    print(result_df[["condition", "mse", "mae"]])
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()