"""
Original Co-TSFA anomaly augmentation module.

This module is for Phase 3B: faithful Co-TSFA reproduction.

It implements the paper-style continuous anomaly curve:

    a(t) = A * t * exp(-B * t^C) / Z

where:
    B = 0.385
    Z = 90409
    A ~ N(74120, 20000^2)
    C ~ N(0.806, 0.2^2)

Important:
    The default setting uses the raw paper-scale curve:
        scaling_mode = "raw"
        normalize_curve = False

    Since the curve parameters were estimated from the original paper's data
    scale, scaling_mode="std" is provided only as a diagnostic option for
    checking scale compatibility on other datasets such as ETTh1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np


Mode = Literal["input_only", "input_output"]
ScalingMode = Literal["raw", "std"]


@dataclass
class CoTSFAAugmentConfig:
    """
    Configuration for original Co-TSFA anomaly generation.

    Parameters
    ----------
    num_augmentations:
        Number of augmented samples per original sequence.
        The paper uses A = 5 augmentations per sequence.

    b:
        Fixed B parameter in the anomaly curve.

    z:
        Fixed Z normalization parameter in the anomaly curve.

    amplitude_mean, amplitude_std:
        Gaussian sampling parameters for A.

    exponent_mean, exponent_std:
        Gaussian sampling parameters for C.

    normalize_curve:
        Whether to normalize the raw curve to max absolute value 1.

        For faithful reproduction, this should be False because the paper's
        curve constraints are defined on the raw curve scale.

    scaling_mode:
        - "raw": apply the raw paper-scale curve directly.
        - "std": multiply the raw curve by channel-wise standard deviation.

        The default is "raw" for reproduction.
        "std" is only for diagnostic checks when dataset scale differs.

    strength:
        Additional multiplier applied after curve generation.
        For faithful reproduction, keep this as 1.0.

    max_retry:
        Maximum number of resampling attempts for valid A and C.

    input_only_start_min_ratio, input_only_start_max_ratio:
        Input-only anomaly starts in early input window.
        Paper setting: t in [0, 0.5L].

    input_output_start_min_ratio, input_output_start_max_ratio:
        Input-output anomaly starts near the end of input window.
        Paper setting: t in [0.85L, 0.95L].
    """

    num_augmentations: int = 5

    # Paper fixed parameters
    b: float = 0.385
    z: float = 90409.0

    # Paper sampling parameters
    amplitude_mean: float = 74120.0
    amplitude_std: float = 20000.0
    exponent_mean: float = 0.806
    exponent_std: float = 0.2

    # Faithful reproduction defaults
    normalize_curve: bool = False
    scaling_mode: ScalingMode = "raw"
    strength: float = 1.0

    # Curve validity constraints
    max_curve_value: float = 2.0
    t30_max_value: float = 0.4
    t30_idx: int = 29

    max_retry: int = 100

    # Timing
    input_only_start_min_ratio: float = 0.0
    input_only_start_max_ratio: float = 0.5
    input_output_start_min_ratio: float = 0.85
    input_output_start_max_ratio: float = 0.95


def _as_3d(arr: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Convert array to 3D shape (B, T, C).

    If input is 2D (T, C), add batch dimension.
    """
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 2:
        return arr[None, :, :], True

    if arr.ndim == 3:
        return arr, False

    raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")


def _restore_dim(arr: np.ndarray, was_2d: bool) -> np.ndarray:
    """
    Remove batch dimension if original input was 2D.
    """
    if was_2d:
        return arr[0]
    return arr


def _channel_std(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute channel-wise standard deviation.

    Input:
        x: (B, T, C)

    Output:
        std: (B, 1, C)
    """
    std = np.std(x, axis=1, keepdims=True)
    return np.maximum(std, eps)


def cotsfa_anomaly_curve(
    length: int,
    amplitude: float,
    exponent: float,
    b: float = 0.385,
    z: float = 90409.0,
    normalize: bool = False,
) -> np.ndarray:
    """
    Generate the Co-TSFA paper-style anomaly curve.

    Formula:
        a(t) = A * t * exp(-B * t^C) / Z

    Parameters
    ----------
    length:
        Length of anomaly curve.

    amplitude:
        A parameter in the paper formula.

    exponent:
        C parameter in the paper formula.

    b:
        Fixed B parameter.

    z:
        Fixed Z normalization parameter.

    normalize:
        If True, normalize curve to max absolute value 1.
        For faithful reproduction, use False.

    Returns
    -------
    curve:
        shape (length,)
    """
    if length <= 0:
        raise ValueError("length must be positive")

    if exponent <= 0:
        raise ValueError("exponent must be positive")

    t = np.arange(1, length + 1, dtype=np.float32)

    curve = amplitude * t * np.exp(-b * (t ** exponent)) / z

    if normalize:
        max_abs = np.max(np.abs(curve))
        if max_abs > 0:
            curve = curve / max_abs

    return curve.astype(np.float32)


def sample_curve_parameters(
    rng: np.random.Generator,
    config: CoTSFAAugmentConfig,
) -> Tuple[float, float]:
    """
    Sample curve parameters A and C.

    A ~ N(74120, 20000^2)
    C ~ N(0.806, 0.2^2)

    Invalid values are handled by the validation/resampling step.
    """
    amplitude = float(rng.normal(config.amplitude_mean, config.amplitude_std))
    exponent = float(rng.normal(config.exponent_mean, config.exponent_std))

    return amplitude, exponent


def _validate_curve(
    curve: np.ndarray,
    config: CoTSFAAugmentConfig,
) -> bool:
    """
    Validate anomaly curve according to paper-style constraints.

    Conditions:
        - curve values should be non-negative
        - max value should be smaller than 2.0
        - value at t=30 should be smaller than 0.4
    """
    if np.any(~np.isfinite(curve)):
        return False

    if np.any(curve < 0):
        return False

    if np.max(curve) >= config.max_curve_value:
        return False

    if config.t30_idx < len(curve):
        if curve[config.t30_idx] >= config.t30_max_value:
            return False

    return True


def sample_valid_curve(
    length: int,
    rng: np.random.Generator,
    config: CoTSFAAugmentConfig,
) -> Tuple[np.ndarray, float, float, bool, int]:
    """
    Sample A and C until the generated curve satisfies paper constraints.

    Returns
    -------
    curve:
        Valid or last sampled curve.

    amplitude:
        Sampled A.

    exponent:
        Sampled C.

    valid:
        Whether the returned curve passed validation.

    num_trials:
        Number of sampling attempts.
    """
    last_curve = None
    last_amplitude = None
    last_exponent = None

    for trial in range(1, config.max_retry + 1):
        amplitude, exponent = sample_curve_parameters(rng, config)

        # A and C should be positive for a valid non-negative curve.
        if amplitude <= 0 or exponent <= 0:
            continue

        curve = cotsfa_anomaly_curve(
            length=length,
            amplitude=amplitude,
            exponent=exponent,
            b=config.b,
            z=config.z,
            normalize=config.normalize_curve,
        )

        last_curve = curve
        last_amplitude = amplitude
        last_exponent = exponent

        if _validate_curve(curve, config):
            return curve, amplitude, exponent, True, trial

    if last_curve is None:
        raise RuntimeError(
            "Failed to sample any numerically valid curve. "
            "Check amplitude/exponent sampling parameters."
        )

    return last_curve, last_amplitude, last_exponent, False, config.max_retry


def _sample_start_index(
    rng: np.random.Generator,
    seq_len: int,
    mode: Mode,
    config: CoTSFAAugmentConfig,
) -> int:
    """
    Sample anomaly start index according to input-only or input-output mode.
    """
    if mode == "input_only":
        low = int(np.floor(config.input_only_start_min_ratio * seq_len))
        high = int(np.floor(config.input_only_start_max_ratio * seq_len))
    elif mode == "input_output":
        low = int(np.floor(config.input_output_start_min_ratio * seq_len))
        high = int(np.floor(config.input_output_start_max_ratio * seq_len))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    low = max(0, min(low, seq_len - 1))
    high = max(low + 1, min(high, seq_len))

    return int(rng.integers(low, high))


def _expand_curve_to_batch_channels(
    base_curve: np.ndarray,
    x_arr: np.ndarray,
    config: CoTSFAAugmentConfig,
) -> np.ndarray:
    """
    Expand 1D curve to shape (B, curve_len, C).

    scaling_mode="raw":
        Apply the raw paper-scale curve directly to all channels.

    scaling_mode="std":
        Multiply the raw curve by channel-wise std.
        This is diagnostic only, not the default reproduction setting.
    """
    bsz, _, channels = x_arr.shape

    curve = base_curve[None, :, None]
    curve = np.repeat(curve, repeats=bsz, axis=0)
    curve = np.repeat(curve, repeats=channels, axis=2)

    if config.scaling_mode == "raw":
        curve = curve * config.strength
    elif config.scaling_mode == "std":
        std = _channel_std(x_arr)
        curve = curve * std * config.strength
    else:
        raise ValueError(f"Unknown scaling_mode: {config.scaling_mode}")

    return curve.astype(np.float32)


def apply_cotsfa_curve_once(
    x: np.ndarray,
    y: np.ndarray,
    mode: Mode,
    config: CoTSFAAugmentConfig | None = None,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Apply one original Co-TSFA anomaly curve to x and possibly y.

    Parameters
    ----------
    x:
        Input sequence, shape (T_x, C) or (B, T_x, C)

    y:
        Forecast target sequence, shape (T_y, C) or (B, T_y, C)

    mode:
        "input_only":
            anomaly starts early and remains inside input window.
            y is unchanged.

        "input_output":
            anomaly starts near the end of input and extends into y.
            y may be changed.

    Returns
    -------
    x_aug:
        Augmented input sequence.

    y_aug:
        Augmented target sequence.

    metadata:
        Dictionary describing sampled parameters and curve properties.
    """
    if config is None:
        config = CoTSFAAugmentConfig()

    rng = np.random.default_rng(seed)

    x_arr, x_was_2d = _as_3d(x)
    y_arr, y_was_2d = _as_3d(y)

    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x and y must have the same batch size")

    if x_arr.shape[2] != y_arr.shape[2]:
        raise ValueError("x and y must have the same number of channels")

    _, seq_len, _ = x_arr.shape
    pred_len = y_arr.shape[1]
    total_len = seq_len + pred_len

    x_aug = x_arr.copy()
    y_aug = y_arr.copy()

    start_idx = _sample_start_index(
        rng=rng,
        seq_len=seq_len,
        mode=mode,
        config=config,
    )

    if mode == "input_only":
        # Ends within the input window.
        curve_len = seq_len - start_idx
    else:
        # Extends from near end of input into the forecasting horizon.
        curve_len = total_len - start_idx

    curve_len = max(1, curve_len)

    base_curve, amplitude, exponent, curve_valid, num_trials = sample_valid_curve(
        length=curve_len,
        rng=rng,
        config=config,
    )

    curve = _expand_curve_to_batch_channels(
        base_curve=base_curve,
        x_arr=x_arr,
        config=config,
    )

    combined = np.concatenate([x_aug, y_aug], axis=1)

    end_idx = start_idx + curve_len
    combined[:, start_idx:end_idx, :] += curve[:, : end_idx - start_idx, :]

    if mode == "input_only":
        x_aug = combined[:, :seq_len, :]
        y_aug = y_arr.copy()
    else:
        x_aug = combined[:, :seq_len, :]
        y_aug = combined[:, seq_len:, :]

    curve_t30 = None
    if config.t30_idx < len(base_curve):
        curve_t30 = float(base_curve[config.t30_idx])

    applied_curve_max = float(np.max(curve))
    applied_curve_min = float(np.min(curve))

    metadata = {
        "type": "cotsfa_original_curve",
        "mode": mode,
        "start_idx": start_idx,
        "curve_len": curve_len,
        "amplitude": float(amplitude),
        "exponent": float(exponent),
        "b": config.b,
        "z": config.z,
        "strength": config.strength,
        "normalize_curve": config.normalize_curve,
        "scaling_mode": config.scaling_mode,
        "curve_valid": bool(curve_valid),
        "num_trials": int(num_trials),
        "raw_curve_max": float(np.max(base_curve)),
        "raw_curve_min": float(np.min(base_curve)),
        "raw_curve_t30": curve_t30,
        "applied_curve_max": applied_curve_max,
        "applied_curve_min": applied_curve_min,
    }

    return (
        _restore_dim(x_aug, x_was_2d),
        _restore_dim(y_aug, y_was_2d),
        metadata,
    )


def generate_cotsfa_augmentations(
    x: np.ndarray,
    y: np.ndarray,
    mode: Mode,
    config: CoTSFAAugmentConfig | None = None,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, list[Dict]]:
    """
    Generate multiple Co-TSFA augmentations for each sequence.

    The paper uses A = 5 augmentations per sequence.

    Returns
    -------
    x_augs:
        shape (A, B, T_x, C) for 3D input
        or (A, T_x, C) for 2D input

    y_augs:
        shape (A, B, T_y, C) for 3D input
        or (A, T_y, C) for 2D input

    metadata_list:
        list of metadata dictionaries
    """
    if config is None:
        config = CoTSFAAugmentConfig()

    rng = np.random.default_rng(seed)

    x_augs = []
    y_augs = []
    metadata_list = []

    for aug_idx in range(config.num_augmentations):
        aug_seed = int(rng.integers(0, 2**31 - 1))

        x_aug, y_aug, metadata = apply_cotsfa_curve_once(
            x=x,
            y=y,
            mode=mode,
            config=config,
            seed=aug_seed,
        )

        metadata["augmentation_index"] = aug_idx

        x_augs.append(x_aug)
        y_augs.append(y_aug)
        metadata_list.append(metadata)

    return np.stack(x_augs, axis=0), np.stack(y_augs, axis=0), metadata_list