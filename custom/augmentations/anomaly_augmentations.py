"""
Anomaly augmentation functions for Phase 2.

Input shape:
    - 2D: (T, C)
    - 3D: (B, T, C)

T = time length
C = number of variables
B = batch size

These functions are designed first for visual checking and later integration
into the Time-Series-Library training pipeline.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any
import numpy as np


def _as_3d(x: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Convert input to 3D array.

    Returns
    -------
    arr : np.ndarray
        Shape (B, T, C)
    was_2d : bool
        True if original input was 2D.
    """
    arr = np.asarray(x, dtype=np.float32).copy()

    if arr.ndim == 2:
        return arr[None, :, :], True

    if arr.ndim == 3:
        return arr, False

    raise ValueError(f"Expected input shape (T, C) or (B, T, C), got {arr.shape}")


def _restore_dim(x: np.ndarray, was_2d: bool) -> np.ndarray:
    if was_2d:
        return x[0]
    return x


def _channel_std(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute per-sample, per-channel standard deviation.

    Input shape: (B, T, C)
    Output shape: (B, 1, C)
    """
    std = np.std(x, axis=1, keepdims=True)
    return np.maximum(std, eps)


def _channel_mean(x: np.ndarray) -> np.ndarray:
    """
    Compute per-sample, per-channel mean.

    Input shape: (B, T, C)
    Output shape: (B, 1, C)
    """
    return np.mean(x, axis=1, keepdims=True)


def continuous_perturbation(
    x: np.ndarray,
    strength: float = 0.05,
    n_knots: int = 8,
    seed: Optional[int] = None,
    return_metadata: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, Any]]:
    """
    Add smooth continuous perturbation to the whole input sequence.

    This is similar in spirit to small continuous perturbations used in
    anomaly-aware contrastive augmentation.

    Parameters
    ----------
    x : np.ndarray
        Input time series, shape (T, C) or (B, T, C).
    strength : float
        Perturbation size relative to per-channel standard deviation.
    n_knots : int
        Number of random knots used to create smooth perturbation.
    seed : int, optional
        Random seed.
    return_metadata : bool
        Whether to return metadata.

    Returns
    -------
    x_aug : np.ndarray
        Augmented input with the same shape as x.
    """
    rng = np.random.default_rng(seed)
    arr, was_2d = _as_3d(x)
    b, t, c = arr.shape

    std = _channel_std(arr)

    knot_x = np.linspace(0, t - 1, n_knots)
    full_x = np.arange(t)

    perturb = np.zeros_like(arr)

    for i in range(b):
        for j in range(c):
            knot_y = rng.normal(loc=0.0, scale=1.0, size=n_knots)
            smooth_noise = np.interp(full_x, knot_x, knot_y)
            perturb[i, :, j] = smooth_noise * std[i, 0, j] * strength

    out = arr + perturb
    out = _restore_dim(out, was_2d)

    metadata = {
        "type": "continuous_perturbation",
        "strength": strength,
        "n_knots": n_knots,
    }

    if return_metadata:
        return out, metadata
    return out


def gaussian_pointwise_noise(
    x: np.ndarray,
    ratio: float = 0.10,
    std_scale: float = 0.50,
    seed: Optional[int] = None,
    independent_channels: bool = False,
    return_metadata: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, Any]]:
    """
    Add Gaussian noise to randomly selected time points.

    Usually used as input-only anomaly because pointwise noise is often
    forecast-irrelevant.

    Parameters
    ----------
    ratio : float
        Proportion of time points to corrupt.
    std_scale : float
        Noise scale relative to per-channel standard deviation.
    independent_channels : bool
        If False, selected time points are corrupted across all channels.
        If True, each channel has independent corrupted points.
    """
    rng = np.random.default_rng(seed)
    arr, was_2d = _as_3d(x)
    b, t, c = arr.shape

    std = _channel_std(arr)
    out = arr.copy()

    if independent_channels:
        mask = rng.random(size=(b, t, c)) < ratio
    else:
        mask = rng.random(size=(b, t, 1)) < ratio
        mask = np.repeat(mask, c, axis=2)

    noise = rng.normal(loc=0.0, scale=1.0, size=(b, t, c)) * std * std_scale
    out[mask] = out[mask] + noise[mask]

    out = _restore_dim(out, was_2d)

    metadata = {
        "type": "gaussian_pointwise_noise",
        "ratio": ratio,
        "std_scale": std_scale,
        "independent_channels": independent_channels,
    }

    if return_metadata:
        return out, metadata
    return out


def missing_observation(
    x: np.ndarray,
    ratio: float = 0.10,
    fill: str = "zero",
    seed: Optional[int] = None,
    independent_channels: bool = False,
    return_metadata: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, Any]]:
    """
    Replace randomly selected observations with zero or channel mean.

    Usually used as input-only anomaly.

    Parameters
    ----------
    fill : {"zero", "mean"}
        Replacement value.
    """
    if fill not in {"zero", "mean"}:
        raise ValueError("fill must be either 'zero' or 'mean'.")

    rng = np.random.default_rng(seed)
    arr, was_2d = _as_3d(x)
    b, t, c = arr.shape

    out = arr.copy()

    if independent_channels:
        mask = rng.random(size=(b, t, c)) < ratio
    else:
        mask = rng.random(size=(b, t, 1)) < ratio
        mask = np.repeat(mask, c, axis=2)

    if fill == "zero":
        fill_values = np.zeros_like(out)
    else:
        fill_values = np.repeat(_channel_mean(arr), t, axis=1)

    out[mask] = fill_values[mask]
    out = _restore_dim(out, was_2d)

    metadata = {
        "type": "missing_observation",
        "ratio": ratio,
        "fill": fill,
        "independent_channels": independent_channels,
    }

    if return_metadata:
        return out, metadata
    return out


def pointwise_spike(
    x: np.ndarray,
    ratio: float = 0.05,
    magnitude: float = 3.0,
    seed: Optional[int] = None,
    independent_channels: bool = False,
    return_metadata: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, Any]]:
    """
    Add large spikes to randomly selected time points.

    Usually used as input-only anomaly.

    Parameters
    ----------
    ratio : float
        Proportion of time points to corrupt.
    magnitude : float
        Spike size relative to per-channel standard deviation.
    """
    rng = np.random.default_rng(seed)
    arr, was_2d = _as_3d(x)
    b, t, c = arr.shape

    std = _channel_std(arr)
    out = arr.copy()

    if independent_channels:
        mask = rng.random(size=(b, t, c)) < ratio
    else:
        mask = rng.random(size=(b, t, 1)) < ratio
        mask = np.repeat(mask, c, axis=2)

    signs = rng.choice([-1.0, 1.0], size=(b, t, c))
    spike_values = signs * std * magnitude

    out[mask] = out[mask] + spike_values[mask]
    out = _restore_dim(out, was_2d)

    metadata = {
        "type": "pointwise_spike",
        "ratio": ratio,
        "magnitude": magnitude,
        "independent_channels": independent_channels,
    }

    if return_metadata:
        return out, metadata
    return out


def level_shift(
    x: np.ndarray,
    start: Optional[int] = None,
    start_ratio: float = 0.60,
    shift_scale: float = 1.0,
    direction: Optional[float] = None,
    seed: Optional[int] = None,
    return_metadata: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply persistent mean shift after a certain time point.

    This can be forecast-relevant if the shift persists into the future.
    """
    rng = np.random.default_rng(seed)
    arr, was_2d = _as_3d(x)
    b, t, c = arr.shape

    std = _channel_std(arr)
    out = arr.copy()

    if start is None:
        start = int(t * start_ratio)

    if not (0 <= start < t):
        raise ValueError(f"start must be in [0, {t - 1}], got {start}")

    if direction is None:
        direction_arr = rng.choice([-1.0, 1.0], size=(b, 1, c))
    else:
        direction_arr = np.full((b, 1, c), float(direction), dtype=np.float32)

    shift = direction_arr * std * shift_scale

    out[:, start:, :] = out[:, start:, :] + shift
    out_restored = _restore_dim(out, was_2d)

    metadata = {
        "type": "level_shift",
        "start": start,
        "start_ratio": start_ratio,
        "shift_scale": shift_scale,
        "direction": direction,
        # Keep original 3D shape (B, 1, C) for later input-output augmentation.
        "shift": shift,
    }

    if return_metadata:
        return out_restored, metadata
    return out_restored


def variance_burst(
    x: np.ndarray,
    start: Optional[int] = None,
    length: Optional[int] = None,
    start_ratio: float = 0.45,
    length_ratio: float = 0.25,
    variance_scale: float = 2.0,
    seed: Optional[int] = None,
    return_metadata: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, Any]]:
    """
    Increase local variance in a selected interval.

    This creates a bursty high-variance segment.

    If start is None, the burst start point is randomly sampled.
    The random sampling is controlled by seed.
    """
    rng = np.random.default_rng(seed)

    arr, was_2d = _as_3d(x)
    b, t, c = arr.shape

    out = arr.copy()
    mean = _channel_mean(arr)

    if length is None:
        length = max(1, int(t * length_ratio))

    if length > t:
        raise ValueError(f"length must be <= sequence length {t}, got {length}")

    if start is None:
        max_start = max(0, t - length)
        start = int(rng.integers(0, max_start + 1))
    else:
        if not (0 <= start < t):
            raise ValueError(f"start must be in [0, {t - 1}], got {start}")

    end = min(t, start + length)

    segment = out[:, start:end, :]
    out[:, start:end, :] = mean + (segment - mean) * variance_scale

    out_restored = _restore_dim(out, was_2d)

    metadata = {
        "type": "variance_burst",
        "start": start,
        "end": end,
        "length": length,
        "start_ratio": start_ratio,
        "length_ratio": length_ratio,
        "variance_scale": variance_scale,
    }

    if return_metadata:
        return out_restored, metadata
    return out_restored


def input_only_pair(
    x: np.ndarray,
    y: np.ndarray,
    augmentation_fn,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Apply anomaly only to x and keep y unchanged.

    This corresponds to forecast-irrelevant perturbation.
    """
    x_aug, metadata = augmentation_fn(x, return_metadata=True, **kwargs)
    y_aug = np.asarray(y, dtype=np.float32).copy()

    metadata["mode"] = "input_only"
    return x_aug, y_aug, metadata


def level_shift_input_output_pair(
    x: np.ndarray,
    y: np.ndarray,
    start: Optional[int] = None,
    start_ratio: float = 0.60,
    shift_scale: float = 1.0,
    direction: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Apply level shift to x and extend the same shift to y.

    This corresponds to forecast-relevant persistent shift.
    """
    x_aug, metadata = level_shift(
        x=x,
        start=start,
        start_ratio=start_ratio,
        shift_scale=shift_scale,
        direction=direction,
        seed=seed,
        return_metadata=True,
    )

    y_arr, y_was_2d = _as_3d(y)

    # metadata["shift"] has shape (B, 1, C).
    # It is broadcast over the prediction length dimension of y.
    shift = metadata["shift"]

    y_aug = y_arr + shift
    y_aug = _restore_dim(y_aug, y_was_2d)

    metadata["mode"] = "input_output"
    return x_aug, y_aug, metadata


def variance_burst_input_output_pair(
    x: np.ndarray,
    y: np.ndarray,
    start: Optional[int] = None,
    length: Optional[int] = None,
    start_ratio: float = 0.45,
    length_ratio: float = 0.25,
    variance_scale: float = 2.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Apply variance burst to x and extend increased variance to y.

    This is a simple forecast-relevant version.
    """
    x_aug, metadata = variance_burst(
        x=x,
        start=start,
        length=length,
        start_ratio=start_ratio,
        length_ratio=length_ratio,
        variance_scale=variance_scale,
        seed=seed,
        return_metadata=True,
    )

    y_arr, y_was_2d = _as_3d(y)
    y_mean = _channel_mean(y_arr)

    y_aug = y_mean + (y_arr - y_mean) * variance_scale
    y_aug = _restore_dim(y_aug, y_was_2d)

    metadata["mode"] = "input_output"
    return x_aug, y_aug, metadata