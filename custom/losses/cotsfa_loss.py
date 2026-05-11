"""
Co-TSFA-style alignment loss for Phase 3.

This module implements a lightweight version of latent-output alignment.

The idea:
    If an augmentation changes the forecast target strongly,
    the latent representation is allowed to change strongly.

    If an augmentation does not change the forecast target,
    the latent representation should remain similar.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def flatten_for_similarity(x: torch.Tensor) -> torch.Tensor:
    """
    Flatten all dimensions except batch dimension.

    Input examples:
        z:      (B, D)
        y:      (B, T, C)
        output: (B, T * C) or (B, D)
    """
    if x.ndim == 2:
        return x

    return x.reshape(x.shape[0], -1)


def cosine_similarity_batch(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute cosine similarity for each sample in a batch.

    Returns:
        sim: shape (B,)
    """
    a_flat = flatten_for_similarity(a)
    b_flat = flatten_for_similarity(b)

    return F.cosine_similarity(a_flat, b_flat, dim=1, eps=eps)


class CoTSFAAlignmentLoss(nn.Module):
    """
    Co-TSFA-style alignment loss.

    L_align = mean(|sim_latent - sim_output|)

    sim_latent:
        cosine similarity between z and z_aug

    sim_output:
        cosine similarity between y and y_aug

    For input-only anomaly:
        y_aug = y
        sim_output should be close to 1
        so z and z_aug should also stay similar.

    For input-output anomaly:
        y_aug differs from y
        sim_output becomes smaller
        so z and z_aug are allowed to differ.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        z: torch.Tensor,
        z_aug: torch.Tensor,
        y: torch.Tensor,
        y_aug: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            loss_align
            sim_latent_mean
            sim_output_mean
        """
        sim_latent = cosine_similarity_batch(z, z_aug)
        sim_output = cosine_similarity_batch(y, y_aug)

        loss_align = torch.mean(torch.abs(sim_latent - sim_output))

        return loss_align, sim_latent.mean(), sim_output.mean()