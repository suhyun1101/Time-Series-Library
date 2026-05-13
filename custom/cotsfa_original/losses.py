"""
Original Co-TSFA alignment loss module.

This module implements the paper-style latent-output alignment loss using
InfoNCE-style softmax dot-product similarity with batch negatives.

This module is for Phase 3B: original Co-TSFA reproduction.

Expected inputs:

    z:      (B, T, D_z)
    z_aug:  (A, B, T, D_z) or (B, T, D_z)

    y:      (B, T, D_y)
    y_aug:  (A, B, T, D_y) or (B, T, D_y)

where:
    B = batch size
    T = time length, usually pred_len
    A = number of augmentations, paper default A=5

For each anchor sample i, time t, and augmentation a:

    positive:
        anchor_i,t dot augmented_a,i,t

    denominator:
        1. augmented batch terms from the same augmentation a:
           anchor_i,t dot augmented_a,j,t for j = 1,...,B

        2. original batch negatives:
           anchor_i,t dot anchor_j,t for j != i

        3. same-sample multiple augmentation terms:
           anchor_i,t dot augmented_k,i,t for k = 1,...,A

The score is:

    score = -log( exp(anchor_i · positive_i) / denominator )

Then:

    L_align = mean(|score_latent - score_target|)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_aug_dim(x: torch.Tensor, name: str) -> torch.Tensor:
    """
    Ensure augmented tensor has shape (A, B, T, D).

    Accepts:
        (B, T, D)       -> converted to (1, B, T, D)
        (A, B, T, D)    -> returned as-is
    """
    if x.ndim == 3:
        return x.unsqueeze(0)

    if x.ndim == 4:
        return x

    raise ValueError(
        f"{name} must have shape (B, T, D) or (A, B, T, D), "
        f"but got shape {tuple(x.shape)}"
    )


def _validate_shapes(
    anchor: torch.Tensor,
    augmented: torch.Tensor,
    name: str,
) -> None:
    """
    Validate anchor and augmented tensor shapes.
    """
    if anchor.ndim != 3:
        raise ValueError(
            f"{name} anchor must have shape (B, T, D), "
            f"but got shape {tuple(anchor.shape)}"
        )

    if augmented.ndim != 4:
        raise ValueError(
            f"{name} augmented must have shape (A, B, T, D), "
            f"but got shape {tuple(augmented.shape)}"
        )

    aug_count, b_aug, t_aug, d_aug = augmented.shape
    b, t, d = anchor.shape

    if aug_count <= 0:
        raise ValueError(f"{name} augmented count A must be positive")

    if b != b_aug:
        raise ValueError(
            f"{name} batch size mismatch: anchor B={b}, augmented B={b_aug}"
        )

    if t != t_aug:
        raise ValueError(
            f"{name} time length mismatch: anchor T={t}, augmented T={t_aug}"
        )

    if d != d_aug:
        raise ValueError(
            f"{name} feature dim mismatch: anchor D={d}, augmented D={d_aug}"
        )


def _maybe_l2_normalize(
    x: torch.Tensor,
    normalize: bool,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Optionally apply L2 normalization.

    The paper describes dot-product similarity, so the default is no
    normalization. This option is kept only for numerical diagnostics.
    """
    if not normalize:
        return x

    return F.normalize(x, p=2, dim=dim, eps=eps)


def paper_contrastive_score(
    anchor: torch.Tensor,
    augmented: torch.Tensor,
    temperature: float = 1.0,
    include_original_negatives: bool = True,
    include_same_sample_augmentations: bool = True,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Compute paper-style InfoNCE score for anchor and augmented views.

    This implementation computes a separate denominator for each augmentation
    view a.

    For each anchor sample i, time t, and augmentation a:

        positive:
            anchor_i,t dot augmented_a,i,t

        denominator:
            1. augmented batch terms from the same augmentation a:
               anchor_i,t dot augmented_a,j,t for j != i
               if include_same_sample_augmentations=True

               If include_same_sample_augmentations=False, j=i is kept here
               so that the positive term is still included in the denominator.

            2. original batch negatives:
               anchor_i,t dot anchor_j,t for j != i

            3. same-sample multiple augmentation terms:
               anchor_i,t dot augmented_k,i,t for k = 1,...,A

    Returns
    -------
    scores:
        InfoNCE-style scores with shape (A, B, T).
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    augmented = _ensure_aug_dim(augmented, name="augmented")
    _validate_shapes(anchor, augmented, name="contrastive score")

    anchor = _maybe_l2_normalize(anchor, normalize=normalize)
    augmented = _maybe_l2_normalize(augmented, normalize=normalize)

    # Input shapes:
    #   anchor:    (B, T, D)
    #   augmented: (A, B, T, D)
    #
    # Reordered shapes:
    #   anchor_t:    (T, B, D)
    #   augmented_t: (T, A, B, D)
    anchor_t = anchor.permute(1, 0, 2)
    augmented_t = augmented.permute(2, 0, 1, 3)

    t_len, batch_size, _ = anchor_t.shape
    aug_count = augmented_t.shape[1]

    # ------------------------------------------------------------
    # 1. Augmented batch terms for each augmentation a
    # ------------------------------------------------------------
    # logits_aug[t, i, a, j]
    #   = anchor_i,t dot augmented_a,j,t
    #
    # Shape:
    #   (T, B, A, B)
    logits_aug = torch.einsum(
        "tid,tajd->tiaj",
        anchor_t,
        augmented_t,
    ) / temperature

    # Positive logits:
    # pos_logits[t, i, a]
    #   = anchor_i,t dot augmented_a,i,t
    #
    # Shape:
    #   (T, B, A)
    pos_index = torch.arange(batch_size, device=anchor.device)
    pos_index = pos_index.view(1, batch_size, 1, 1).expand(
        t_len,
        batch_size,
        aug_count,
        1,
    )

    pos_logits = logits_aug.gather(dim=3, index=pos_index).squeeze(-1)

    # Denominator part 1:
    # Augmented batch terms from the same augmentation a.
    #
    # If same-sample augmentation terms are included in Part 3,
    # then j=i is already included there. To avoid duplicate positive terms,
    # exclude j=i from Part 1.
    #
    # If Part 3 is disabled, keep j=i in Part 1 so that the positive term
    # remains included in the denominator.
    if include_same_sample_augmentations:
        diag_mask = torch.eye(batch_size, device=anchor.device, dtype=torch.bool)
        diag_mask = diag_mask.view(1, batch_size, 1, batch_size)

        logits_aug_part = logits_aug.masked_fill(diag_mask, float("-inf"))
    else:
        logits_aug_part = logits_aug

    denominator_parts = [logits_aug_part]

    # ------------------------------------------------------------
    # 2. Original batch negatives, excluding self-pair
    # ------------------------------------------------------------
    if include_original_negatives:
        # logits_original[t, i, j]
        #   = anchor_i,t dot anchor_j,t
        #
        # Shape:
        #   (T, B, B)
        logits_original = torch.einsum(
            "tid,tjd->tij",
            anchor_t,
            anchor_t,
        ) / temperature

        # Exclude self-pair j=i.
        mask = torch.eye(batch_size, device=anchor.device, dtype=torch.bool)
        mask = mask.view(1, batch_size, batch_size)

        logits_original = logits_original.masked_fill(mask, float("-inf"))

        # Expand over augmentation dimension.
        #
        # Shape:
        #   (T, B, A, B)
        logits_original = logits_original.unsqueeze(2).expand(
            t_len,
            batch_size,
            aug_count,
            batch_size,
        )

        denominator_parts.append(logits_original)

    # ------------------------------------------------------------
    # 3. Same-sample multiple augmentation terms
    # ------------------------------------------------------------
    if include_same_sample_augmentations:
        # same_aug_logits[t, i, k]
        #   = anchor_i,t dot augmented_k,i,t
        #
        # This is exactly the positive logit for each augmentation k.
        #
        # Shape:
        #   (T, B, A)
        same_aug_logits = pos_logits

        # For each positive augmentation a, include all k=1,...,A terms.
        #
        # Shape:
        #   (T, B, A, A)
        same_aug_logits = same_aug_logits.unsqueeze(2).expand(
            t_len,
            batch_size,
            aug_count,
            aug_count,
        )

        denominator_parts.append(same_aug_logits)

    # Concatenate denominator terms.
    #
    # Shape:
    #   (T, B, A, B + optional B + optional A)
    denominator_logits = torch.cat(denominator_parts, dim=-1)

    # log denominator for each t, i, a.
    #
    # Shape:
    #   (T, B, A)
    log_denominator = torch.logsumexp(denominator_logits, dim=-1)

    # Score:
    #   -log(exp(pos) / denominator)
    #
    # Shape:
    #   (T, B, A)
    scores_tba = -(pos_logits - log_denominator)

    # Return shape:
    #   (A, B, T)
    scores = scores_tba.permute(2, 1, 0).contiguous()

    return scores


@dataclass
class CoTSFAOriginalLossOutput:
    """
    Output container for original Co-TSFA alignment loss.
    """

    loss_align: torch.Tensor
    latent_score_mean: torch.Tensor
    target_score_mean: torch.Tensor
    latent_score_std: torch.Tensor
    target_score_std: torch.Tensor


class CoTSFAOriginalAlignmentLoss(nn.Module):
    """
    Original Co-TSFA latent-output alignment loss.

    This loss computes InfoNCE-style scores in latent space and target space,
    then minimizes their absolute difference.

    L_align = mean(|score_latent - score_target|)
    """

    def __init__(
        self,
        temperature: float = 1.0,
        include_original_negatives: bool = True,
        include_same_sample_augmentations: bool = True,
        normalize: bool = False,
    ) -> None:
        super().__init__()

        self.temperature = temperature
        self.include_original_negatives = include_original_negatives
        self.include_same_sample_augmentations = include_same_sample_augmentations
        self.normalize = normalize

    def forward(
        self,
        z: torch.Tensor,
        z_aug: torch.Tensor,
        y: torch.Tensor,
        y_aug: torch.Tensor,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | CoTSFAOriginalLossOutput:
        """
        Compute original Co-TSFA alignment loss.

        Parameters
        ----------
        z:
            Original latent representation, shape (B, T, D_z).

        z_aug:
            Augmented latent representation, shape (A, B, T, D_z)
            or (B, T, D_z).

        y:
            Original target sequence, shape (B, T, D_y).

        y_aug:
            Augmented target sequence, shape (A, B, T, D_y)
            or (B, T, D_y).

        return_details:
            If True, return a CoTSFAOriginalLossOutput object.

            If False, return:
                loss_align, latent_score_mean, target_score_mean

        Returns
        -------
        Either:
            (loss_align, latent_score_mean, target_score_mean)

        or:
            CoTSFAOriginalLossOutput
        """
        z_aug = _ensure_aug_dim(z_aug, name="z_aug")
        y_aug = _ensure_aug_dim(y_aug, name="y_aug")

        if z_aug.shape[0] != y_aug.shape[0]:
            raise ValueError(
                "z_aug and y_aug must have the same number of augmentations. "
                f"Got z_aug A={z_aug.shape[0]}, y_aug A={y_aug.shape[0]}"
            )

        latent_scores = paper_contrastive_score(
            anchor=z,
            augmented=z_aug,
            temperature=self.temperature,
            include_original_negatives=self.include_original_negatives,
            include_same_sample_augmentations=self.include_same_sample_augmentations,
            normalize=self.normalize,
        )

        target_scores = paper_contrastive_score(
            anchor=y,
            augmented=y_aug,
            temperature=self.temperature,
            include_original_negatives=self.include_original_negatives,
            include_same_sample_augmentations=self.include_same_sample_augmentations,
            normalize=self.normalize,
        )

        if latent_scores.shape != target_scores.shape:
            raise ValueError(
                "latent_scores and target_scores must have the same shape. "
                f"Got {tuple(latent_scores.shape)} and {tuple(target_scores.shape)}"
            )

        loss_align = torch.mean(torch.abs(latent_scores - target_scores))

        latent_score_mean = latent_scores.mean()
        target_score_mean = target_scores.mean()
        latent_score_std = latent_scores.std()
        target_score_std = target_scores.std()

        if return_details:
            return CoTSFAOriginalLossOutput(
                loss_align=loss_align,
                latent_score_mean=latent_score_mean,
                target_score_mean=target_score_mean,
                latent_score_std=latent_score_std,
                target_score_std=target_score_std,
            )

        return loss_align, latent_score_mean, target_score_mean