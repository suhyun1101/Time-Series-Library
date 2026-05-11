"""
Co-TSFA-style TimesNet wrapper for Phase 3.

This module preserves the original Time-Series-Library TimesNet implementation
and adds methods for extracting latent representations.

Original TimesNet:
    model(...) -> forecast output

CoTSFATimesNet:
    forward_with_latent(...) -> forecast output, latent representation
    forward_cotsfa(...)      -> clean output, augmented output, clean latent, augmented latent
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch

from models.TimesNet import Model as TimesNetModel


LatentSlice = Literal["full", "pred", "seq"]


class CoTSFATimesNet(TimesNetModel):
    """
    TimesNet extension for Co-TSFA-style training.

    The original TimesNet forecast path is preserved.
    This class only adds latent extraction methods.

    Latent candidate:
        enc_out after TimesBlock layers and before final projection.

    In the long-term forecasting task, enc_out has shape:

        (B, seq_len + pred_len, d_model)

    For alignment with forecasting targets, the default latent slice is "pred":

        z = enc_out[:, -pred_len:, :]

    This aligns the latent representation with the prediction horizon.
    """

    def _select_latent(
        self,
        latent_full: torch.Tensor,
        latent_slice: LatentSlice = "pred",
    ) -> torch.Tensor:
        """
        Select which part of the latent representation to use.

        Parameters
        ----------
        latent_full:
            Tensor with shape (B, seq_len + pred_len, d_model).
        latent_slice:
            - "full": use the full latent sequence
            - "pred": use the prediction horizon part
            - "seq": use the original input sequence part

        Returns
        -------
        latent:
            Selected latent representation.
        """
        if latent_slice == "full":
            return latent_full

        if latent_slice == "pred":
            return latent_full[:, -self.pred_len:, :]

        if latent_slice == "seq":
            return latent_full[:, :self.seq_len, :]

        raise ValueError(f"Unknown latent_slice: {latent_slice}")

    def forecast_with_latent(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        latent_slice: LatentSlice = "pred",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run TimesNet forecasting and return both forecast and latent.

        This method mirrors the original TimesNet forecast() method.
        The only difference is that it stores enc_out before projection.

        Returns
        -------
        dec_out:
            Forecast output before final slicing, shape:
            (B, seq_len + pred_len, C)

        latent:
            Selected latent representation.
        """
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)

        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc = x_enc.div(stdev)

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Extend temporal dimension from seq_len to seq_len + pred_len
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        # TimesNet blocks
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Latent representation before final projection
        latent_full = enc_out
        latent = self._select_latent(latent_full, latent_slice=latent_slice)

        # Project back to feature dimension
        dec_out = self.projection(enc_out)

        # De-normalization
        dec_out = dec_out.mul(
            stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len + self.seq_len, 1
            )
        )
        dec_out = dec_out.add(
            means[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len + self.seq_len, 1
            )
        )

        return dec_out, latent

    def forward_with_latent(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        latent_slice: LatentSlice = "pred",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward method for forecasting with latent representation.

        Returns
        -------
        forecast:
            Shape (B, pred_len, C)

        latent:
            Shape depends on latent_slice.
            Default "pred": (B, pred_len, d_model)
        """
        if self.task_name not in ["long_term_forecast", "short_term_forecast"]:
            raise NotImplementedError(
                "forward_with_latent is currently implemented only for forecasting tasks."
            )

        dec_out, latent = self.forecast_with_latent(
            x_enc=x_enc,
            x_mark_enc=x_mark_enc,
            x_dec=x_dec,
            x_mark_dec=x_mark_dec,
            latent_slice=latent_slice,
        )

        forecast = dec_out[:, -self.pred_len:, :]
        return forecast, latent

    def forward_cotsfa(
        self,
        x_enc: torch.Tensor,
        x_enc_aug: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        latent_slice: LatentSlice = "pred",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward clean and augmented inputs for Co-TSFA-style training.

        Parameters
        ----------
        x_enc:
            Clean input sequence.
        x_enc_aug:
            Augmented input sequence.
        x_mark_enc:
            Time features for encoder input.
        x_dec:
            Decoder input.
        x_mark_dec:
            Time features for decoder input.
        latent_slice:
            Which latent part to use.

        Returns
        -------
        y_hat:
            Forecast from clean input.
        y_hat_aug:
            Forecast from augmented input.
        z:
            Latent representation from clean input.
        z_aug:
            Latent representation from augmented input.
        """
        y_hat, z = self.forward_with_latent(
            x_enc=x_enc,
            x_mark_enc=x_mark_enc,
            x_dec=x_dec,
            x_mark_dec=x_mark_dec,
            latent_slice=latent_slice,
        )

        y_hat_aug, z_aug = self.forward_with_latent(
            x_enc=x_enc_aug,
            x_mark_enc=x_mark_enc,
            x_dec=x_dec,
            x_mark_dec=x_mark_dec,
            latent_slice=latent_slice,
        )

        return y_hat, y_hat_aug, z, z_aug