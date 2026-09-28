"""Partial-label aggregator q_phi(M | obs) for SA-PFE-EMRF Stage S1.1.

Network maps the union of all partial mask observations + per-channel presence
flags into a posterior over the full union mask M (K_union channels).  This is
the deterministic single-step posterior-mean estimator described in plan §S1.1
(mask-only diffusion is a deferred S1.4 stretch).

Inputs at forward time
----------------------
  partial_mask  : (B, K, D, H, W) float in [0,1]
                  union of observed partial masks; channel k is the partial
                  prediction for organ id `scored_ids[k]` if *any* source
                  observed that channel for this case, else zeros.

  presence_mask : (B, K, D, H, W) float in {0,1}
                  per-channel flag broadcast over spatial dims; presence_mask[k]
                  = 1 if any source labelled organ k for this case (i.e. its
                  absence in partial_mask means "really absent"), else 0
                  (meaning "unobserved -- aggregator must infer from priors").

These two channels are stacked along the channel axis: total 2*K input channels.
Compared to feeding raw partial masks per source, this scheme is symmetric in
the source ordering and keeps the input channel count fixed at 2*K regardless
of how many partial sources J there are.

Outputs
-------
  logits : (B, K, D, H, W) raw logits.
  prob   : sigmoid(logits) -- per-channel posterior probability used as the
           soft mask `M_hat` downstream.

Training loss (see scripts/train_aggregator.py)
-----------------------------------------------
  L = BCE(prob, M_gt) * presence_weight + lambda_anat * E_anat(prob)

  presence-aware weighting follows tex Lemma 5.3 -- the BCE on channels with
  no observation is allowed (the aggregator is supposed to fill them in from
  anatomical priors), but loss on observed channels gets weight w_obs while
  unobserved channels get weight w_un (default w_un=1, w_obs=1.5 -- gentle
  push to respect direct observations).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import BasicUNet


@dataclass
class AggregatorConfig:
    n_union_channels: int  # K
    base_features: Sequence[int] = (32, 32, 64, 128, 256, 32)
    norm: str = "instance"  # GroupNorm needs num_groups; instance is simpler
    act: str = "leakyrelu"
    dropout: float = 0.0
    use_presence_mask: bool = True


class PartialMaskAggregator(nn.Module):
    """Single-step posterior-mean aggregator q_phi(M|obs)."""

    def __init__(self, cfg: AggregatorConfig):
        super().__init__()
        self.cfg = cfg
        in_ch = 2 * cfg.n_union_channels if cfg.use_presence_mask else cfg.n_union_channels
        self.net = BasicUNet(
            spatial_dims=3,
            in_channels=in_ch,
            out_channels=cfg.n_union_channels,
            features=tuple(cfg.base_features),
            act=cfg.act,
            norm=cfg.norm,
            dropout=cfg.dropout,
        )

    def forward(self, partial_mask: torch.Tensor, presence_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return raw logits (B, K, D, H, W)."""
        if self.cfg.use_presence_mask:
            if presence_mask is None:
                raise ValueError("aggregator constructed with use_presence_mask=True but no presence_mask given")
            x = torch.cat([partial_mask, presence_mask], dim=1)
        else:
            x = partial_mask
        return self.net(x)

    def posterior_mean(self, partial_mask: torch.Tensor, presence_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sigmoid posterior mean in [0,1]."""
        return torch.sigmoid(self(partial_mask, presence_mask))


def build_partial_inputs(
    full_mask_onehot: torch.Tensor,
    expert_channel_groups: Sequence[Sequence[int]],
    observed_groups: Sequence[Sequence[int]],
    erasure_prob: float = 0.0,
    rng: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic-partial-observation generator (tex Algorithm 1, step 2).

    Args
    ----
      full_mask_onehot       : (B, K, D, H, W) one-hot ground-truth full mask.
      expert_channel_groups  : list of length J, each entry is the channel
                               indices in 0..K-1 belonging to expert j (S_j
                               in the tex).  Partition assumed disjoint.
      observed_groups        : list of length B, each entry is the subset of
                               expert ids 0..J-1 that this batch element saw
                               as observations.  Allows simulating "some
                               cases have only A, some only B, some both".
      erasure_prob           : probability of erasing each channel locally
                               (Bernoulli-blanking per channel).  Bernoulli
                               applied per (batch,channel), zeroes the whole
                               channel.  More sophisticated local erasure
                               masks can be added later.
      rng                    : optional torch.Generator for reproducibility.

    Returns
    -------
      partial_mask  : (B, K, D, H, W) -- union of partial observations.
      presence_mask : (B, K, D, H, W) -- per-channel presence flag broadcast
                       over spatial dims.
    """
    B, K = full_mask_onehot.shape[0], full_mask_onehot.shape[1]
    device = full_mask_onehot.device
    partial = torch.zeros_like(full_mask_onehot)
    presence_per_ch = torch.zeros((B, K), device=device, dtype=full_mask_onehot.dtype)

    for b in range(B):
        for expert_idx in observed_groups[b]:
            ch_ids = expert_channel_groups[expert_idx]
            for ch in ch_ids:
                if erasure_prob > 0:
                    u = torch.rand((), device=device, generator=rng).item()
                    if u < erasure_prob:
                        continue
                partial[b, ch] = full_mask_onehot[b, ch]
                presence_per_ch[b, ch] = 1.0

    # Broadcast presence flag over spatial dims to match conv input.
    presence_mask = presence_per_ch.view(B, K, 1, 1, 1).expand_as(partial).contiguous()
    return partial, presence_mask


def presence_aware_bce(
    prob: torch.Tensor,
    target: torch.Tensor,
    presence_mask: torch.Tensor,
    w_obs: float = 1.5,
    w_unobs: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-channel BCE with presence-aware weighting.

    Args
    ----
      prob          : (B,K,D,H,W) sigmoid output in (0,1).
      target        : (B,K,D,H,W) ground-truth one-hot full mask in {0,1}.
      presence_mask : (B,K,D,H,W) presence flag broadcast over voxels.
      w_obs/w_unobs : per-channel weights based on whether organ was observed.

    Returns scalar mean BCE.
    """
    prob = prob.clamp(eps, 1.0 - eps)
    bce = -(target * torch.log(prob) + (1.0 - target) * torch.log(1.0 - prob))
    # Channel-level weight: read presence at any voxel (broadcast identical) using mean.
    ch_presence = presence_mask.mean(dim=(2, 3, 4))  # (B,K) in {0,1}
    ch_weight = w_obs * ch_presence + w_unobs * (1.0 - ch_presence)  # (B,K)
    ch_weight_vol = ch_weight.view(*ch_weight.shape, 1, 1, 1)
    weighted = bce * ch_weight_vol
    return weighted.mean()


__all__ = [
    "AggregatorConfig",
    "PartialMaskAggregator",
    "build_partial_inputs",
    "presence_aware_bce",
]
