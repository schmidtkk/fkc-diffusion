"""Closed-form anatomy prior energy E_anat(M) for SA-PFE-EMRF (tex §5.2).

Three energy terms, all differentiable w.r.t. a soft mask tensor M:

  E_vol(M)  -- per-organ log-volume z-score (deviates from learned (mu_k, sigma_k)).
  E_excl(M) -- pairwise overlap integral sum_{k<l} <M^k, M^l>.
  E_adj(M)  -- centroid-distance z-score for a fixed set of adjacent organ pairs,
               deviates from learned (mu_dist_kl, sigma_dist_kl).

Used in two places:
  * spatial-aware FKC corrector (src/fkc_poe_sampler.py, S1.3) -- as part of the
    weight increment exp(-Delta_beta * E_sp) where E_sp = E_obs + E_anat + E_reg.
    Needs gradients for the optional velocity correction term.
  * Stage-1/2 ablation metrics (scripts/compose_experiment_sa.py, S1.4) -- as a
    no-grad violation-rate evaluator. Returns per-term breakdown for the report.

Soft-mask convention
--------------------
M : torch.Tensor of shape (B, K, D, H, W), values in [0, 1].
The K channel axis is ordered by `scored_ids` from the anatomy_stats.json the
prior was constructed with.  Hard labels can be one-hot encoded to this shape
via `hard_to_soft(label_vol, scored_ids)`.

Anchor stats file
-----------------
The constructor takes the JSON path written by
`scripts/compute_anatomy_stats.py`.  See that script for the schema (schema_version=1).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


SCHEMA_VERSION_EXPECTED = 1
EPS = 1.0  # additive smoothing inside log(vol + EPS), matches stats script.


# =============================================================================
# Stats container
# =============================================================================


@dataclass
class AnatomyStats:
    """In-memory view of anatomy_stats.json, indexed by 0..K-1 channel positions."""

    scored_ids: list[int]  # length K, MAISI organ ids (channel order)
    id_to_idx: dict[int, int]
    id_to_name: dict[int, str]
    voxel_volume_mm3: float
    grid_shape: tuple[int, int, int]  # (D, H, W) at the resolution stats were computed
    # Per-organ log-volume mu/sigma per channel idx (length K).  NaN where presence_rate<min.
    log_vol_mu: torch.Tensor  # (K,)
    log_vol_sigma: torch.Tensor  # (K,)
    presence_rate: torch.Tensor  # (K,)
    # Centroid mu/sigma per channel in voxel coords (DHW order).
    centroid_mu_vox: torch.Tensor  # (K, 3)
    centroid_sigma_vox: torch.Tensor  # (K, 3)
    # Pairwise centroid-distance stats.  Keys are (i,j) channel-idx pairs with i<j.
    pair_dist_mu: dict[tuple[int, int], float] = field(default_factory=dict)
    pair_dist_sigma: dict[tuple[int, int], float] = field(default_factory=dict)
    pair_dist_n: dict[tuple[int, int], int] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path, device: str | torch.device = "cpu") -> "AnatomyStats":
        path = Path(path)
        payload = json.loads(path.read_text())
        if payload.get("schema_version") != SCHEMA_VERSION_EXPECTED:
            raise ValueError(
                f"anatomy_stats schema_version mismatch in {path}: "
                f"expected {SCHEMA_VERSION_EXPECTED}, got {payload.get('schema_version')}"
            )
        scored_ids = [int(x) for x in payload["scored_ids"]]
        id_to_idx = {k: i for i, k in enumerate(scored_ids)}
        id_to_name = {int(k): v for k, v in payload["id_to_name"].items()}
        K = len(scored_ids)
        log_vol_mu = torch.full((K,), float("nan"), device=device)
        log_vol_sigma = torch.full((K,), float("nan"), device=device)
        presence_rate = torch.zeros(K, device=device)
        centroid_mu_vox = torch.full((K, 3), float("nan"), device=device)
        centroid_sigma_vox = torch.full((K, 3), float("nan"), device=device)

        for sid_str, entry in payload["per_organ"].items():
            sid = int(sid_str)
            if sid not in id_to_idx:
                continue
            idx = id_to_idx[sid]
            presence_rate[idx] = float(entry.get("presence_rate", 0.0))
            lv = entry.get("log_volume_mm3")
            if lv is not None:
                log_vol_mu[idx] = float(lv["mu"])
                log_vol_sigma[idx] = float(lv["sigma"]) if lv["sigma"] > 0 else 1.0
            c = entry.get("centroid_vox")
            if c is not None:
                centroid_mu_vox[idx] = torch.tensor(c["mu"], device=device, dtype=torch.float32)
                sig = torch.tensor(c["sigma"], device=device, dtype=torch.float32)
                centroid_sigma_vox[idx] = torch.where(sig > 0, sig, torch.ones_like(sig))

        pair_dist_mu: dict[tuple[int, int], float] = {}
        pair_dist_sigma: dict[tuple[int, int], float] = {}
        pair_dist_n: dict[tuple[int, int], int] = {}
        for _, entry in payload.get("per_pair", {}).items():
            id_a, id_b = int(entry["id_a"]), int(entry["id_b"])
            if id_a not in id_to_idx or id_b not in id_to_idx:
                continue
            i, j = sorted((id_to_idx[id_a], id_to_idx[id_b]))
            dist = entry["centroid_dist_vox"]
            pair_dist_mu[(i, j)] = float(dist["mu"])
            pair_dist_sigma[(i, j)] = float(dist["sigma"]) if dist["sigma"] > 0 else 1.0
            pair_dist_n[(i, j)] = int(entry["n_cases_joint"])

        return cls(
            scored_ids=scored_ids,
            id_to_idx=id_to_idx,
            id_to_name=id_to_name,
            voxel_volume_mm3=float(payload["voxel_volume_mm3"]),
            grid_shape=tuple(int(x) for x in payload["label_grid_shape"]),
            log_vol_mu=log_vol_mu,
            log_vol_sigma=log_vol_sigma,
            presence_rate=presence_rate,
            centroid_mu_vox=centroid_mu_vox,
            centroid_sigma_vox=centroid_sigma_vox,
            pair_dist_mu=pair_dist_mu,
            pair_dist_sigma=pair_dist_sigma,
            pair_dist_n=pair_dist_n,
        )


# =============================================================================
# Soft mask helpers
# =============================================================================


def hard_to_soft(label_vol: torch.Tensor, scored_ids: Sequence[int]) -> torch.Tensor:
    """Convert (B,D,H,W) or (D,H,W) int label volume into (B,K,D,H,W) one-hot soft mask."""
    if label_vol.dim() == 3:
        label_vol = label_vol.unsqueeze(0)
    if label_vol.dim() != 4:
        raise ValueError(f"label_vol must be 3D (D,H,W) or 4D (B,D,H,W); got {label_vol.shape}")
    B, D, H, W = label_vol.shape
    out = torch.zeros((B, len(scored_ids), D, H, W), dtype=torch.float32, device=label_vol.device)
    for k, sid in enumerate(scored_ids):
        out[:, k] = (label_vol == sid).float()
    return out


def _soft_volume(M: torch.Tensor) -> torch.Tensor:
    """sum over spatial dims -> (B, K) soft voxel count."""
    return M.flatten(2).sum(dim=2)


def _soft_centroid(M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Centroid in voxel coords (D,H,W order).  Returns (B, K, 3)."""
    B, K, D, H, W = M.shape
    device = M.device
    dz = torch.arange(D, device=device, dtype=M.dtype).view(1, 1, D, 1, 1)
    dy = torch.arange(H, device=device, dtype=M.dtype).view(1, 1, 1, H, 1)
    dx = torch.arange(W, device=device, dtype=M.dtype).view(1, 1, 1, 1, W)
    mass = M.flatten(2).sum(dim=2) + eps  # (B,K)
    cz = (M * dz).flatten(2).sum(dim=2) / mass
    cy = (M * dy).flatten(2).sum(dim=2) / mass
    cx = (M * dx).flatten(2).sum(dim=2) / mass
    return torch.stack([cz, cy, cx], dim=-1)


# =============================================================================
# Energy module
# =============================================================================


class AnatomyPrior(nn.Module):
    """E_anat(M) = w_vol*E_vol + w_excl*E_excl + w_adj*E_adj.

    All three terms operate on a soft mask M : (B, K, D, H, W) in [0,1].
    """

    def __init__(
        self,
        stats: AnatomyStats,
        w_vol: float = 1.0,
        w_excl: float = 1.0,
        w_adj: float = 1.0,
        min_presence_rate: float = 0.1,
        adj_max_pairs: Optional[int] = None,
        adj_pair_ids: Optional[Sequence[tuple[int, int]]] = None,
        sigma_clamp: float = 5.0,
    ):
        super().__init__()
        self.stats = stats
        self.K = len(stats.scored_ids)
        self.w_vol = float(w_vol)
        self.w_excl = float(w_excl)
        self.w_adj = float(w_adj)
        self.min_presence_rate = float(min_presence_rate)
        self.sigma_clamp = float(sigma_clamp)

        # Which channel indices have a usable log-volume prior?
        vol_usable = (
            (stats.presence_rate >= min_presence_rate)
            & torch.isfinite(stats.log_vol_mu)
            & torch.isfinite(stats.log_vol_sigma)
        )
        # Replace NaN entries with safe finite defaults so the z-score arithmetic
        # never produces NaN even before the usability mask is applied (0 * NaN = NaN
        # under IEEE 754, which would still poison the sum).
        log_vol_mu_safe = torch.where(vol_usable, stats.log_vol_mu, torch.zeros_like(stats.log_vol_mu))
        log_vol_sigma_safe = torch.where(vol_usable, stats.log_vol_sigma, torch.ones_like(stats.log_vol_sigma))
        self.register_buffer("log_vol_mu", log_vol_mu_safe)
        self.register_buffer("log_vol_sigma", log_vol_sigma_safe)
        self.register_buffer("presence_rate", stats.presence_rate.clone())
        self.register_buffer("centroid_mu_vox", stats.centroid_mu_vox.clone())
        self.register_buffer("vol_usable", vol_usable)

        # Build the adjacency pair list (idx_a, idx_b) ordered.
        if adj_pair_ids is not None:
            id_pairs = [(min(a, b), max(a, b)) for a, b in adj_pair_ids if a != b]
            adj_idx_pairs: list[tuple[int, int]] = []
            for a, b in id_pairs:
                if a in stats.id_to_idx and b in stats.id_to_idx:
                    i, j = sorted((stats.id_to_idx[a], stats.id_to_idx[b]))
                    if (i, j) in stats.pair_dist_mu:
                        adj_idx_pairs.append((i, j))
        else:
            # Default: use all pairs the stats file knows about, optionally capped.
            adj_idx_pairs = list(stats.pair_dist_mu.keys())
            if adj_max_pairs is not None and len(adj_idx_pairs) > adj_max_pairs:
                # Pick the most stable (lowest sigma) pairs.
                adj_idx_pairs.sort(key=lambda p: stats.pair_dist_sigma.get(p, 1e9))
                adj_idx_pairs = adj_idx_pairs[:adj_max_pairs]

        self._adj_pairs = adj_idx_pairs
        if adj_idx_pairs:
            mu = torch.tensor([stats.pair_dist_mu[p] for p in adj_idx_pairs], dtype=torch.float32)
            sg = torch.tensor([stats.pair_dist_sigma[p] for p in adj_idx_pairs], dtype=torch.float32)
            self.register_buffer("adj_mu", mu)
            self.register_buffer("adj_sigma", sg)
            self.register_buffer(
                "adj_idx",
                torch.tensor(adj_idx_pairs, dtype=torch.long),  # (P, 2)
            )
        else:
            self.register_buffer("adj_mu", torch.zeros(0))
            self.register_buffer("adj_sigma", torch.ones(0))
            self.register_buffer("adj_idx", torch.zeros((0, 2), dtype=torch.long))

    # ------------------------------------------------------------------ E_vol

    def e_vol(self, M: torch.Tensor) -> torch.Tensor:
        """Per-batch scalar z-score volume penalty (mean across usable channels)."""
        B, K, D, H, W = M.shape
        assert K == self.K, f"channel dim {K} != prior K {self.K}"
        vol_vox = _soft_volume(M)  # (B,K)
        vol_mm3 = vol_vox * self.stats.voxel_volume_mm3
        log_vol = torch.log(vol_mm3 + EPS)
        z = (log_vol - self.log_vol_mu.unsqueeze(0)) / torch.clamp(
            self.log_vol_sigma.unsqueeze(0), min=1e-3
        )
        z = torch.clamp(z, -self.sigma_clamp, self.sigma_clamp)
        z2 = z * z
        mask = self.vol_usable.float().unsqueeze(0)  # (1,K)
        denom = torch.clamp(mask.sum(), min=1.0)
        return (z2 * mask).sum(dim=1) / denom  # (B,)

    # ----------------------------------------------------------------- E_excl

    def e_excl(self, M: torch.Tensor) -> torch.Tensor:
        """Pairwise overlap mean over K(K-1)/2 pairs, normalized per voxel.

        Implementation: sum_{k<l} <M^k, M^l> = 0.5 * (sum_k M^k)^2 - sum_k (M^k)^2
        Then divide by D*H*W to keep magnitude comparable across resolutions.
        """
        B, K, D, H, W = M.shape
        total = M.sum(dim=1)  # (B,D,H,W)
        sq_each = (M * M).sum(dim=1)
        overlap = 0.5 * (total * total - sq_each)
        return overlap.flatten(1).sum(dim=1) / float(D * H * W)  # (B,)

    # ------------------------------------------------------------------ E_adj

    def e_adj(self, M: torch.Tensor) -> torch.Tensor:
        """Centroid-distance z-score for the adjacency pair list, mean over pairs."""
        if self.adj_idx.numel() == 0:
            return torch.zeros(M.shape[0], device=M.device)
        cen = _soft_centroid(M)  # (B,K,3)
        i_idx = self.adj_idx[:, 0]
        j_idx = self.adj_idx[:, 1]
        ci = cen[:, i_idx]  # (B,P,3)
        cj = cen[:, j_idx]
        d = torch.linalg.norm(ci - cj, dim=-1)  # (B,P)
        z = (d - self.adj_mu.unsqueeze(0)) / torch.clamp(self.adj_sigma.unsqueeze(0), min=1e-3)
        z = torch.clamp(z, -self.sigma_clamp, self.sigma_clamp)
        return (z * z).mean(dim=1)

    # --------------------------------------------------------------- combined

    def forward(self, M: torch.Tensor, return_breakdown: bool = False):
        """E_anat(M) -> scalar per batch, optionally with per-term dict."""
        e_v = self.e_vol(M)
        e_x = self.e_excl(M)
        e_a = self.e_adj(M)
        total = self.w_vol * e_v + self.w_excl * e_x + self.w_adj * e_a
        if return_breakdown:
            return total, {
                "e_vol": e_v.detach(),
                "e_excl": e_x.detach(),
                "e_adj": e_a.detach(),
            }
        return total

    # ---------------------------------------------------- violation reporting

    @torch.no_grad()
    def violations(self, M: torch.Tensor, z_thresh: float = 3.0) -> dict[str, torch.Tensor]:
        """No-grad per-channel violation flags useful for the ablation report."""
        vol_vox = _soft_volume(M)
        log_vol = torch.log(vol_vox * self.stats.voxel_volume_mm3 + EPS)
        z_vol = (log_vol - self.log_vol_mu.unsqueeze(0)) / torch.clamp(
            self.log_vol_sigma.unsqueeze(0), min=1e-3
        )
        out_of_range = (z_vol.abs() > z_thresh) & self.vol_usable.unsqueeze(0)

        # Per-pair overlap fraction (out of min volume of the two organs).
        K = M.shape[1]
        denom = M.flatten(2).sum(dim=2)  # (B,K)
        prod_sum = torch.einsum("bkdhw,bldhw->bkl", M, M)  # (B,K,K)
        # Self-overlap is non-informative; mask diagonal.
        eye = torch.eye(K, device=M.device, dtype=torch.bool).unsqueeze(0)
        prod_sum = prod_sum.masked_fill(eye, 0.0)
        return {
            "z_vol": z_vol.detach(),
            "out_of_range_vol": out_of_range.detach(),
            "n_out_of_range": out_of_range.float().sum(dim=1).detach(),
            "pair_overlap_voxels": prod_sum.detach(),
            "min_organ_voxels": denom.detach(),
        }


__all__ = [
    "AnatomyPrior",
    "AnatomyStats",
    "hard_to_soft",
]
