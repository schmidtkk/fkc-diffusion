"""Latent-resolution ownership gates + private-zone masks for Shielded Local
Expert Composition (see sample_shielded_local in fkc_poe_sampler.py).

The MAISI VAE downsamples 256^3 CT -> 4x64^3 latent (factor 4). Ownership /
private masks are built from the full-res integer cond masks and reduced to
64^3 by 4x4x4 max-pooling (any-occupancy -> 1), then dilated / blurred in
latent space.

Mutually-exclusive soft ownership (per the MVP spec):
    g_A = s_A / (s_A + s_B + tau),  g_B = s_B / (s_A + s_B + tau)
where s_k is a blurred, dilated indicator of expert k's OWN organs (not body,
not background). Base owns the rest: g_0 = tau / (s_A + s_B + tau).
"""
from __future__ import annotations
import numpy as np
import torch
from scipy import ndimage as ndi


def _maxpool_to_latent(vol_bool: np.ndarray, latent_dim: int = 64) -> np.ndarray:
    """4x4x4 (or factor f) max-pool a 256^3 bool volume to latent_dim^3."""
    f = vol_bool.shape[0] // latent_dim
    if f < 1:
        return vol_bool.astype(np.float32)
    s = latent_dim * f
    v = vol_bool[:s, :s, :s].reshape(latent_dim, f, latent_dim, f, latent_dim, f)
    return v.max(axis=(1, 3, 5)).astype(np.float32)


def build_shield_masks(
    cond_A_arr: np.ndarray,      # (256,256,256) int — A's cond (organs at IDs, body=200, bg=0)
    cond_B_arr: np.ndarray,
    abd_ids: set,                # A's OWN organ ids (exclude body 200)
    spine_ids: set,              # B's OWN organ ids
    latent_dim: int = 64,
    r_private: int = 3,          # private-zone dilation (latent voxels)
    r_soft: int = 5,             # ownership soft-boundary dilation (latent voxels)
    sigma: float = 1.5,          # gaussian blur for soft ownership
    tau: float = 0.1,            # base share / normalisation floor
    device=None,
):
    """Returns (g_A, g_B, P_A, P_B) each a (1,1,L,L,L) float tensor in [0,1]."""
    A_priv = np.isin(cond_A_arr, list(abd_ids))
    B_priv = np.isin(cond_B_arr, list(spine_ids))

    A_lat = _maxpool_to_latent(A_priv, latent_dim)   # (L,L,L) {0,1}
    B_lat = _maxpool_to_latent(B_priv, latent_dim)

    # Hard private zones (dilated)
    P_A = ndi.binary_dilation(A_lat > 0, iterations=r_private).astype(np.float32)
    P_B = ndi.binary_dilation(B_lat > 0, iterations=r_private).astype(np.float32)

    # Soft ownership scores: dilated + blurred indicator
    s_A = ndi.gaussian_filter(ndi.binary_dilation(A_lat > 0, iterations=r_soft).astype(np.float32), sigma)
    s_B = ndi.gaussian_filter(ndi.binary_dilation(B_lat > 0, iterations=r_soft).astype(np.float32), sigma)
    denom = s_A + s_B + tau
    g_A = s_A / denom
    g_B = s_B / denom

    def _t(a):
        return torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0).to(device) if device is not None \
            else torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0)

    return _t(g_A), _t(g_B), _t(P_A), _t(P_B)
