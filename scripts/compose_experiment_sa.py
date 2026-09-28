#!/usr/bin/env python3
"""Stage-1 SA-PFE-EMRF ablation harness on the TS held-out test set.

Compares four arms with the same trained ControlNet experts and the same
synthetic partial-observation protocol:

    A0  partial_direct   + cfg_poe sampler          (current baseline)
    A1  aggregator-fill  + cfg_poe sampler          (S1.1 q_phi(M|obs))
    A2  partial_direct   + spatial-aware FKC         (S1.3)
    A3  aggregator-fill  + spatial-aware FKC         (S1.1 + S1.3)

The "partial observation" is built by randomly dropping a fraction of each
expert's vocabulary per case.  Without dropout, Path 1 has no missingness
signal -- A0 and A1 are then expected to tie.  With dropout, A1 has the
opportunity to recover the dropped channels via the aggregator.

All four arms are wired.  A2/A3 use ``sample_sa_fkc_poe`` (tex Algorithm 3,
Stage-1 form): spatial-energy potential, ROI-block resampling, coarse-to-fine
β ramp.  At ``K_SMC=1`` the spatial FK weight update is a constant shift on
the single particle, so A2 ≈ A0 and A3 ≈ A1; raise ``--K_SMC`` to observe
FKC dynamics.  Per-step Z_1_hat decode (full Algorithm 3) is a Stage-2 milestone.

Usage
-----
  python scripts/compose_experiment_sa.py \\
    --arm A0 --n-cases 5 --n-steps 30 --drop-prob 0.3 \\
    --aggregator-ckpt checkpoints/aggregator_v0/aggregator_ep19.pt \\
    --out reports/sa_pfe_stage1_a0.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom as sp_zoom
from torch.amp import autocast

REPO_ROOT = Path(__file__).resolve().parents[1]
MAISI_ROOT = REPO_ROOT / "maisi"
sys.path.insert(0, str(MAISI_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from scripts.utils import define_instance, binarize_labels  # noqa: E402
from scripts.diff_model_setting import load_config  # noqa: E402
from src.aggregator import (  # noqa: E402
    AggregatorConfig,
    PartialMaskAggregator,
    build_partial_inputs,
)
from src.anatomy_prior import AnatomyPrior, AnatomyStats  # noqa: E402
from src.fkc_poe_sampler import (  # noqa: E402
    ControlNetExpert,
    SamplerConfig,
    SpatialEnergyEvaluator,
    SpatialFKCConfig,
    sample_fkc_poe_n,
    sample_sa_fkc_poe,
)


# =============================================================================
# Defaults (mirror compose_experiment.py so the two harnesses stay aligned)
# =============================================================================

MODEL_CFG = str(REPO_ROOT / "configs/config_expert_long_60ep.json")
NETWORK_CFG = str(REPO_ROOT / "configs/config_network_amos22_15ch.json")
ENV_CFG = str(REPO_ROOT / "configs/env_ts_expert_A.json")
TS_DATA = REPO_ROOT / "data/phase1_ts_iso15"
LABEL_PARTITION = REPO_ROOT / "data/label_partition.json"
ANATOMY_STATS = TS_DATA / "anatomy_stats.json"
CKPT_A = REPO_ROOT / "checkpoints/ts_expert_A_iso15/checkpoint_latest.pt"
CKPT_B = REPO_ROOT / "checkpoints/ts_expert_B_iso15/checkpoint_latest.pt"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs/compose_experiment_sa"


def log(msg: str) -> None:
    print(f"[{time.strftime('%X')}] {msg}", flush=True)


# =============================================================================
# Backbone loading (UNet + AE + scheduler + ControlNets)
# =============================================================================


def load_backbone(device: torch.device):
    cfg = load_config(ENV_CFG, MODEL_CFG, NETWORK_CFG)
    log("loading UNet...")
    unet = define_instance(cfg, "diffusion_unet_def").to(device)
    sd = torch.load(cfg.trained_diffusion_path, map_location=device, weights_only=False)
    unet.load_state_dict(sd["unet_state_dict"], strict=False)
    scale_factor = sd["scale_factor"]
    if hasattr(scale_factor, "to"):
        scale_factor = scale_factor.to(device)
    else:
        scale_factor = torch.tensor(scale_factor, device=device)
    unet.eval()
    for p in unet.parameters():
        p.requires_grad_(False)

    log("loading AutoEncoder...")
    ae = define_instance(cfg, "autoencoder_def").to(device)
    ae_sd = torch.load(cfg.trained_autoencoder_path, map_location=device, weights_only=False)
    if isinstance(ae_sd, dict) and "unet_state_dict" in ae_sd:
        ae_sd = ae_sd["unet_state_dict"]
    ae.load_state_dict(ae_sd)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    for m in ae.modules():
        if hasattr(m, "norm_float16"):
            m.norm_float16 = False

    noise_scheduler = define_instance(cfg, "noise_scheduler")
    return unet, ae, scale_factor, noise_scheduler, cfg


def load_controlnet(ckpt_path: Path, device: torch.device):
    cfg = load_config(ENV_CFG, MODEL_CFG, NETWORK_CFG)
    controlnet = define_instance(cfg, "controlnet_def").to(device)
    ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    controlnet.load_state_dict(ck["controlnet_state_dict"], strict=True)
    controlnet.eval()
    for p in controlnet.parameters():
        p.requires_grad_(False)
    return controlnet


def load_aggregator(ckpt_path: Path, device: torch.device) -> tuple[PartialMaskAggregator, list[int], list[list[int]]]:
    """Returns (network, scored_ids, expert_channel_groups) for the saved checkpoint."""
    ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg_dict = ck["cfg"]
    cfg = AggregatorConfig(
        n_union_channels=cfg_dict["n_union_channels"],
        base_features=tuple(cfg_dict["base_features"]),
        norm=cfg_dict.get("norm", "instance"),
        act=cfg_dict.get("act", "leakyrelu"),
        dropout=cfg_dict.get("dropout", 0.0),
        use_presence_mask=cfg_dict.get("use_presence_mask", True),
    )
    net = PartialMaskAggregator(cfg).to(device)
    net.load_state_dict(ck["model_state"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, list(ck["scored_ids"]), [list(g) for g in ck["expert_channel_groups"]]


# =============================================================================
# Partial-observation construction
# =============================================================================


def build_test_partials(
    label_volume_t: torch.Tensor,        # (D, H, W) int volume, MAISI ids
    expert_a_ids: list[int],
    expert_b_ids: list[int],
    drop_prob: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    """Drop a Bernoulli fraction of each expert's vocabulary per case.

    Returns
    -------
      partial_a_volume : (D,H,W) int -- label vol with only kept_A ids preserved.
      partial_b_volume : (D,H,W) int -- label vol with only kept_B ids preserved.
      kept_a_ids       : list of MAISI ids kept for A
      kept_b_ids       : list of MAISI ids kept for B
    """
    rng = rng or np.random.default_rng()
    if drop_prob > 0:
        keep_mask_a = rng.random(len(expert_a_ids)) >= drop_prob
        keep_mask_b = rng.random(len(expert_b_ids)) >= drop_prob
    else:
        keep_mask_a = np.ones(len(expert_a_ids), dtype=bool)
        keep_mask_b = np.ones(len(expert_b_ids), dtype=bool)
    kept_a = [sid for sid, k in zip(expert_a_ids, keep_mask_a) if k]
    kept_b = [sid for sid, k in zip(expert_b_ids, keep_mask_b) if k]

    pa = torch.zeros_like(label_volume_t)
    pb = torch.zeros_like(label_volume_t)
    for sid in kept_a:
        pa[label_volume_t == sid] = sid
    for sid in kept_b:
        pb[label_volume_t == sid] = sid
    # Always preserve body silhouette id 200 so MAISI's body context is intact.
    body = label_volume_t == 200
    pa[body] = 200
    pb[body] = 200
    return pa, pb, kept_a, kept_b


def aggregator_filled_labels(
    aggregator: PartialMaskAggregator,
    partial_a_volume: torch.Tensor,  # (D,H,W) int with kept_A ids only (+ id 200)
    partial_b_volume: torch.Tensor,
    scored_ids: list[int],
    expert_a_channel_idx: list[int],
    expert_b_channel_idx: list[int],
    device: torch.device,
    grid_size: int | None = None,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run q_phi(M|obs) and re-derive per-expert filled label volumes.

    The aggregator output is K_union sigmoid posteriors.  For each expert we
    binarise at `threshold` on the channels belonging to that expert and write
    the corresponding MAISI ids back into a label volume of the original shape.
    Body silhouette (id 200) is preserved from the input partials.
    """
    K = aggregator.cfg.n_union_channels
    full_D, full_H, full_W = partial_a_volume.shape
    grid = grid_size or full_D

    id_to_idx = {sid: i for i, sid in enumerate(scored_ids)}
    # Build one-hot partial input on each expert's vocab.
    partial_oh = torch.zeros(1, K, full_D, full_H, full_W, device=device)
    presence = torch.zeros(1, K, full_D, full_H, full_W, device=device)
    for sid in scored_ids:
        idx = id_to_idx[sid]
        in_a = (partial_a_volume == sid)
        in_b = (partial_b_volume == sid)
        partial_oh[0, idx] = torch.maximum(in_a.float(), in_b.float())
        if in_a.any() or in_b.any():
            presence[0, idx] = 1.0

    if grid != full_D:
        partial_oh_lr = F.adaptive_avg_pool3d(partial_oh, grid)
        presence_lr = F.adaptive_avg_pool3d(presence, grid)
    else:
        partial_oh_lr = partial_oh
        presence_lr = presence

    with torch.no_grad():
        prob_lr = aggregator.posterior_mean(partial_oh_lr, presence_lr)
        if grid != full_D:
            prob = F.interpolate(prob_lr, size=(full_D, full_H, full_W), mode="trilinear", align_corners=False)
        else:
            prob = prob_lr

    out_a = partial_a_volume.clone()
    out_b = partial_b_volume.clone()
    for ch_idx in expert_a_channel_idx:
        sid = scored_ids[ch_idx]
        keep = (prob[0, ch_idx] >= threshold) & (partial_a_volume == 0) & (partial_b_volume == 0)
        out_a[keep] = sid
    for ch_idx in expert_b_channel_idx:
        sid = scored_ids[ch_idx]
        keep = (prob[0, ch_idx] >= threshold) & (partial_a_volume == 0) & (partial_b_volume == 0)
        out_b[keep] = sid
    return out_a, out_b


# =============================================================================
# Generation (composed cfg_poe, reused from compose_experiment.py)
# =============================================================================


def run_composed(experts, unet, ae, noise_scheduler, scale_factor,
                 spacings, modalities, conds, uncond, device,
                 composition: str = "cfg_poe", n_steps: int = 30, K_SMC: int = 1):
    C, D, H, W = 4, conds[0].shape[2] // 4, conds[0].shape[3] // 4, conds[0].shape[4] // 4
    exps = [ControlNetExpert(name=f"expert_{i}", controlnet=exp["controlnet"],
                             controlnet_cond=conds[i]) for i, exp in enumerate(experts)]
    config = SamplerConfig(K_SMC=K_SMC, n_steps=n_steps, composition=composition,
                           use_fkc_weights=False, seed=1234)
    latents, log_w, diag = sample_fkc_poe_n(
        exps, unet, noise_scheduler, scale_factor,
        spacing_tensor=spacings, modality_tensor=modalities,
        latent_shape=(C, D, H, W), config=config,
        uncond_controlnet_cond=uncond if composition == "cfg_poe" else None,
        include_modality=True, include_body_region=False,
    )
    best_idx = log_w.argmax().item() if K_SMC > 1 else 0
    with torch.no_grad():
        ct_norm = ae.decode_stage_2_outputs(latents[best_idx:best_idx + 1] / scale_factor)
        ct_norm = torch.clip(ct_norm, 0.0, 1.0)
    return (ct_norm * 2000.0 - 1000.0).squeeze().cpu().numpy(), diag


def run_sa_composed(experts, unet, ae, noise_scheduler, scale_factor,
                    spacings, modalities, conds, uncond, device,
                    M_soft_per_particle: torch.Tensor,
                    partial_obs: torch.Tensor,
                    presence: torch.Tensor,
                    spatial_evaluator: SpatialEnergyEvaluator,
                    spatial_config: SpatialFKCConfig,
                    composition: str = "cfg_poe", n_steps: int = 30, K_SMC: int = 1):
    """Spatial-aware FKC composition (arms A2/A3) -- tex Algorithm 3 Stage-1.

    M_soft_per_particle / partial_obs / presence are at the aggregator's grid
    (default 64^3 in compose_experiment_sa); the spatial energy evaluator works
    natively on that grid.  Particle latents live at MAISI's latent grid; the
    two are coupled only through the conditioning masks fed to ControlNet.
    """
    C, D, H, W = 4, conds[0].shape[2] // 4, conds[0].shape[3] // 4, conds[0].shape[4] // 4
    exps = [ControlNetExpert(name=f"expert_{i}", controlnet=exp["controlnet"],
                             controlnet_cond=conds[i]) for i, exp in enumerate(experts)]
    config = SamplerConfig(K_SMC=K_SMC, n_steps=n_steps, composition=composition,
                           use_fkc_weights=False, seed=1234)
    latents, log_w, diag = sample_sa_fkc_poe(
        exps, unet, noise_scheduler, scale_factor,
        spacing_tensor=spacings, modality_tensor=modalities,
        latent_shape=(C, D, H, W), config=config,
        spatial_config=spatial_config,
        M_soft_per_particle=M_soft_per_particle,
        partial_obs=partial_obs,
        presence=presence,
        spatial_evaluator=spatial_evaluator,
        uncond_controlnet_cond=uncond if composition == "cfg_poe" else None,
        include_modality=True, include_body_region=False,
    )
    best_idx = log_w.argmax().item() if K_SMC > 1 else 0
    with torch.no_grad():
        ct_norm = ae.decode_stage_2_outputs(latents[best_idx:best_idx + 1] / scale_factor)
        ct_norm = torch.clip(ct_norm, 0.0, 1.0)
    return (ct_norm * 2000.0 - 1000.0).squeeze().cpu().numpy(), diag


def build_spatial_inputs(
    scored_ids: list[int],
    partial_a_volume: torch.Tensor,
    partial_b_volume: torch.Tensor,
    aggregator: PartialMaskAggregator | None,
    expert_a_channel_idx: list[int],
    expert_b_channel_idx: list[int],
    device: torch.device,
    K_SMC: int,
    grid_size: int,
    target_K: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (M_soft_per_particle, partial_obs, presence) at ``grid_size``.

    - partial_obs/presence are derived from the partial label volumes.
    - M_soft_per_particle is the per-particle soft mask.  For A2 it equals the
      partial-direct one-hot mask (same across particles).  For A3 it is the
      aggregator's posterior mean (also same across particles in Stage-1; per-
      particle variation is a Stage-2 milestone via stochastic aggregator).
    """
    K = len(scored_ids)
    full_D, full_H, full_W = partial_a_volume.shape
    id_to_idx = {sid: i for i, sid in enumerate(scored_ids)}
    partial_oh = torch.zeros(1, K, full_D, full_H, full_W, device=device)
    presence = torch.zeros(1, K, full_D, full_H, full_W, device=device)
    for sid in scored_ids:
        idx = id_to_idx[sid]
        in_a = (partial_a_volume == sid)
        in_b = (partial_b_volume == sid)
        partial_oh[0, idx] = torch.maximum(in_a.float(), in_b.float())
        if in_a.any() or in_b.any():
            presence[0, idx] = 1.0

    if grid_size != full_D:
        partial_oh_lr = F.adaptive_avg_pool3d(partial_oh, grid_size)
        presence_lr = F.adaptive_avg_pool3d(presence, grid_size)
    else:
        partial_oh_lr = partial_oh
        presence_lr = presence

    if aggregator is not None:
        with torch.no_grad():
            M_soft = aggregator.posterior_mean(partial_oh_lr, presence_lr)  # (1, K, g, g, g)
    else:
        M_soft = partial_oh_lr

    # If the anatomy-prior expects K+1 (body silhouette is last), pad with zeros.
    # The body channel contributes a constant penalty across particles -- no FKC bias.
    if target_K is not None and target_K > M_soft.shape[1]:
        npad = target_K - M_soft.shape[1]
        zpad_M = torch.zeros((1, npad, *M_soft.shape[2:]), device=M_soft.device, dtype=M_soft.dtype)
        zpad_O = torch.zeros((1, npad, *partial_oh_lr.shape[2:]),
                             device=partial_oh_lr.device, dtype=partial_oh_lr.dtype)
        zpad_P = torch.zeros((1, npad, *presence_lr.shape[2:]),
                             device=presence_lr.device, dtype=presence_lr.dtype)
        M_soft = torch.cat([M_soft, zpad_M], dim=1)
        partial_oh_lr = torch.cat([partial_oh_lr, zpad_O], dim=1)
        presence_lr = torch.cat([presence_lr, zpad_P], dim=1)

    M_soft_p = M_soft.expand(K_SMC, *M_soft.shape[1:]).contiguous()
    return M_soft_p, partial_oh_lr, presence_lr


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", type=str, choices=["A0", "A1", "A2", "A3"], default="A0")
    ap.add_argument("--n-cases", type=int, default=5)
    ap.add_argument("--n-steps", type=int, default=30)
    ap.add_argument("--K_SMC", type=int, default=1)
    ap.add_argument("--drop-prob", type=float, default=0.3,
                    help="Per-expert channel dropout probability for synthetic partial obs.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--aggregator-ckpt", type=Path, default=None,
                    help="Required for arms A1/A3; ignored for A0/A2.")
    ap.add_argument("--anatomy-stats", type=Path, default=ANATOMY_STATS)
    ap.add_argument("--ckpt-a", type=Path, default=CKPT_A)
    ap.add_argument("--ckpt-b", type=Path, default=CKPT_B)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--device", type=str, default="cuda:0")
    # Spatial-aware FKC knobs (A2/A3 only).
    ap.add_argument("--sa-beta-lo", type=float, default=0.0)
    ap.add_argument("--sa-beta-hi", type=float, default=1.0)
    ap.add_argument("--sa-w-obs", type=float, default=1.0)
    ap.add_argument("--sa-w-anat", type=float, default=1.0)
    ap.add_argument("--sa-n-blocks", type=int, default=1)
    ap.add_argument("--sa-ess-frac", type=float, default=0.5)
    ap.add_argument("--sa-grid", type=int, default=64,
                    help="Grid for spatial-energy evaluation (aggregator native grid).")
    ap.add_argument("--agg-threshold", type=float, default=0.5,
                    help="Aggregator binarisation threshold (A1/A3 only).")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    # ---- partition + scored ids ---------------------------------------------
    partition = json.loads(LABEL_PARTITION.read_text())
    expert_a_ids = list(partition["expert_A_abdominal"])
    expert_b_ids = list(partition["expert_B_spine"])
    union_ids = sorted(set(expert_a_ids) | set(expert_b_ids))

    # ---- backbone -----------------------------------------------------------
    unet, ae, scale_factor, noise_scheduler, _ = load_backbone(device)
    log(f"loading ControlNet A from {args.ckpt_a}")
    ctrl_a = load_controlnet(args.ckpt_a, device)
    log(f"loading ControlNet B from {args.ckpt_b}")
    ctrl_b = load_controlnet(args.ckpt_b, device)
    experts = [
        {"name": "Expert A (abdominal)", "controlnet": ctrl_a},
        {"name": "Expert B (spine)", "controlnet": ctrl_b},
    ]

    # ---- aggregator + anatomy prior (A1/A3) ---------------------------------
    aggregator = None
    scored_ids = union_ids
    expert_a_channel_idx: list[int] = []
    expert_b_channel_idx: list[int] = []
    if args.arm in ("A1", "A3"):
        if args.aggregator_ckpt is None or not args.aggregator_ckpt.exists():
            raise FileNotFoundError(
                f"Arm {args.arm} requires --aggregator-ckpt pointing to a trained "
                f"src.aggregator.PartialMaskAggregator checkpoint."
            )
        log(f"loading aggregator from {args.aggregator_ckpt}")
        aggregator, scored_ids, ecg = load_aggregator(args.aggregator_ckpt, device)
        expert_a_channel_idx, expert_b_channel_idx = ecg[0], ecg[1]

    anatomy_prior = None
    if args.anatomy_stats.exists():
        stats = AnatomyStats.from_json(args.anatomy_stats, device=device)
        anatomy_prior = AnatomyPrior(stats).to(device)
        log(f"anatomy prior loaded ({len(anatomy_prior.adj_idx)} adjacency pairs).")
    else:
        log(f"[warn] anatomy stats absent at {args.anatomy_stats}; violation report disabled.")

    # Spatial-FKC components for A2/A3.
    spatial_evaluator = None
    spatial_config = None
    if args.arm in ("A2", "A3"):
        if anatomy_prior is None:
            raise FileNotFoundError(
                f"Arm {args.arm} requires anatomy stats at {args.anatomy_stats} "
                "to build the spatial energy evaluator."
            )
        spatial_evaluator = SpatialEnergyEvaluator(
            anatomy_prior=anatomy_prior,
            w_obs=args.sa_w_obs,
            w_anat=args.sa_w_anat,
        )
        spatial_config = SpatialFKCConfig(
            w_obs=args.sa_w_obs,
            w_anat=args.sa_w_anat,
            beta_lo=args.sa_beta_lo,
            beta_hi=args.sa_beta_hi,
            n_blocks_per_axis=args.sa_n_blocks,
            ess_threshold_frac=args.sa_ess_frac,
        )
        log(f"spatial-FKC: w_obs={args.sa_w_obs} w_anat={args.sa_w_anat} "
            f"β=[{args.sa_beta_lo},{args.sa_beta_hi}] n_blocks={args.sa_n_blocks}")

    # ---- test set -----------------------------------------------------------
    ds = json.loads((TS_DATA / "expert_U_dataset.json").read_text())
    entries = ds["training"]  # phase1_ts_iso15 uses 'training' key only -- treat tail as holdout
    test_entries = entries[-args.n_cases:]
    log(f"running arm {args.arm} on {len(test_entries)} test cases (drop_prob={args.drop_prob})")

    per_case_results: list[dict] = []
    for i, entry in enumerate(test_entries):
        lbl_path = TS_DATA / entry["label"]
        ct_path = TS_DATA / entry["image"]
        if not (lbl_path.exists() and ct_path.exists()):
            log(f"  case {i}: missing files, skip")
            continue
        log(f"  case {i}: {lbl_path.name}")
        lbl_u_np = nib.load(str(lbl_path)).get_fdata().astype(np.int64)
        lbl_u = torch.from_numpy(lbl_u_np).to(device)

        pa, pb, kept_a, kept_b = build_test_partials(
            lbl_u, expert_a_ids, expert_b_ids, drop_prob=args.drop_prob, rng=rng,
        )
        log(f"    kept {len(kept_a)}/{len(expert_a_ids)} A ids, {len(kept_b)}/{len(expert_b_ids)} B ids")

        if args.arm in ("A1", "A3"):
            pa, pb = aggregator_filled_labels(
                aggregator, pa, pb, scored_ids,
                expert_a_channel_idx, expert_b_channel_idx, device,
                grid_size=args.sa_grid,
                threshold=args.agg_threshold,
            )
            n_filled_a = int((pa != 0).sum().item())
            n_filled_b = int((pb != 0).sum().item())
            log(f"    aggregator @ thresh={args.agg_threshold}: pa={n_filled_a} pb={n_filled_b} non-zero voxels")

        cond_a = binarize_labels(pa.unsqueeze(0).unsqueeze(0).long()).float().repeat(args.K_SMC, 1, 1, 1, 1)
        cond_b = binarize_labels(pb.unsqueeze(0).unsqueeze(0).long()).float().repeat(args.K_SMC, 1, 1, 1, 1)
        uncond = binarize_labels(torch.zeros_like(lbl_u).unsqueeze(0).unsqueeze(0).long()).float().repeat(args.K_SMC, 1, 1, 1, 1)

        spacing = (torch.tensor([[1.5, 1.5, 1.5]], device=device, dtype=torch.float32)
                   .repeat(args.K_SMC, 1) * 100.0)
        modality = torch.ones((args.K_SMC,), device=device, dtype=torch.long)

        gt_ct = nib.load(str(ct_path)).get_fdata().astype(np.float32)
        if gt_ct.shape != lbl_u_np.shape:
            gt_ct = sp_zoom(gt_ct, [lbl_u_np.shape[d] / gt_ct.shape[d] for d in range(3)], order=1)
        gt_ct = np.clip(gt_ct, -1000, 1000)

        t0 = time.time()
        try:
            if args.arm in ("A2", "A3"):
                M_soft_p, partial_obs_lr, presence_lr = build_spatial_inputs(
                    scored_ids, pa, pb,
                    aggregator if args.arm == "A3" else None,
                    expert_a_channel_idx, expert_b_channel_idx,
                    device, K_SMC=args.K_SMC, grid_size=args.sa_grid,
                    target_K=anatomy_prior.K if anatomy_prior is not None else None,
                )
                gen, diag = run_sa_composed(
                    experts, unet, ae, noise_scheduler, scale_factor,
                    spacing, modality, [cond_a, cond_b], uncond, device,
                    M_soft_per_particle=M_soft_p,
                    partial_obs=partial_obs_lr,
                    presence=presence_lr,
                    spatial_evaluator=spatial_evaluator,
                    spatial_config=spatial_config,
                    composition="cfg_poe", n_steps=args.n_steps, K_SMC=args.K_SMC,
                )
            else:
                gen, diag = run_composed(
                    experts, unet, ae, noise_scheduler, scale_factor,
                    spacing, modality, [cond_a, cond_b], uncond, device,
                    composition="cfg_poe", n_steps=args.n_steps, K_SMC=args.K_SMC,
                )
            elapsed = time.time() - t0
        except Exception as e:
            log(f"    FAIL: {e}")
            torch.cuda.empty_cache(); gc.collect()
            continue

        gt_body = gt_ct > -300
        gen_body = gen > -500
        body_mask = gt_body & gen_body
        if body_mask.sum() == 0:
            log(f"    empty body mask, skip metrics")
            continue
        mae = float(np.abs(gen[body_mask] - gt_ct[body_mask]).mean())
        mse = float(((gen[body_mask] - gt_ct[body_mask]) ** 2).mean())
        psnr = 20 * np.log10(2000.0 / max(np.sqrt(mse), 1e-6))

        # Optional anatomy violation -- compute on the partial mask we conditioned on
        violations: dict = {}
        if anatomy_prior is not None:
            from src.anatomy_prior import hard_to_soft
            cond_oh = hard_to_soft(pa.unsqueeze(0).cpu(), scored_ids)  # (1,K,D,H,W)
            cond_oh_lr = F.adaptive_avg_pool3d(cond_oh, 64).to(device)
            if cond_oh_lr.shape[1] != anatomy_prior.K:
                npad = anatomy_prior.K - cond_oh_lr.shape[1]
                zpad = torch.zeros((1, npad, *cond_oh_lr.shape[2:]),
                                   device=device, dtype=cond_oh_lr.dtype)
                cond_oh_lr = torch.cat([cond_oh_lr, zpad], dim=1)
            v = anatomy_prior.violations(cond_oh_lr)
            violations = {
                "n_out_of_range_vol": int(v["n_out_of_range"].sum().item()),
                "pair_overlap_total_voxels": float(v["pair_overlap_voxels"].sum().item()),
            }

        case_result = {
            "case": lbl_path.name,
            "kept_a": kept_a,
            "kept_b": kept_b,
            "psnr": psnr,
            "mae_body": mae,
            "time_s": elapsed,
            "violations": violations,
        }
        per_case_results.append(case_result)
        log(f"    PSNR={psnr:.2f}dB MAE={mae:.0f}HU time={elapsed:.1f}s")
        torch.cuda.empty_cache(); gc.collect()

    # ---- summary ------------------------------------------------------------
    if per_case_results:
        psnrs = np.array([r["psnr"] for r in per_case_results])
        maes = np.array([r["mae_body"] for r in per_case_results])
        summary = {
            "arm": args.arm,
            "drop_prob": args.drop_prob,
            "n_cases": len(per_case_results),
            "psnr_mean": float(psnrs.mean()),
            "psnr_std": float(psnrs.std(ddof=1)) if len(psnrs) > 1 else 0.0,
            "mae_mean": float(maes.mean()),
            "per_case": per_case_results,
        }
    else:
        summary = {"arm": args.arm, "drop_prob": args.drop_prob, "n_cases": 0,
                   "psnr_mean": None, "psnr_std": None, "mae_mean": None,
                   "per_case": []}

    out_json = args.out / f"summary_arm_{args.arm}_drop{args.drop_prob:.2f}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    log(f"wrote {out_json}")
    if summary["n_cases"] > 0:
        log(f"ARM {args.arm}  PSNR mean={summary['psnr_mean']:.2f} +/- {summary['psnr_std']:.2f}  "
            f"MAE_body={summary['mae_mean']:.1f}  n={summary['n_cases']}")


if __name__ == "__main__":
    main()
