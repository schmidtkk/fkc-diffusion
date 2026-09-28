#!/usr/bin/env python
"""Verify the spatial-aware FKC component (sample_sa_fkc_poe).

Setup: same TS CT, aligned A/B masks from labels_U. To give the SA-FKC
spatial energy something to discriminate on across K_SMC particles, apply
per-particle random channel dropout to the conditioning masks. Each particle
sees a different subset of organ channels; the body silhouette (id 200) is
preserved for all particles.

Compares against the K_SMC=1 baseline (aligned cfg_poe) from Section 6.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import argparse

import nibabel as nib
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
MAISI = REPO / "maisi"
sys.path.insert(0, str(MAISI))
sys.path.insert(0, str(REPO))

from scripts.utils import define_instance, binarize_labels  # noqa: E402
from scripts.diff_model_setting import load_config  # noqa: E402
from src.fkc_poe_sampler import (  # noqa: E402
    ControlNetExpert, SamplerConfig, SpatialFKCConfig, SpatialEnergyEvaluator,
    sample_sa_fkc_poe, snis_weights,
)
from src.anatomy_prior import AnatomyPrior, AnatomyStats  # noqa: E402

DATA = REPO / "data/phase1_ts_iso15"
OUT = REPO / "outputs/verify_path1_sa_fkc"
OUT.mkdir(parents=True, exist_ok=True)
REP = REPO / "reports/verify_path1"


def build_aligned(labels_U, keep_ids):
    out = np.full_like(labels_U, 200, dtype=np.int32)
    body_mask = labels_U != 0
    out[~body_mask] = 0
    for sid in keep_ids:
        out[labels_U == sid] = sid
    return out


def psnr_mae(gen, gt):
    gt_body = gt > -300
    gen_body = gen > -500
    mask = gt_body & gen_body
    if mask.sum() == 0:
        return float("nan"), float("nan")
    mae = float(np.abs(gen[mask] - gt[mask]).mean())
    mse = float(((gen[mask] - gt[mask]) ** 2).mean())
    return 20 * float(np.log10(2000.0 / max(np.sqrt(mse), 1e-6))), mae


def per_particle_dropout_mask(base_mask, expert_ids, drop_prob, rng):
    """Produce a mask where each non-body organ channel from expert_ids is
    dropped (set to 200) with probability drop_prob. Body=200 is preserved.
    """
    out = base_mask.copy()
    for sid in expert_ids:
        if rng.random() < drop_prob:
            out[out == sid] = 200
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["case_0000", "case_0149", "case_0500",
                                                    "case_1050", "case_1062"])
    ap.add_argument("--k-smc", type=int, default=4)
    ap.add_argument("--drop-prob", type=float, default=0.3)
    ap.add_argument("--n-steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    MODEL_CFG = str(REPO / "configs/config_expert_long_60ep.json")
    NETWORK_CFG = str(REPO / "configs/config_network_amos22_15ch.json")
    ENV_CFG = str(REPO / "configs/env_ts_expert_A.json")
    CKPT_A = REPO / "checkpoints/ts_expert_A_iso15/checkpoint_latest.pt"
    CKPT_B = REPO / "checkpoints/ts_expert_B_iso15/checkpoint_latest.pt"
    partition = json.loads((REPO / "data/label_partition.json").read_text())
    expert_A_ids = partition["expert_A_abdominal"]
    expert_B_ids = partition["expert_B_spine"]
    cfg_obj = load_config(ENV_CFG, MODEL_CFG, NETWORK_CFG)

    print("[load] backbone + ControlNets ...")
    unet = define_instance(cfg_obj, "diffusion_unet_def").to(device)
    sd = torch.load(cfg_obj.trained_diffusion_path, map_location=device, weights_only=False)
    unet.load_state_dict(sd["unet_state_dict"], strict=False)
    scale_factor = sd["scale_factor"]
    scale_factor = scale_factor.to(device) if hasattr(scale_factor, "to") else torch.tensor(scale_factor, device=device)
    unet.eval()
    for p in unet.parameters(): p.requires_grad_(False)

    # AE on CPU to free GPU for K_SMC ControlNet passes (HANDOFF §4.4 workaround)
    ae = define_instance(cfg_obj, "autoencoder_def").cpu()
    ae_sd = torch.load(cfg_obj.trained_autoencoder_path, map_location="cpu", weights_only=False)
    if isinstance(ae_sd, dict) and "unet_state_dict" in ae_sd:
        ae_sd = ae_sd["unet_state_dict"]
    ae.load_state_dict(ae_sd)
    ae.eval()
    for p in ae.parameters(): p.requires_grad_(False)
    for m in ae.modules():
        if hasattr(m, "norm_float16"): m.norm_float16 = False
    ae_device = torch.device("cpu")
    noise_scheduler = define_instance(cfg_obj, "noise_scheduler")

    ctrl_a = define_instance(cfg_obj, "controlnet_def").to(device)
    ctrl_a.load_state_dict(torch.load(str(CKPT_A), map_location=device, weights_only=False)["controlnet_state_dict"], strict=True)
    ctrl_a.eval()
    for p in ctrl_a.parameters(): p.requires_grad_(False)
    ctrl_b = define_instance(cfg_obj, "controlnet_def").to(device)
    ctrl_b.load_state_dict(torch.load(str(CKPT_B), map_location=device, weights_only=False)["controlnet_state_dict"], strict=True)
    ctrl_b.eval()
    for p in ctrl_b.parameters(): p.requires_grad_(False)

    # Anatomy prior for spatial energy
    print("[load] anatomy prior ...")
    stats = AnatomyStats.from_json(str(REPO / "data/phase1_ts_iso15/anatomy_stats.json"), device=device)
    anatomy_prior = AnatomyPrior(stats).to(device)
    print(f"  K_anat = {anatomy_prior.K}, {len(anatomy_prior.adj_idx)} adjacency pairs")

    sa_eval = SpatialEnergyEvaluator(anatomy_prior, w_obs=1.0, w_anat=1.0)
    sa_cfg = SpatialFKCConfig(beta_lo=0.0, beta_hi=2.0, n_blocks_per_axis=1,
                              ess_threshold_frac=0.5, do_resample=True)

    results = {}
    union_ids = sorted(set(expert_A_ids) | set(expert_B_ids))  # 42 ids (no body)
    K_scored = len(union_ids)
    K_anat = anatomy_prior.K
    print(f"  K_scored (no body) = {K_scored}; K_anat = {K_anat}")

    def mask_to_softonehot(mask_int, scored_ids):
        """(D,H,W) int → (K, D, H, W) float one-hot on scored_ids."""
        K = len(scored_ids)
        out = np.zeros((K, *mask_int.shape), dtype=np.float32)
        for k, sid in enumerate(scored_ids):
            out[k] = (mask_int == sid).astype(np.float32)
        return out

    for case in args.cases:
        ct = nib.load(str(DATA / "images" / f"{case}.nii.gz")).get_fdata().astype(np.float32)
        lu = nib.load(str(DATA / "labels_U" / f"{case}.nii.gz")).get_fdata().astype(np.int32)
        affine = nib.load(str(DATA / "images" / f"{case}.nii.gz")).affine
        la_base = build_aligned(lu, expert_A_ids)
        lb_base = build_aligned(lu, expert_B_ids)

        rng = np.random.default_rng(args.seed + hash(case) % 10000)
        cond_a_list, cond_b_list, m_soft_list = [], [], []
        for k in range(args.k_smc):
            la_k = per_particle_dropout_mask(la_base, expert_A_ids, args.drop_prob, rng)
            lb_k = per_particle_dropout_mask(lb_base, expert_B_ids, args.drop_prob, rng)
            cond_a_list.append(binarize_labels(torch.from_numpy(la_k).long().unsqueeze(0).unsqueeze(0)).float().to(device))
            cond_b_list.append(binarize_labels(torch.from_numpy(lb_k).long().unsqueeze(0).unsqueeze(0)).float().to(device))
            # M_soft for spatial energy: one-hot on aligned-union (no body), pad with body channel later
            merged = np.where(la_k != 0, la_k, lb_k)
            m_soft_list.append(mask_to_softonehot(merged, union_ids))
        cond_a = torch.cat(cond_a_list, dim=0)  # (K, 8, D, H, W)
        cond_b = torch.cat(cond_b_list, dim=0)
        m_soft_p = torch.from_numpy(np.stack(m_soft_list)).to(device)  # (K, K_scored, D, H, W)
        # Pad with zero body channel to match anatomy_prior.K
        if m_soft_p.shape[1] < K_anat:
            pad = torch.zeros((args.k_smc, K_anat - m_soft_p.shape[1], *m_soft_p.shape[2:]), device=device, dtype=m_soft_p.dtype)
            m_soft_p = torch.cat([m_soft_p, pad], dim=1)
        # Partial obs / presence at K_anat — use the COMMON aligned masks (intersection of observations)
        common_merged = np.where(la_base != 0, la_base, lb_base)
        partial_obs = torch.from_numpy(mask_to_softonehot(common_merged, union_ids)).unsqueeze(0).to(device)
        presence = (partial_obs > 0).float()
        if partial_obs.shape[1] < K_anat:
            pad = torch.zeros((1, K_anat - partial_obs.shape[1], *partial_obs.shape[2:]), device=device, dtype=partial_obs.dtype)
            partial_obs = torch.cat([partial_obs, pad], dim=1)
            presence = torch.cat([presence, pad], dim=1)

        uncond = binarize_labels(torch.zeros_like(torch.from_numpy(la_base)).long().unsqueeze(0).unsqueeze(0)).float().to(device)

        spacing = torch.tensor([[1.5, 1.5, 1.5]], device=device, dtype=torch.float32).repeat(args.k_smc, 1) * 100.0
        modality = torch.ones((args.k_smc,), device=device, dtype=torch.long)
        C, D, H, W = 4, cond_a.shape[2] // 4, cond_a.shape[3] // 4, cond_a.shape[4] // 4

        experts = [
            ControlNetExpert(name="A", controlnet=ctrl_a, controlnet_cond=cond_a),
            ControlNetExpert(name="B", controlnet=ctrl_b, controlnet_cond=cond_b),
        ]
        config = SamplerConfig(K_SMC=args.k_smc, n_steps=args.n_steps, composition="cfg_poe",
                                use_fkc_weights=True, seed=args.seed)

        print(f"\n[case] {case}: K_SMC={args.k_smc}, drop_prob={args.drop_prob}, β=[{sa_cfg.beta_lo},{sa_cfg.beta_hi}]")
        t0 = time.time()
        latents, log_w, diag = sample_sa_fkc_poe(
            experts, unet, noise_scheduler, scale_factor,
            spacing_tensor=spacing, modality_tensor=modality,
            latent_shape=(C, D, H, W), config=config,
            spatial_config=sa_cfg, M_soft_per_particle=m_soft_p,
            partial_obs=partial_obs, presence=presence,
            spatial_evaluator=sa_eval,
            uncond_controlnet_cond=uncond.repeat(args.k_smc, 1, 1, 1, 1),
            include_modality=True, include_body_region=False,
        )
        t_gen = time.time() - t0
        w = snis_weights(log_w)
        best_idx = int(torch.argmax(w).item())
        with torch.no_grad():
            best_latent = (latents[best_idx:best_idx + 1] / scale_factor).cpu()
            torch.cuda.empty_cache()
            ct_norm = ae.decode_stage_2_outputs(best_latent)
            ct_norm = torch.clip(ct_norm, 0.0, 1.0)
        gen = (ct_norm * 2000.0 - 1000.0).squeeze().cpu().numpy()
        ps, mae = psnr_mae(gen, ct)
        ess_mean = float(np.mean(diag.ess_curve))
        ess_min = float(np.min(diag.ess_curve))
        n_resamples = len(diag.resample_events)
        weights = w.detach().cpu().tolist()
        print(f"  PSNR={ps:.2f}dB ess_mean={ess_mean:.2f} ess_min={ess_min:.2f} n_resamples={n_resamples} best_p={best_idx}")
        print(f"  particle weights={weights}")
        results[case] = {
            "psnr": ps, "mae": mae, "time_s": t_gen,
            "ess_mean": ess_mean, "ess_min": ess_min,
            "n_resamples": n_resamples, "best_particle": best_idx,
            "weights": weights,
        }
        nib.save(nib.Nifti1Image(gen.astype(np.float32), affine), str(OUT / f"{case}_sa_fkc_K{args.k_smc}.nii.gz"))
        torch.cuda.empty_cache()

    out_p = REP / f"sa_fkc_k{args.k_smc}_summary.json"
    out_p.write_text(json.dumps(results, indent=2))
    print(f"\n[done] {out_p}")
    psnrs = [r["psnr"] for r in results.values()]
    ess_mins = [r["ess_min"] for r in results.values()]
    n_res = [r["n_resamples"] for r in results.values()]
    print(f"\nAggregate (n={len(psnrs)}): mean PSNR={np.mean(psnrs):.3f}±{np.std(psnrs):.3f}, "
          f"mean ESS_min={np.mean(ess_mins):.2f}, total resamples={sum(n_res)}")


if __name__ == "__main__":
    main()
