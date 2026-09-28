#!/usr/bin/env python
"""Verify the FKC particle component: K_SMC=4 + fkc_weights=True vs K_SMC=1.

Uses fixed (aligned) labels A/B from labels_U (no registration issue).
For each case, runs cfg_poe at K_SMC=1 and K_SMC=4 (with FKC weights), then
picks the best-weight particle from K=4. Compares PSNR / Dice_CF on a small
subset to keep the test cheap.
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
from src.fkc_poe_sampler import ControlNetExpert, SamplerConfig, sample_fkc_poe_n, snis_weights  # noqa: E402


def build_aligned(labels_U: np.ndarray, keep_ids: list[int]) -> np.ndarray:
    keep_set = set(keep_ids)
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

DATA = REPO / "data/phase1_ts_iso15"
OUT = REPO / "outputs/verify_path1_fkc"
OUT.mkdir(parents=True, exist_ok=True)
REP = REPO / "reports/verify_path1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["case_0000", "case_0149", "case_0500",
                                                     "case_1050", "case_1062"])
    ap.add_argument("--k-smc", type=int, default=4)
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

    cfg = load_config(ENV_CFG, MODEL_CFG, NETWORK_CFG)
    print("[load] UNet/AE/CNs ...")
    unet = define_instance(cfg, "diffusion_unet_def").to(device)
    sd = torch.load(cfg.trained_diffusion_path, map_location=device, weights_only=False)
    unet.load_state_dict(sd["unet_state_dict"], strict=False)
    scale_factor = sd["scale_factor"]
    scale_factor = scale_factor.to(device) if hasattr(scale_factor, "to") else torch.tensor(scale_factor, device=device)
    unet.eval()
    for p in unet.parameters(): p.requires_grad_(False)

    ae = define_instance(cfg, "autoencoder_def").to(device)
    ae_sd = torch.load(cfg.trained_autoencoder_path, map_location=device, weights_only=False)
    if isinstance(ae_sd, dict) and "unet_state_dict" in ae_sd:
        ae_sd = ae_sd["unet_state_dict"]
    ae.load_state_dict(ae_sd)
    ae.eval()
    for p in ae.parameters(): p.requires_grad_(False)
    for m in ae.modules():
        if hasattr(m, "norm_float16"): m.norm_float16 = False
    noise_scheduler = define_instance(cfg, "noise_scheduler")

    ctrl_a = define_instance(cfg, "controlnet_def").to(device)
    ctrl_a.load_state_dict(torch.load(str(CKPT_A), map_location=device, weights_only=False)["controlnet_state_dict"], strict=True)
    ctrl_a.eval()
    for p in ctrl_a.parameters(): p.requires_grad_(False)
    ctrl_b = define_instance(cfg, "controlnet_def").to(device)
    ctrl_b.load_state_dict(torch.load(str(CKPT_B), map_location=device, weights_only=False)["controlnet_state_dict"], strict=True)
    ctrl_b.eval()
    for p in ctrl_b.parameters(): p.requires_grad_(False)

    def gen(mask_a, mask_b, k_smc, use_fkc):
        cond_a = binarize_labels(torch.from_numpy(mask_a).long().unsqueeze(0).unsqueeze(0).to(device)).float()
        cond_b = binarize_labels(torch.from_numpy(mask_b).long().unsqueeze(0).unsqueeze(0).to(device)).float()
        uncond = binarize_labels(torch.zeros_like(torch.from_numpy(mask_a)).long().unsqueeze(0).unsqueeze(0).to(device)).float()
        spacing = torch.tensor([[1.5, 1.5, 1.5]], device=device, dtype=torch.float32).repeat(k_smc, 1) * 100.0
        modality = torch.ones((k_smc,), device=device, dtype=torch.long)
        C, D, H, W = 4, cond_a.shape[2] // 4, cond_a.shape[3] // 4, cond_a.shape[4] // 4
        experts = [
            ControlNetExpert(name="A", controlnet=ctrl_a, controlnet_cond=cond_a),
            ControlNetExpert(name="B", controlnet=ctrl_b, controlnet_cond=cond_b),
        ]
        config = SamplerConfig(K_SMC=k_smc, n_steps=30, composition="cfg_poe",
                               use_fkc_weights=use_fkc, seed=1234)
        latents, log_w, diag = sample_fkc_poe_n(
            experts, unet, noise_scheduler, scale_factor,
            spacing_tensor=spacing, modality_tensor=modality,
            latent_shape=(C, D, H, W), config=config,
            uncond_controlnet_cond=uncond,
            include_modality=True, include_body_region=False,
        )
        # Pick the best particle by SNIS weights
        w = snis_weights(log_w)
        best_idx = int(torch.argmax(w).item())
        with torch.no_grad():
            ct_norm = ae.decode_stage_2_outputs(latents[best_idx:best_idx + 1] / scale_factor)
            ct_norm = torch.clip(ct_norm, 0.0, 1.0)
        return (ct_norm * 2000.0 - 1000.0).squeeze().cpu().numpy(), {
            "ess_curve": diag.ess_curve,
            "weights": w.detach().cpu().tolist(),
            "best_particle": best_idx,
        }

    results = {}
    for case in args.cases:
        ct = nib.load(str(DATA / "images" / f"{case}.nii.gz")).get_fdata().astype(np.float32)
        lu = nib.load(str(DATA / "labels_U" / f"{case}.nii.gz")).get_fdata().astype(np.int32)
        la_aligned = build_aligned(lu, expert_A_ids)
        lb_aligned = build_aligned(lu, expert_B_ids)
        affine = nib.load(str(DATA / "images" / f"{case}.nii.gz")).affine
        print(f"\n[case] {case}")
        case_res = {}
        # K_SMC=1 (baseline cfg_poe)
        t0 = time.time()
        g1, d1 = gen(la_aligned, lb_aligned, k_smc=1, use_fkc=False)
        p1, m1 = psnr_mae(g1, ct)
        t1 = time.time() - t0
        print(f"  K=1     PSNR={p1:.2f}dB time={t1:.1f}s")
        # K_SMC=4 (FKC + best particle)
        t0 = time.time()
        gk, dk = gen(la_aligned, lb_aligned, k_smc=args.k_smc, use_fkc=True)
        pk, mk = psnr_mae(gk, ct)
        tk = time.time() - t0
        ess_mean = float(np.mean(dk["ess_curve"]))
        ess_min = float(np.min(dk["ess_curve"]))
        print(f"  K={args.k_smc} FKC  PSNR={pk:.2f}dB time={tk:.1f}s  ess_mean={ess_mean:.2f}  ess_min={ess_min:.2f}  best_p={dk['best_particle']}")

        case_res = {
            "k1_psnr": p1, "k1_mae": m1, "k1_time_s": t1,
            "k4_psnr": pk, "k4_mae": mk, "k4_time_s": tk,
            "ess_mean": ess_mean, "ess_min": ess_min,
            "best_particle": dk["best_particle"],
            "weights": dk["weights"],
            "delta_psnr": pk - p1,
        }
        results[case] = case_res
        nib.save(nib.Nifti1Image(g1.astype(np.float32), affine), str(OUT / f"{case}_fixed_cfg_poe_K1.nii.gz"))
        nib.save(nib.Nifti1Image(gk.astype(np.float32), affine), str(OUT / f"{case}_fixed_cfg_poe_K{args.k_smc}_fkc.nii.gz"))

    out_p = REP / f"fkc_k{args.k_smc}_summary.json"
    out_p.write_text(json.dumps(results, indent=2))
    print(f"\n[done] {out_p}")

    # Aggregate
    deltas = [r["delta_psnr"] for r in results.values()]
    print(f"\nAggregate (n={len(deltas)}): mean ΔPSNR (K{args.k_smc}_fkc − K1) = {np.mean(deltas):+.3f} ± {np.std(deltas):.3f} dB")
    print(f"  median ess: {np.mean([r['ess_mean'] for r in results.values()]):.2f}  (full-K means no FK collapse)")


if __name__ == "__main__":
    main()
