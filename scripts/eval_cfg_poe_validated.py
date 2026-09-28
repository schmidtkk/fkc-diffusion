#!/usr/bin/env python
"""Path 1 final cfg_poe + Oracle eval using VALIDATED checkpoints.

Uses:
  - Expert A: checkpoints/ts_expert_A_validated/checkpoint_best.pt
  - Expert B: checkpoints/ts_expert_B_validated/checkpoint_best.pt
  - Oracle:   checkpoints/ts_oracle_validated/checkpoint_best.pt

Generates cfg_poe + Oracle on n=11 (or N) cases, segments with Mixed-DA,
computes Dice_CF with the CORRECT cond_v3 = labels_U. Then compares
cfg_poe vs Oracle.

Headlines this generates:
  - cfg_poe_validated Dice_CF mean ± std
  - Oracle_validated Dice_CF mean ± std
  - Paired Δ + wins
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
MAISI = REPO / "maisi"
sys.path.insert(0, str(MAISI))
sys.path.insert(0, str(REPO))

from scripts.utils import define_instance, binarize_labels  # noqa: E402
from scripts.diff_model_setting import load_config  # noqa: E402
from src.fkc_poe_sampler import ControlNetExpert, SamplerConfig, sample_fkc_poe_n  # noqa: E402

DATA = REPO / "data/phase1_ts_iso15"
REP = REPO / "reports/verify_path1"
DA_MODEL_FOLDER = REPO / "data/nnunet_results/Dataset703_MixedDA/nnUNetTrainer_50epochs__nnUNetPlans__3d_fullres"
MANIFEST = REPO / "data/nnunet_dataset_manifest.json"

NEW_CKPT_A = REPO / "checkpoints/ts_expert_A_validated/checkpoint_best.pt"
NEW_CKPT_B = REPO / "checkpoints/ts_expert_B_validated/checkpoint_best.pt"
NEW_CKPT_O = REPO / "checkpoints/ts_oracle_validated/checkpoint_best.pt"

OUT_DIR = REPO / "outputs/eval_validated_v4"


def build_aligned(labels_U, keep_ids):
    out = np.full_like(labels_U, 200, dtype=np.int32)
    body = labels_U != 0
    out[~body] = 0
    for sid in keep_ids:
        out[labels_U == sid] = sid
    return out


def build_correct_cond_v3(lu, expert_A_ids, expert_B_ids):
    out = np.full_like(lu, 200, dtype=np.int32)
    body = lu != 0
    out[~body] = 0
    for sid in set(expert_A_ids) | set(expert_B_ids):
        out[lu == sid] = sid
    return out


def dice_per_label(pred, target, ids):
    out = {}
    for sid in ids:
        p = pred == sid; t = target == sid
        if not t.any() and not p.any():
            out[sid] = float("nan"); continue
        inter = int((p & t).sum()); denom = int(p.sum() + t.sum())
        out[sid] = float(2 * inter / max(denom, 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cases", type=int, default=11)
    ap.add_argument("--n-steps", type=int, default=30)
    ap.add_argument("--cases-file", default="aggregate_filtered_n11.json",
                    help="JSON in reports/verify_path1/ with 'clean_cases' list.")
    ap.add_argument("--out-suffix", default="",
                    help="Suffix for output dir + summary json (e.g. '_n50').")
    args = ap.parse_args()

    for ck in (NEW_CKPT_A, NEW_CKPT_B, NEW_CKPT_O):
        if not ck.exists():
            raise SystemExit(f"missing: {ck}")

    OUT_DIR = REPO / f"outputs/eval_validated_v4{args.out_suffix}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    cfg = load_config(
        str(REPO / "configs/env_ts_expert_A.json"),
        str(REPO / "configs/config_expert_long_60ep.json"),
        str(REPO / "configs/config_network_amos22_15ch.json"),
    )

    print("[load] UNet/AE/CN ...")
    unet = define_instance(cfg, "diffusion_unet_def").to(device)
    sd = torch.load(cfg.trained_diffusion_path, map_location=device, weights_only=False)
    unet.load_state_dict(sd["unet_state_dict"], strict=False)
    scale_factor = sd["scale_factor"]
    scale_factor = scale_factor.to(device) if hasattr(scale_factor, "to") else torch.tensor(scale_factor).to(device)
    unet.eval()
    for p in unet.parameters(): p.requires_grad_(False)

    ae = define_instance(cfg, "autoencoder_def").cpu()
    ae_sd = torch.load(cfg.trained_autoencoder_path, map_location="cpu", weights_only=False)
    if isinstance(ae_sd, dict) and "unet_state_dict" in ae_sd:
        ae_sd = ae_sd["unet_state_dict"]
    ae.load_state_dict(ae_sd)
    ae.eval()
    for p in ae.parameters(): p.requires_grad_(False)
    for m in ae.modules():
        if hasattr(m, "norm_float16"): m.norm_float16 = False
    noise_scheduler = define_instance(cfg, "noise_scheduler")

    ctrl_a = define_instance(cfg, "controlnet_def").to(device)
    ctrl_a.load_state_dict(torch.load(str(NEW_CKPT_A), map_location=device, weights_only=False)["controlnet_state_dict"], strict=True)
    ctrl_a.eval()
    for p in ctrl_a.parameters(): p.requires_grad_(False)

    ctrl_b = define_instance(cfg, "controlnet_def").to(device)
    ctrl_b.load_state_dict(torch.load(str(NEW_CKPT_B), map_location=device, weights_only=False)["controlnet_state_dict"], strict=True)
    ctrl_b.eval()
    for p in ctrl_b.parameters(): p.requires_grad_(False)

    ctrl_o = define_instance(cfg, "controlnet_def").to(device)
    ctrl_o.load_state_dict(torch.load(str(NEW_CKPT_O), map_location=device, weights_only=False)["controlnet_state_dict"], strict=True)
    ctrl_o.eval()
    for p in ctrl_o.parameters(): p.requires_grad_(False)

    # Mixed-DA segmenter
    print("[load] Mixed-DA nnU-Net")
    os.environ.setdefault("nnUNet_raw", str(REPO / "data/nnunet_raw"))
    os.environ.setdefault("nnUNet_preprocessed", str(REPO / "data/nnunet_preprocessed"))
    os.environ.setdefault("nnUNet_results", str(REPO / "data/nnunet_results"))
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                                perform_everything_on_device=True, device=device, verbose=False)
    try:
        predictor.initialize_from_trained_model_folder(
            model_training_output_dir=str(DA_MODEL_FOLDER), use_folds=(0,), checkpoint_name="checkpoint_final.pth")
    except FileNotFoundError:
        predictor.initialize_from_trained_model_folder(
            model_training_output_dir=str(DA_MODEL_FOLDER), use_folds=(0,), checkpoint_name="checkpoint_best.pth")

    manifest = json.loads(MANIFEST.read_text())
    nn_to_maisi = {int(k): int(v) for k, v in manifest["nnunet_to_maisi"].items()}
    maisi_to_nn = {int(k): int(v) for k, v in manifest["maisi_to_nnunet"].items()}
    evaluable_ids = sorted(maisi_to_nn.keys())
    partition = json.loads((REPO / "data/label_partition.json").read_text())
    expert_A_ids = partition["expert_A_abdominal"]
    expert_B_ids = partition["expert_B_spine"]

    def remap(pred_nn):
        out = np.zeros_like(pred_nn, dtype=np.int32)
        for nn_id, mid in nn_to_maisi.items():
            out[pred_nn == int(nn_id)] = int(mid)
        return out

    agg = json.loads((REP / args.cases_file).read_text())
    cases = agg["clean_cases"][:args.n_cases]
    print(f"[info] {len(cases)} cases from {args.cases_file}")

    def gen_cfg_poe(la, lb):
        cond_a = binarize_labels(torch.from_numpy(la).long().unsqueeze(0).unsqueeze(0).to(device)).float()
        cond_b = binarize_labels(torch.from_numpy(lb).long().unsqueeze(0).unsqueeze(0).to(device)).float()
        cond_un = torch.zeros_like(cond_a)
        spacing = torch.tensor([[1.5, 1.5, 1.5]], device=device, dtype=torch.float32) * 100.0
        modality = torch.ones((1,), device=device, dtype=torch.long)
        C, D, H, W = 4, cond_a.shape[2] // 4, cond_a.shape[3] // 4, cond_a.shape[4] // 4
        experts = [
            ControlNetExpert(name="A", controlnet=ctrl_a, controlnet_cond=cond_a),
            ControlNetExpert(name="B", controlnet=ctrl_b, controlnet_cond=cond_b),
        ]
        scfg = SamplerConfig(K_SMC=1, n_steps=args.n_steps, composition="cfg_poe",
                             use_fkc_weights=False, cfg_scale=1.0, seed=1234)
        latents, _, _ = sample_fkc_poe_n(
            experts, unet, noise_scheduler, scale_factor,
            spacing_tensor=spacing, modality_tensor=modality,
            latent_shape=(C, D, H, W), config=scfg,
            uncond_controlnet_cond=cond_un,
            include_modality=True, include_body_region=False,
        )
        with torch.no_grad():
            best_latent = (latents[0:1] / scale_factor).cpu()
            torch.cuda.empty_cache()
            ct_norm = ae.decode_stage_2_outputs(best_latent)
            ct_norm = torch.clip(ct_norm, 0.0, 1.0)
        return (ct_norm * 2000.0 - 1000.0).squeeze().cpu().numpy()

    def gen_oracle(lu_mask):
        cond_u = binarize_labels(torch.from_numpy(lu_mask).long().unsqueeze(0).unsqueeze(0).to(device)).float()
        spacing = torch.tensor([[1.5, 1.5, 1.5]], device=device, dtype=torch.float32) * 100.0
        modality = torch.ones((1,), device=device, dtype=torch.long)
        C, D, H, W = 4, cond_u.shape[2] // 4, cond_u.shape[3] // 4, cond_u.shape[4] // 4
        experts = [ControlNetExpert(name="O", controlnet=ctrl_o, controlnet_cond=cond_u)]
        scfg = SamplerConfig(K_SMC=1, n_steps=args.n_steps, composition="residual_sum",
                             use_fkc_weights=False, seed=1234)
        latents, _, _ = sample_fkc_poe_n(
            experts, unet, noise_scheduler, scale_factor,
            spacing_tensor=spacing, modality_tensor=modality,
            latent_shape=(C, D, H, W), config=scfg,
            uncond_controlnet_cond=None,
            include_modality=True, include_body_region=False,
        )
        with torch.no_grad():
            best_latent = (latents[0:1] / scale_factor).cpu()
            torch.cuda.empty_cache()
            ct_norm = ae.decode_stage_2_outputs(best_latent)
            ct_norm = torch.clip(ct_norm, 0.0, 1.0)
        return (ct_norm * 2000.0 - 1000.0).squeeze().cpu().numpy()

    results = {"cfg_poe": {}, "oracle": {}}

    for case in cases:
        lu = nib.load(str(DATA / "labels_U" / f"{case}.nii.gz")).get_fdata().astype(np.int32)
        affine = nib.load(str(DATA / "labels_U" / f"{case}.nii.gz")).affine
        la_aligned = build_aligned(lu, expert_A_ids)
        lb_aligned = build_aligned(lu, expert_B_ids)
        cond_v3 = build_correct_cond_v3(lu, expert_A_ids, expert_B_ids)
        ids_in_cond = [s for s in evaluable_ids if (cond_v3 == s).any()]
        ids_in_ora = [s for s in evaluable_ids if (lu == s).any()]

        # cfg_poe
        t0 = time.time()
        gen_cp = gen_cfg_poe(la_aligned, lb_aligned)
        nib.save(nib.Nifti1Image(gen_cp.astype(np.float32), affine), str(OUT_DIR / f"{case}_cfg_poe.nii.gz"))
        print(f"  [cfg_poe] {case} gen={time.time()-t0:.1f}s", end="")

        pred_nn = predictor.predict_single_npy_array(
            input_image=gen_cp[None].astype(np.float32),
            image_properties={"spacing": [1.5, 1.5, 1.5]},
            segmentation_previous_stage=None, save_or_return_probabilities=False)
        pred_m = remap(pred_nn)
        dcf = dice_per_label(pred_m, cond_v3, ids_in_cond)
        cf_vals = [v for v in dcf.values() if not np.isnan(v)]
        results["cfg_poe"][case] = {"dice_cf_mean": float(np.nanmean(cf_vals)) if cf_vals else float("nan"),
                                    "dice_cf_per_id": {str(k): float(v) for k, v in dcf.items()}}
        print(f"  Dice_CF={results['cfg_poe'][case]['dice_cf_mean']:.3f}")

        # oracle
        t0 = time.time()
        gen_or = gen_oracle(lu)
        nib.save(nib.Nifti1Image(gen_or.astype(np.float32), affine), str(OUT_DIR / f"{case}_oracle.nii.gz"))
        print(f"  [oracle]  {case} gen={time.time()-t0:.1f}s", end="")

        pred_nn = predictor.predict_single_npy_array(
            input_image=gen_or[None].astype(np.float32),
            image_properties={"spacing": [1.5, 1.5, 1.5]},
            segmentation_previous_stage=None, save_or_return_probabilities=False)
        pred_m = remap(pred_nn)
        dcf = dice_per_label(pred_m, lu, ids_in_ora)
        cf_vals = [v for v in dcf.values() if not np.isnan(v)]
        results["oracle"][case] = {"dice_cf_mean": float(np.nanmean(cf_vals)) if cf_vals else float("nan"),
                                   "dice_cf_per_id": {str(k): float(v) for k, v in dcf.items()}}
        print(f"  Dice_CF={results['oracle'][case]['dice_cf_mean']:.3f}")
        torch.cuda.empty_cache()

    out_p = REP / f"eval_validated_v4_summary{args.out_suffix}.json"
    out_p.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_p}")

    cfgvs = [v["dice_cf_mean"] for v in results["cfg_poe"].values()]
    oravs = [v["dice_cf_mean"] for v in results["oracle"].values()]
    print(f"\n=== Aggregate (validated retrains, n={len(cfgvs)}) ===")
    print(f"  cfg_poe : {np.mean(cfgvs):.3f} ± {np.std(cfgvs):.3f}")
    print(f"  oracle  : {np.mean(oravs):.3f} ± {np.std(oravs):.3f}")
    common = sorted(set(results["cfg_poe"].keys()) & set(results["oracle"].keys()))
    if common:
        diffs = [results["cfg_poe"][c]["dice_cf_mean"] - results["oracle"][c]["dice_cf_mean"] for c in common]
        wins = sum(1 for d in diffs if d > 0)
        print(f"  paired Δ = {np.mean(diffs):+.3f} ± {np.std(diffs):.3f}, cfg_poe wins {wins}/{len(diffs)}")


if __name__ == "__main__":
    main()
