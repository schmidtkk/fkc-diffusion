#!/usr/bin/env python
"""Path 2 cross-dataset cfg_poe evaluation — the paper headline.

Pipeline (per TS-60 eval case):
  1. Load labels_U (42-organ TS union ground truth)
  2. Build cond_A = AMOS22-vocab subset of labels_U + body silhouette 200
  3. Build cond_B = CTSpine1K-vocab subset of labels_U + body silhouette 200
  4. Generate three variants:
     a. cfg_poe(A, B)  — combined experts via CFG-style PoE
     b. residual_sum(A, B) — naive residual sum baseline
     c. Single Expert A on cond_A (ceiling for abd-only)
     d. Single Expert B on cond_B (ceiling for spine-only)
  5. Segment each synth CT with:
     - Segmenter_A → abdominal Dice vs cond_A organs (MAISI IDs 1, 3–15)
     - Segmenter_B → spine Dice vs cond_B organs (MAISI IDs 33–56)
  6. Aggregate per-region.

The headline question: does composing the two single-dataset experts via cfg_poe
match (or beat) the single-expert sanity ceiling per region?

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_path2_cfg_poe.py \
      [--methods cfg_poe,residual_sum,single_A,single_B] [--n-cases 60] [--no-segB]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import SimpleITK as sitk

REPO = Path(__file__).resolve().parents[1]
MAISI = REPO / "maisi"
sys.path.insert(0, str(MAISI))
sys.path.insert(0, str(REPO))

from scripts.utils import define_instance, binarize_labels  # noqa: E402
from scripts.diff_model_setting import load_config  # noqa: E402
from src.cond_encoding import to_one_hot as _to_one_hot  # noqa: E402
from src.fkc_poe_sampler import ControlNetExpert, SamplerConfig, sample_fkc_poe_n  # noqa: E402

# === paths ===
DATA = REPO / "data/phase1_ts_iso15"  # TS eval data (labels_U at 1.5mm)
REP = REPO / "reports/verify_path1"
OUT = REPO / "outputs/eval_path2_cfg_poe"
OUT.mkdir(parents=True, exist_ok=True)

CKPT_A = REPO / "checkpoints/path2_expert_A_amos22/checkpoint_best.pt"
CKPT_B = REPO / "checkpoints/path2_expert_B_ctspine/checkpoint_best.pt"

SEG_A_FOLDER = REPO / "data/nnunet_results/Dataset800_AMOSAbd/nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres"
SEG_B_FOLDER = REPO / "data/nnunet_results/Dataset801_CTSpineVert/nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres"

# === vocabs ===
# AMOS22 abdominal MAISI IDs
ABD_MAISI_IDS = {1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
# CTSpine vertebrae MAISI IDs (C1=56..L5=33)
SPINE_MAISI_IDS = set(range(33, 57))  # 33..56

NN_A_TO_MAISI = {1: 1, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10,
                 10: 11, 11: 12, 12: 13, 13: 14, 14: 15}
NN_B_TO_MAISI = {k + 1: 56 - k for k in range(24)}  # NN 1=56, NN 2=55, ..., NN 24=33

MAISI_NAMES_ABD = {1: "liver", 3: "spleen", 4: "pancreas", 5: "R-kidney", 6: "aorta",
                   7: "IVC", 8: "R-adrenal", 9: "L-adrenal", 10: "gallbladder",
                   11: "esophagus", 12: "stomach", 13: "duodenum", 14: "L-kidney",
                   15: "bladder"}
MAISI_NAMES_SPINE = {56 - k: ("C" + str(k + 1) if k < 7 else
                              "T" + str(k - 6) if k < 19 else
                              "L" + str(k - 18)) for k in range(24)}


def build_cond(labels_U: np.ndarray, vocab: set, body_id: int = 200) -> np.ndarray:
    """Build a MAISI-style cond mask: kept-vocab organs at native IDs, all other body voxels at 200."""
    out = np.full_like(labels_U, body_id, dtype=np.int32)
    out[labels_U == 0] = 0
    for sid in vocab:
        out[labels_U == sid] = sid
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


def remap_nn(pn, nn_to_maisi):
    out = np.zeros_like(pn, dtype=np.int32)
    for nn_id, mid in nn_to_maisi.items():
        out[pn == nn_id] = mid
    return out


def make_uncond(cond):
    """Uncond cond = all-zero binarized labels."""
    return torch.zeros_like(cond)


def generate(experts, unet, ae, scale_factor, noise_scheduler, device,
             cond_for_uncond, composition, seed=1234, n_steps=30):
    spacing = torch.tensor([[1.5, 1.5, 1.5]], device=device, dtype=torch.float32) * 100.0
    modality = torch.ones((1,), device=device, dtype=torch.long)
    C = 4
    D = experts[0].controlnet_cond.shape[2] // 4
    H = experts[0].controlnet_cond.shape[3] // 4
    W = experts[0].controlnet_cond.shape[4] // 4

    uncond = None
    if composition == "cfg_poe":
        uncond = make_uncond(cond_for_uncond)

    scfg = SamplerConfig(K_SMC=1, n_steps=n_steps, composition=composition,
                         use_fkc_weights=False, seed=seed)
    latents, _, _ = sample_fkc_poe_n(
        experts, unet, noise_scheduler, scale_factor,
        spacing_tensor=spacing, modality_tensor=modality,
        latent_shape=(C, D, H, W), config=scfg,
        uncond_controlnet_cond=uncond,
        include_modality=True, include_body_region=False,
    )
    with torch.no_grad():
        best_latent = (latents[0:1] / scale_factor).cpu()
        torch.cuda.empty_cache()
        ct_norm = ae.decode_stage_2_outputs(best_latent)
        ct_norm = torch.clip(ct_norm, 0.0, 1.0)
    gen = (ct_norm * 2000.0 - 1000.0).squeeze().cpu().numpy()
    return gen


def segment_with(predictor, sitk_io, gen_arr, affine, tmpdir, key):
    """Save gen to tmp, read with SimpleITKIO, predict, return prediction in (X,Y,Z)."""
    p = tmpdir / f"_{key}.nii.gz"
    nib.save(nib.Nifti1Image(gen_arr.astype(np.float32), affine), str(p))
    data, props = sitk_io.read_images([str(p)])
    pn = predictor.predict_single_npy_array(
        input_image=data, image_properties=props,
        segmentation_previous_stage=None, save_or_return_probabilities=False,
    ).astype(np.int32)
    p.unlink(missing_ok=True)
    return pn.transpose(2, 1, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="cfg_poe,residual_sum,single_A,single_B",
                    help="comma-separated subset of cfg_poe,residual_sum,single_A,single_B")
    ap.add_argument("--n-cases", type=int, default=60, help="cap eval-set cases (max 60)")
    ap.add_argument("--no-segB", action="store_true",
                    help="skip Segmenter_B segmentation (use when Seg_B not converged)")
    ap.add_argument("--cond-encoding", choices=["binarize", "onehot"], default="binarize",
                    help="conditioning encoding; MUST match how the checkpoints were trained")
    ap.add_argument("--network-config", default=str(REPO / "configs/config_network_amos22_15ch.json"),
                    help="network config (use config_network_onehot43.json for onehot ckpts)")
    ap.add_argument("--ckpt-a", default=str(CKPT_A), help="Expert A controlnet checkpoint")
    ap.add_argument("--ckpt-b", default=str(CKPT_B), help="Expert B controlnet checkpoint")
    ap.add_argument("--tag", default="", help="suffix for output summary filename (e.g. scratch_onehot)")
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    ckpt_a, ckpt_b = Path(args.ckpt_a), Path(args.ckpt_b)
    gen_out_dir = OUT.parent / (OUT.name + (f"_{args.tag}" if args.tag else ""))
    gen_out_dir.mkdir(parents=True, exist_ok=True)

    if args.cond_encoding == "onehot":
        def encode_cond(arr):
            return _to_one_hot(torch.from_numpy(arr).long().unsqueeze(0).unsqueeze(0).to(device))
    else:
        def encode_cond(arr):
            return binarize_labels(torch.from_numpy(arr).long().unsqueeze(0).unsqueeze(0).to(device)).float()

    device = torch.device("cuda:0")

    print(f"[info] methods={methods}  n_cases≤{args.n_cases}  use_seg_B={not args.no_segB}  "
          f"encoding={args.cond_encoding}  ckpt_a={ckpt_a}")
    if not ckpt_a.exists():
        raise SystemExit(f"Missing Expert A ckpt: {ckpt_a}")
    if not ckpt_b.exists() and any(m in ["cfg_poe", "residual_sum", "single_B"] for m in methods):
        raise SystemExit(f"Missing Expert B ckpt: {ckpt_b}")
    if not SEG_A_FOLDER.exists():
        raise SystemExit(f"Missing Segmenter_A: {SEG_A_FOLDER}")

    cfg = load_config(
        str(REPO / "configs/env_path2_expert_A_amos22.json"),
        str(REPO / "configs/config_expert_aligned_finetune.json"),
        args.network_config,
    )

    print("[load] UNet/AE/scheduler ...")
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

    print("[load] Expert A controlnet")
    ctrlA = define_instance(cfg, "controlnet_def").to(device)
    ctrlA.load_state_dict(torch.load(str(ckpt_a), map_location=device,
                                     weights_only=False)["controlnet_state_dict"], strict=True)
    ctrlA.eval()
    for p in ctrlA.parameters(): p.requires_grad_(False)

    ctrlB = None
    if any(m in ["cfg_poe", "residual_sum", "single_B"] for m in methods):
        print("[load] Expert B controlnet")
        ctrlB = define_instance(cfg, "controlnet_def").to(device)
        ctrlB.load_state_dict(torch.load(str(ckpt_b), map_location=device,
                                         weights_only=False)["controlnet_state_dict"], strict=True)
        ctrlB.eval()
        for p in ctrlB.parameters(): p.requires_grad_(False)

    print("[load] Segmenter_A nnU-Net")
    os.environ.setdefault("nnUNet_raw", str(REPO / "data/nnunet_raw"))
    os.environ.setdefault("nnUNet_preprocessed", str(REPO / "data/nnunet_preprocessed"))
    os.environ.setdefault("nnUNet_results", str(REPO / "data/nnunet_results"))
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    predictor_A = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                  perform_everything_on_device=True, device=device, verbose=False)
    for ck_name in ["checkpoint_final.pth", "checkpoint_best.pth", "checkpoint_latest.pth"]:
        try:
            predictor_A.initialize_from_trained_model_folder(
                model_training_output_dir=str(SEG_A_FOLDER), use_folds=(0,), checkpoint_name=ck_name)
            print(f"  Seg_A loaded {ck_name}")
            break
        except FileNotFoundError:
            continue

    predictor_B = None
    if SEG_B_FOLDER.exists() and not args.no_segB:
        try:
            predictor_B = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                          perform_everything_on_device=True, device=device, verbose=False)
            for ck_name in ["checkpoint_final.pth", "checkpoint_best.pth", "checkpoint_latest.pth"]:
                try:
                    predictor_B.initialize_from_trained_model_folder(
                        model_training_output_dir=str(SEG_B_FOLDER), use_folds=(0,), checkpoint_name=ck_name)
                    print(f"  Seg_B loaded {ck_name}")
                    break
                except FileNotFoundError:
                    continue
        except Exception as e:
            print(f"  Seg_B unavailable: {e}")
            predictor_B = None

    sitk_io = SimpleITKIO()
    tmpdir = Path(tempfile.mkdtemp(prefix="path2_cfgpoe_"))

    eval_meta = json.load(open(DATA / "eval_dataset.json"))["training"]
    eval_meta = eval_meta[:args.n_cases]
    print(f"[info] evaluating {len(eval_meta)} TS held-out cases")

    # Per-method, per-region accumulators
    results = {m: {"abd_per_case": [], "spine_per_case": [],
                   "abd_per_organ": defaultdict(list),
                   "spine_per_organ": defaultdict(list)} for m in methods}

    for i, ent in enumerate(eval_meta):
        case = ent.get("ts_case", Path(ent["image"]).name.replace("_emb.nii.gz", ""))
        case_id = f"case_eval_{i:02d}"
        print(f"\n[case {i+1}/{len(eval_meta)}] {case_id} ({case})")

        # Load labels_U from disk
        lbl_p = DATA / ent["label"]
        lbl_arr = nib.load(str(lbl_p)).get_fdata().astype(np.int32)
        affine = nib.load(str(lbl_p)).affine

        # Build cond_A and cond_B
        cond_A_arr = build_cond(lbl_arr, ABD_MAISI_IDS, body_id=200)
        cond_B_arr = build_cond(lbl_arr, SPINE_MAISI_IDS, body_id=200)
        cond_A = encode_cond(cond_A_arr)
        cond_B = encode_cond(cond_B_arr)

        ids_abd = [s for s in ABD_MAISI_IDS if (lbl_arr == s).any()]
        ids_spine = [s for s in SPINE_MAISI_IDS if (lbl_arr == s).any()]

        for method in methods:
            out_p = gen_out_dir / f"{case_id}_{method}.nii.gz"
            if out_p.exists():
                # Resume: skip generation, load saved synth CT
                print(f"    [{method}] synth exists, re-segmenting only")
                gen = nib.load(str(out_p)).get_fdata().astype(np.float32)
                gen_time = 0.0
            else:
                t0 = time.time()
                if method == "cfg_poe":
                    experts = [ControlNetExpert(name="A", controlnet=ctrlA, controlnet_cond=cond_A),
                               ControlNetExpert(name="B", controlnet=ctrlB, controlnet_cond=cond_B)]
                    gen = generate(experts, unet, ae, scale_factor, noise_scheduler, device,
                                   cond_A, "cfg_poe")
                elif method == "residual_sum":
                    experts = [ControlNetExpert(name="A", controlnet=ctrlA, controlnet_cond=cond_A),
                               ControlNetExpert(name="B", controlnet=ctrlB, controlnet_cond=cond_B)]
                    gen = generate(experts, unet, ae, scale_factor, noise_scheduler, device,
                                   cond_A, "residual_sum")
                elif method == "single_A":
                    experts = [ControlNetExpert(name="A", controlnet=ctrlA, controlnet_cond=cond_A)]
                    gen = generate(experts, unet, ae, scale_factor, noise_scheduler, device,
                                   cond_A, "residual_sum")
                elif method == "single_B":
                    experts = [ControlNetExpert(name="B", controlnet=ctrlB, controlnet_cond=cond_B)]
                    gen = generate(experts, unet, ae, scale_factor, noise_scheduler, device,
                                   cond_B, "residual_sum")
                else:
                    raise ValueError(method)
                gen_time = time.time() - t0
                nib.save(nib.Nifti1Image(gen.astype(np.float32), affine), str(out_p))

            # Segment + score
            t0 = time.time()
            pn_A = segment_with(predictor_A, sitk_io, gen, affine, tmpdir, f"{case_id}_{method}_A")
            pm_A = remap_nn(pn_A, NN_A_TO_MAISI)
            d_abd = dice_per_label(pm_A, lbl_arr, ids_abd)
            cf_abd = [v for v in d_abd.values() if not np.isnan(v)]
            mean_abd = float(np.mean(cf_abd)) if cf_abd else float("nan")
            results[method]["abd_per_case"].append(mean_abd)
            for sid, d in d_abd.items():
                if not np.isnan(d):
                    results[method]["abd_per_organ"][sid].append(d)

            mean_spine = float("nan")
            if predictor_B is not None:
                pn_B = segment_with(predictor_B, sitk_io, gen, affine, tmpdir, f"{case_id}_{method}_B")
                pm_B = remap_nn(pn_B, NN_B_TO_MAISI)
                d_spine = dice_per_label(pm_B, lbl_arr, ids_spine)
                cf_spine = [v for v in d_spine.values() if not np.isnan(v)]
                mean_spine = float(np.mean(cf_spine)) if cf_spine else float("nan")
                results[method]["spine_per_case"].append(mean_spine)
                for sid, d in d_spine.items():
                    if not np.isnan(d):
                        results[method]["spine_per_organ"][sid].append(d)
            seg_time = time.time() - t0

            print(f"    [{method}] gen={gen_time:.1f}s seg={seg_time:.1f}s  "
                  f"abd={mean_abd:.3f} spine={mean_spine:.3f}")
            torch.cuda.empty_cache()

    # Aggregate
    summary = {"n_cases": len(eval_meta), "methods": {}}
    print(f"\n\n========== Path 2 cross-dataset summary (n={len(eval_meta)}) ==========")
    for m, R in results.items():
        abd = R["abd_per_case"]
        spine = R["spine_per_case"]
        abd_mean = float(np.mean([a for a in abd if not np.isnan(a)])) if abd else float("nan")
        abd_std = float(np.std([a for a in abd if not np.isnan(a)])) if abd else float("nan")
        spine_mean = float(np.mean([s for s in spine if not np.isnan(s)])) if spine else float("nan")
        spine_std = float(np.std([s for s in spine if not np.isnan(s)])) if spine else float("nan")
        per_organ_abd = {int(sid): float(np.mean(v)) for sid, v in R["abd_per_organ"].items()}
        per_organ_spine = {int(sid): float(np.mean(v)) for sid, v in R["spine_per_organ"].items()}
        summary["methods"][m] = {
            "abd_mean": abd_mean, "abd_std": abd_std,
            "spine_mean": spine_mean, "spine_std": spine_std,
            "per_organ_abd": per_organ_abd, "per_organ_spine": per_organ_spine,
            "abd_per_case": abd, "spine_per_case": spine,
        }
        print(f"  {m:<14} abd={abd_mean:.3f}±{abd_std:.3f}  spine={spine_mean:.3f}±{spine_std:.3f}")

    _suffix = f"_{args.tag}" if args.tag else ""
    out_p = REP / f"eval_path2_cfg_poe_summary{_suffix}.json"
    out_p.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_p}")


if __name__ == "__main__":
    main()
