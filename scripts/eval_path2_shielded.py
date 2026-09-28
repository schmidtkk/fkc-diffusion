#!/usr/bin/env python
"""MVP eval: Shielded Local Expert Composition vs naive cfg_poe (abd via Seg_A).

Tests whether the 5-module shield (private-zone shield + null-sub + ownership
gate/veto + RMS budget + time-sync) recovers the resident-region (abd) Dice that
naive cfg_poe loses (clean cfg_poe abd 0.338 < single_A 0.378).

Methods (same TS cases, same seed, contamination-clean one-hot ckpts):
  single_A      : Expert A only (residual_sum)            — resident-region ceiling
  naive_cfg_poe : Expert A+B cfg_poe                       — the broken composition
  gated         : shield OFF, gate+nullsub+budget+sync ON  — does gating alone help?
  shielded      : all 5 modules ON                          — the MVP

Segmented with Specialist Segmenter_A → abd Dice vs cond_A.

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/eval_path2_shielded.py --n-cases 20 \
      --methods single_A,naive_cfg_poe,gated,shielded
"""
from __future__ import annotations
import argparse, json, os, sys, tempfile, time
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "maisi"))
sys.path.insert(0, str(REPO))

from scripts.utils import define_instance  # noqa: E402
from scripts.diff_model_setting import load_config  # noqa: E402
from src.fkc_poe_sampler import (ControlNetExpert, SamplerConfig, sample_fkc_poe_n,  # noqa: E402
                                 ShieldConfig, sample_shielded_local, sample_region_inpaint)
from src.cond_encoding import to_one_hot, to_inpaint_cond  # noqa: E402
from src.shield_masks import build_shield_masks  # noqa: E402

DATA = REPO / "data/phase1_ts_iso15"
REP = REPO / "reports/verify_path1"
OUT = REPO / "outputs/eval_path2_cfg_poe_path1_indomain_v2"; OUT.mkdir(parents=True, exist_ok=True)  # co-locate with v2 CTs for union-proxy scoring

CKPT_A = REPO / "checkpoints/ts_expert_A_scratch_onehot_v2/checkpoint_best.pt"   # Path-1 v2 (abd-fixed)
CKPT_B = REPO / "checkpoints/ts_expert_B_scratch_onehot/checkpoint_best.pt"      # Path-1 spine
SEG_A = REPO / "data/nnunet_results/Dataset800_AMOSAbd/nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres"
NETCFG = REPO / "configs/config_network_onehot43.json"

ABD_IDS = {1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
SPINE_IDS = set(range(33, 57))
NN_A_TO_MAISI = {1:1,2:3,3:4,4:5,5:6,6:7,7:8,8:9,9:10,10:11,11:12,12:13,13:14,14:15}
NAMES = {1:"liver",3:"spleen",4:"pancreas",5:"R-kidney",6:"aorta",7:"IVC",8:"R-adrenal",
         9:"L-adrenal",10:"gallbladder",11:"esophagus",12:"stomach",13:"duodenum",14:"L-kidney",15:"bladder"}


def build_cond(labels_U, vocab, body_id=200):
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


def remap(pn):
    o = np.zeros_like(pn, dtype=np.int32)
    for nn, mid in NN_A_TO_MAISI.items():
        o[pn == nn] = mid
    return o


def decode(ae, latents, scale_factor):
    with torch.no_grad():
        lat = (latents[0:1] / scale_factor).cpu()
        torch.cuda.empty_cache()
        ct = torch.clip(ae.decode_stage_2_outputs(lat), 0.0, 1.0)
    return (ct * 2000.0 - 1000.0).squeeze().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cases", type=int, default=20)
    ap.add_argument("--case-start", type=int, default=0)           # absolute start index into eval split (for resuming partial sweeps)
    ap.add_argument("--methods", default="single_A,naive_cfg_poe,gated,shielded")
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--beta-sync", type=float, default=0.1)
    ap.add_argument("--gate-floor", type=float, default=1.0)
    ap.add_argument("--unc-power", type=float, default=2.0)        # uncertainty-PoE precision sharpness
    ap.add_argument("--unc-support-mix", type=float, default=0.5)  # uncertainty-PoE OOD robustifier
    ap.add_argument("--unc-smooth", type=int, default=3)           # uncertainty-PoE confidence-field smoothing kernel
    ap.add_argument("--no-unc-self-norm", action="store_true")     # disable per-expert confidence self-normalization
    ap.add_argument("--mc-samples", type=int, default=2)           # T2: MC latent-perturbation samples
    ap.add_argument("--mc-eps", type=float, default=0.08)          # T2: latent perturbation scale
    ap.add_argument("--mc-kappa", type=float, default=1.0)         # T2: outlier z-threshold for the confidence gate
    ap.add_argument("--mc-sharp", type=float, default=2.0)         # T2: sigmoid sharpness of the outlier gate
    ap.add_argument("--unc-nullsub", action="store_true")          # use d_k=v_c-v_n (null-sub) for the confidence signal
    ap.add_argument("--save-label", default="")                    # override filename token so many configs share one dir
    ap.add_argument("--cfg-scale", type=float, default=1.0)        # cfg_poe guidance scale: v_uncond + s*sum(deltas)
    ap.add_argument("--cfg-a", type=float, default=1.5)            # asym_cfg: abd-expert guidance scale
    ap.add_argument("--cfg-b", type=float, default=1.0)            # asym_cfg: spine-expert guidance scale
    ap.add_argument("--auto-s-max", type=float, default=1.5)       # auto_cfg: guidance for a confidently in-dist expert
    ap.add_argument("--auto-tau-lo", type=float, default=0.8)      # auto_cfg: rho_test/rho_base threshold -> OOD (s=1.0)
    ap.add_argument("--rho-base-a", type=float, default=1.24)      # auto_cfg: A expert in-dist own/foreign baseline
    ap.add_argument("--rho-base-b", type=float, default=2.41)      # auto_cfg: B expert in-dist own/foreign baseline
    ap.add_argument("--inpaint-tau", type=float, default=0.6)      # consensus_inpaint: Stage-2 re-noise level
    ap.add_argument("--inpaint-cfg", type=float, default=1.5)      # consensus_inpaint: Stage-2 per-expert guidance
    ap.add_argument("--inpaint-dilate", type=int, default=3)       # consensus_inpaint: latent dilation kernel (odd)
    ap.add_argument("--inpaint-mode", default="repaint", choices=["repaint","sdedit"])  # RF-RePaint (known-fix+resample) | pure SDEdit
    ap.add_argument("--inpaint-resample", type=int, default=5)     # RF-RePaint time-travel depth U
    ap.add_argument("--inpaint-eta", type=float, default=0.1)      # RF-RePaint stochastic-reverse strength
    ap.add_argument("--no-budget", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--no-shield", action="store_true")
    # FKC on/off ablation (publishability gap #3 / off-manifold test). Defaults = headline FKC-OFF
    # (deterministic guided-PoE). --fkc-eta>0 turns on the annealed-Langevin correction; --k-smc>1 +
    # --use-fkc-weights turns on the Feynman-Kac importance-resampling corrector (Skreta'25).
    ap.add_argument("--fkc-eta", type=float, default=0.0)         # Langevin noise strength (stochastic correction)
    ap.add_argument("--k-smc", type=int, default=1)              # SMC particles (>1 activates FK importance resampling)
    ap.add_argument("--use-fkc-weights", action="store_true")    # FK importance weights (only meaningful with --k-smc>1)
    ap.add_argument("--corrector-m", type=int, default=0)        # A1: # annealed-Langevin corrector steps per level (0=off)
    ap.add_argument("--corrector-c", type=float, default=0.05)   # A1: corrector step-size coeff, eta=c*s^2
    ap.add_argument("--corrector-s-lo", type=float, default=0.2) # A1: apply corrector for normalized level s in [lo,hi]
    ap.add_argument("--corrector-s-hi", type=float, default=0.8)
    ap.add_argument("--corrector-snr", type=float, default=0.0)  # A1: SNR-adaptive step (>0 overrides corrector-c)
    ap.add_argument("--per-expert-uncond", action="store_true")  # B0: per-expert uncond -> coherent deltas (no retraining)
    ap.add_argument("--soft-pull-gamma", type=float, default=0.1) # soft_pull method: regional velocity-blend strength (0=off)
    ap.add_argument("--tag", default="")
    ap.add_argument("--ckpt-a", default=str(CKPT_A))
    ap.add_argument("--ckpt-b", default=str(CKPT_B))
    ap.add_argument("--ckpt-a-inpaint", default="")            # trained_inpaint: 45-ch inpaint ControlNet, expert A
    ap.add_argument("--ckpt-b-inpaint", default="")            # trained_inpaint: 45-ch inpaint ControlNet, expert B
    ap.add_argument("--inpaint-net-config", default="configs/config_network_inpaint45.json")  # 45-ch network config for the inpaint experts
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")

    cfg = load_config(str(REPO / "configs/env_path2_expert_A_amos22.json"),
                      str(REPO / "configs/config_expert_aligned_finetune.json"), str(NETCFG))
    print("[load] UNet/AE/scheduler")
    unet = define_instance(cfg, "diffusion_unet_def").to(device)
    sd = torch.load(cfg.trained_diffusion_path, map_location=device, weights_only=False)
    unet.load_state_dict(sd["unet_state_dict"], strict=False)
    scale_factor = sd["scale_factor"]; scale_factor = scale_factor.to(device) if hasattr(scale_factor,"to") else torch.tensor(scale_factor).to(device)
    unet.eval()
    for p in unet.parameters(): p.requires_grad_(False)
    ae = define_instance(cfg, "autoencoder_def").cpu()
    ae_sd = torch.load(cfg.trained_autoencoder_path, map_location="cpu", weights_only=False)
    if isinstance(ae_sd, dict) and "unet_state_dict" in ae_sd: ae_sd = ae_sd["unet_state_dict"]
    ae.load_state_dict(ae_sd); ae.eval()
    for p in ae.parameters(): p.requires_grad_(False)
    for m in ae.modules():
        if hasattr(m, "norm_float16"): m.norm_float16 = False
    noise_scheduler = define_instance(cfg, "noise_scheduler")

    print("[load] Expert A, B (clean one-hot)")
    ctrlA = define_instance(cfg, "controlnet_def").to(device)
    ctrlA.load_state_dict(torch.load(str(args.ckpt_a), map_location=device, weights_only=False)["controlnet_state_dict"], strict=True)
    ctrlA.eval()
    for p in ctrlA.parameters(): p.requires_grad_(False)
    ctrlB = define_instance(cfg, "controlnet_def").to(device)
    ctrlB.load_state_dict(torch.load(str(args.ckpt_b), map_location=device, weights_only=False)["controlnet_state_dict"], strict=True)
    ctrlB.eval()
    for p in ctrlB.parameters(): p.requires_grad_(False)

    # --- trained_inpaint: load the two 45-ch inpaint ControlNets (if requested).
    # These MUST be instantiated with the 45-ch inpaint network config (conditioning_embedding_in_channels=45),
    # NOT the eval's default 43-ch one-hot config — otherwise the cond-embedding conv_in shape mismatches.
    ctrlA_inp = ctrlB_inp = None
    if args.ckpt_a_inpaint and args.ckpt_b_inpaint:
        print("[load] Expert A_inp, B_inp (45-ch trained inpaint)")
        inpaint_net = args.inpaint_net_config
        if not os.path.isabs(inpaint_net):
            inpaint_net = str(REPO / inpaint_net)
        cfg_inpaint = load_config(str(REPO / "configs/env_path2_expert_A_amos22.json"),
                                  str(REPO / "configs/config_expert_aligned_finetune.json"), inpaint_net)
        ctrlA_inp = define_instance(cfg_inpaint, "controlnet_def").to(device)
        ctrlA_inp.load_state_dict(torch.load(str(args.ckpt_a_inpaint), map_location=device, weights_only=False)["controlnet_state_dict"], strict=True)
        ctrlA_inp.eval()
        for p in ctrlA_inp.parameters(): p.requires_grad_(False)
        ctrlB_inp = define_instance(cfg_inpaint, "controlnet_def").to(device)
        ctrlB_inp.load_state_dict(torch.load(str(args.ckpt_b_inpaint), map_location=device, weights_only=False)["controlnet_state_dict"], strict=True)
        ctrlB_inp.eval()
        for p in ctrlB_inp.parameters(): p.requires_grad_(False)

    print("[load] Segmenter_A")
    os.environ.setdefault("nnUNet_raw", str(REPO / "data/nnunet_raw"))
    os.environ.setdefault("nnUNet_preprocessed", str(REPO / "data/nnunet_preprocessed"))
    os.environ.setdefault("nnUNet_results", str(REPO / "data/nnunet_results"))
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                perform_everything_on_device=True, device=device, verbose=False)
    for ck in ["checkpoint_final.pth", "checkpoint_best.pth"]:
        try:
            predictor.initialize_from_trained_model_folder(str(SEG_A), use_folds=(0,), checkpoint_name=ck); break
        except FileNotFoundError: continue
    sitk_io = SimpleITKIO(); tmpdir = Path(tempfile.mkdtemp(prefix="shield_"))

    spacing = torch.tensor([[1.5,1.5,1.5]], device=device, dtype=torch.float32) * 100.0
    modality = torch.ones((1,), device=device, dtype=torch.long)

    eval_meta = json.load(open(DATA / "eval_dataset.json"))["training"][args.case_start:args.case_start + args.n_cases]
    print(f"[info] {len(eval_meta)} cases (start={args.case_start})  methods={methods}")
    results = {m: {"per_case": [], "per_organ": defaultdict(list)} for m in methods}

    for i, ent in enumerate(eval_meta, start=args.case_start):
        cid = f"case_eval_{i:02d}"
        lbl = nib.load(str(DATA / ent["label"])).get_fdata().astype(np.int32)
        affine = nib.load(str(DATA / ent["label"])).affine
        ids_abd = [s for s in ABD_IDS if (lbl == s).any()]
        condA_arr = build_cond(lbl, ABD_IDS); condB_arr = build_cond(lbl, SPINE_IDS)
        condA = to_one_hot(torch.from_numpy(condA_arr).long().unsqueeze(0).unsqueeze(0).to(device))
        condB = to_one_hot(torch.from_numpy(condB_arr).long().unsqueeze(0).unsqueeze(0).to(device))
        uncond = torch.zeros_like(condA)
        C, D, H, W = 4, condA.shape[2]//4, condA.shape[3]//4, condA.shape[4]//4
        g_A, g_B, P_A, P_B = build_shield_masks(condA_arr, condB_arr, ABD_IDS, SPINE_IDS,
                                                latent_dim=D, device=device)
        print(f"\n[case {i+1}/{len(eval_meta)}] {cid}  abd ids={sorted(ids_abd)}")

        for method in methods:
            t0 = time.time()
            gen = None   # composite methods (consensus_inpaint) set gen directly; others set lat then decode below
            if method == "single_A":
                experts = [ControlNetExpert("A", ctrlA, condA)]
                scfg = SamplerConfig(K_SMC=1, n_steps=30, composition="residual_sum", use_fkc_weights=False, seed=1234)
                lat, _, _ = sample_fkc_poe_n(experts, unet, noise_scheduler, scale_factor, spacing, modality,
                                             (C,D,H,W), scfg, uncond_controlnet_cond=None,
                                             include_modality=True, include_body_region=False)
            elif method == "consensus_inpaint":
                # Two-stage: (1) PoE consensus; (2) each expert refines its own dilated region
                # (RF-RePaint x0-space known-fix+resample, OR pure SDEdit); composite in IMAGE space.
                from scipy.ndimage import binary_dilation
                experts = [ControlNetExpert("A", ctrlA, condA), ControlNetExpert("B", ctrlB, condB)]
                scfg = SamplerConfig(K_SMC=1, n_steps=30, composition="cfg_poe", use_fkc_weights=False,
                                     cfg_scale=1.0, seed=1234)
                lat_cons, _, _ = sample_fkc_poe_n(experts, unet, noise_scheduler, scale_factor, spacing, modality,
                                                  (C,D,H,W), scfg, uncond_controlnet_cond=uncond,
                                                  include_modality=True, include_body_region=False)
                z_cons = lat_cons[0:1]
                def _lm(b):
                    tt = torch.from_numpy(b.astype(np.float32))[None,None].to(device)
                    return torch.nn.functional.max_pool3d(tt, 4, 4)[0,0]
                kk = args.inpaint_dilate
                def _dil(m):
                    return torch.nn.functional.max_pool3d(m[None,None], kernel_size=kk, stride=1, padding=kk//2)[0,0]
                RA = _dil(_lm(np.isin(condA_arr, list(ABD_IDS))))
                RB = _dil(_lm(np.isin(condB_arr, list(SPINE_IDS))))
                rep = (args.inpaint_mode == "repaint")
                ip_kw = dict(n_steps=30, cfg_scale=args.inpaint_cfg, uncond_controlnet_cond=uncond, seed=1234,
                             tau_re=args.inpaint_tau, known_fix=rep,
                             resample_steps=(args.inpaint_resample if rep else 1),
                             eta=(args.inpaint_eta if rep else 0.0),
                             include_modality=True, include_body_region=False)
                zA = sample_region_inpaint(experts[0], unet, noise_scheduler, scale_factor, spacing, modality,
                                           (C,D,H,W), z_cons=z_cons, region_mask=RA, **ip_kw)
                zB = sample_region_inpaint(experts[1], unet, noise_scheduler, scale_factor, spacing, modality,
                                           (C,D,H,W), z_cons=z_cons, region_mask=RB, **ip_kw)
                # IMAGE-space composite (decode each; overwrite by full-res organ region)
                ctC = decode(ae, z_cons, scale_factor); ctA = decode(ae, zA, scale_factor); ctB = decode(ae, zB, scale_factor)
                abd_m = binary_dilation(np.isin(condA_arr, list(ABD_IDS)), iterations=kk)
                spn_m = binary_dilation(np.isin(condB_arr, list(SPINE_IDS)), iterations=kk) & (~abd_m)
                gen = ctC.copy(); gen[abd_m] = ctA[abd_m]; gen[spn_m] = ctB[spn_m]
            elif method == "trained_inpaint":
                # Two-stage like consensus_inpaint, but the Stage-2 refiner is each TRAINED 45-ch inpaint
                # expert (label one-hot | masked-CT | mask R) running a single-expert cfg_poe generation
                # conditioned on the masked PoE-consensus image — NOT RF-RePaint on the gen experts.
                if ctrlA_inp is None or ctrlB_inp is None:
                    raise ValueError("trained_inpaint requires --ckpt-a-inpaint and --ckpt-b-inpaint")
                from scipy.ndimage import binary_dilation
                # Stage 1: PoE consensus latent (identical to consensus_inpaint)
                experts = [ControlNetExpert("A", ctrlA, condA), ControlNetExpert("B", ctrlB, condB)]
                scfg = SamplerConfig(K_SMC=1, n_steps=30, composition="cfg_poe", use_fkc_weights=False,
                                     cfg_scale=1.0, seed=1234)
                lat_cons, _, _ = sample_fkc_poe_n(experts, unet, noise_scheduler, scale_factor, spacing, modality,
                                                  (C,D,H,W), scfg, uncond_controlnet_cond=uncond,
                                                  include_modality=True, include_body_region=False)
                z_cons = lat_cons[0:1]
                # Stage 2 setup: decode consensus to HU, renormalize to the masked-CT [0,1] convention (matches
                # load_ct_norm in train_controlnet_inpaint.py: clip((HU+1000)/2000,0,1)).
                ct_cons = decode(ae, z_cons, scale_factor)                                   # (D,H,W) HU numpy
                ct_norm = np.clip((ct_cons + 1000.0) / 2000.0, 0.0, 1.0).astype(np.float32)
                ct_norm_t = torch.from_numpy(ct_norm)[None, None].to(device)                 # (1,1,D,H,W)
                dd = int(args.inpaint_dilate)

                def _inpaint_z(ctrl_inp, cond_arr, id_set, name):
                    # Full-res organ mask -> fixed dilation R (NO random jitter at inference); masked-CT; 45-ch cond.
                    organ = torch.from_numpy(np.isin(cond_arr, list(id_set)).astype(np.float32))[None, None].to(device)
                    R = torch.nn.functional.max_pool3d(organ, kernel_size=2 * dd + 1, stride=1, padding=dd)
                    R = (R > 0.5).float()
                    masked_ct = ct_norm_t * (1.0 - R)
                    label = torch.from_numpy(cond_arr).long().unsqueeze(0).unsqueeze(0).to(device)
                    cond_inp = to_inpaint_cond(label, masked_ct, R)                          # (1,45,D_full,H_full,W_full)
                    uncond_inp = torch.zeros_like(cond_inp)                                  # cfg uncond = 45-ch zeros
                    expert = ControlNetExpert(name, ctrl_inp, cond_inp)
                    scfg_inp = SamplerConfig(K_SMC=1, n_steps=30, composition="cfg_poe", use_fkc_weights=False,
                                             cfg_scale=args.inpaint_cfg, seed=1234)
                    lat_k, _, _ = sample_fkc_poe_n([expert], unet, noise_scheduler, scale_factor, spacing, modality,
                                                   (C,D,H,W), scfg_inp, uncond_controlnet_cond=uncond_inp,
                                                   include_modality=True, include_body_region=False)
                    return lat_k[0:1]

                zA = _inpaint_z(ctrlA_inp, condA_arr, ABD_IDS, "A_inp")
                zB = _inpaint_z(ctrlB_inp, condB_arr, SPINE_IDS, "B_inp")
                # IMAGE-space composite (identical to consensus_inpaint's final lines)
                kk = args.inpaint_dilate
                ctA = decode(ae, zA, scale_factor); ctB = decode(ae, zB, scale_factor)
                abd_m = binary_dilation(np.isin(condA_arr, list(ABD_IDS)), iterations=kk)
                spn_m = binary_dilation(np.isin(condB_arr, list(SPINE_IDS)), iterations=kk) & (~abd_m)
                gen = ct_cons.copy(); gen[abd_m] = ctA[abd_m]; gen[spn_m] = ctB[spn_m]
            elif method == "naive_mean":
                # LINEAR-WEIGHTING baseline: convex average of conditional velocities (1/K)Σ v_ck.
                # The "geometric barycenter" of A7 (NOT a product) — predicted to blur disjoint conditions.
                experts = [ControlNetExpert("A", ctrlA, condA), ControlNetExpert("B", ctrlB, condB)]
                scfg = SamplerConfig(K_SMC=1, n_steps=30, composition="naive_mean", use_fkc_weights=False, seed=1234)
                lat, _, _ = sample_fkc_poe_n(experts, unet, noise_scheduler, scale_factor, spacing, modality,
                                             (C,D,H,W), scfg, uncond_controlnet_cond=None,
                                             include_modality=True, include_body_region=False)
            elif method == "naive_cfg_poe":
                experts = [ControlNetExpert("A", ctrlA, condA), ControlNetExpert("B", ctrlB, condB)]
                scfg = SamplerConfig(K_SMC=args.k_smc, n_steps=30, composition="cfg_poe",
                                     use_fkc_weights=args.use_fkc_weights, fkc_eta=args.fkc_eta,
                                     cfg_scale=args.cfg_scale, seed=1234,
                                     corrector_M=args.corrector_m, corrector_c=args.corrector_c,
                                     corrector_s_lo=args.corrector_s_lo, corrector_s_hi=args.corrector_s_hi,
                                     corrector_snr=args.corrector_snr, per_expert_uncond=args.per_expert_uncond)
                lat, lw, _ = sample_fkc_poe_n(experts, unet, noise_scheduler, scale_factor, spacing, modality,
                                             (C,D,H,W), scfg, uncond_controlnet_cond=uncond,
                                             include_modality=True, include_body_region=False)
                if lat.shape[0] > 1:  # FKC-SMC: collapse particles to the max-weight one for decode/scoring
                    lat = lat[int(lw.argmax()):int(lw.argmax()) + 1]
            elif method == "asym_cfg":
                # ASYMMETRIC per-expert guidance: v_uncond + cfg_a*d_A + cfg_b*d_B.
                # Motivated by the global cfg sweep: abd benefits from >1 guidance, spine degrades past 1.5.
                experts = [ControlNetExpert("A", ctrlA, condA), ControlNetExpert("B", ctrlB, condB)]
                scfg = SamplerConfig(K_SMC=1, n_steps=30, composition="cfg_poe", use_fkc_weights=False,
                                     cfg_scale_per_expert=[args.cfg_a, args.cfg_b], seed=1234)
                lat, _, _ = sample_fkc_poe_n(experts, unet, noise_scheduler, scale_factor, spacing, modality,
                                             (C,D,H,W), scfg, uncond_controlnet_cond=uncond,
                                             include_modality=True, include_body_region=False)
            elif method == "auto_cfg":
                # AUTO per-expert guidance: s_k from localization confidence (own/foreign increment ratio)
                # vs each expert's in-dist baseline. Reproduces symmetric(in-domain)/asymmetric(OOD) automatically.
                experts = [ControlNetExpert("A", ctrlA, condA), ControlNetExpert("B", ctrlB, condB)]
                def _lm(b):
                    t = torch.from_numpy(b.astype(np.float32))[None, None].to(device)
                    return torch.nn.functional.max_pool3d(t, 4, 4)[0, 0]
                own_A = _lm(np.isin(condA_arr, list(ABD_IDS))); for_A = _lm(condA_arr == 200)
                own_B = _lm(np.isin(condB_arr, list(SPINE_IDS))); for_B = _lm(condB_arr == 200)
                scfg = SamplerConfig(K_SMC=args.k_smc, n_steps=30, composition="cfg_poe",
                                     use_fkc_weights=args.use_fkc_weights, fkc_eta=args.fkc_eta,
                                     auto_guidance=True, auto_s_max=args.auto_s_max, auto_tau_lo=args.auto_tau_lo,
                                     auto_rho_base=[args.rho_base_a, args.rho_base_b], seed=1234)
                lat, lw, dg = sample_fkc_poe_n(experts, unet, noise_scheduler, scale_factor, spacing, modality,
                                             (C,D,H,W), scfg, uncond_controlnet_cond=uncond,
                                             own_masks=[own_A, own_B], foreign_masks=[for_A, for_B],
                                             include_modality=True, include_body_region=False)
                if lat.shape[0] > 1:  # FKC-SMC: collapse particles to the max-weight one for decode/scoring
                    lat = lat[int(lw.argmax()):int(lw.argmax()) + 1]
                if dg.expert_region_mag:
                    sk = np.array(dg.expert_region_mag).mean(axis=0)  # mean s_k over steps
                    print(f"    [auto_cfg] mean s_k = abd {sk[0]:.3f}  spine {sk[1]:.3f}")
            elif method == "soft_pull":
                # SOFT-PULL "expert-repaint": naive cfg_poe + per-region soft velocity-blend toward each
                # expert's own velocity (= position-pull toward each expert's own-region x0 solution, tau-free).
                experts = [ControlNetExpert("A", ctrlA, condA), ControlNetExpert("B", ctrlB, condB)]
                def _lm(b):
                    t = torch.from_numpy(b.astype(np.float32))[None, None].to(device)
                    return torch.nn.functional.max_pool3d(t, 4, 4)[0, 0]
                own_A = _lm(np.isin(condA_arr, list(ABD_IDS)))
                own_B = _lm(np.isin(condB_arr, list(SPINE_IDS)))
                scfg = SamplerConfig(K_SMC=1, n_steps=30, composition="cfg_poe", use_fkc_weights=False,
                                     soft_pull_gamma=args.soft_pull_gamma, seed=1234)
                lat, _, _ = sample_fkc_poe_n(experts, unet, noise_scheduler, scale_factor, spacing, modality,
                                             (C,D,H,W), scfg, uncond_controlnet_cond=uncond,
                                             own_masks=[own_A, own_B],
                                             include_modality=True, include_body_region=False)
            elif method in ("gated", "shielded"):
                experts = [ControlNetExpert("A", ctrlA, condA), ControlNetExpert("B", ctrlB, condB)]
                shp = ShieldConfig(lambda_expert=args.lam, rho=args.rho, beta_sync=args.beta_sync,
                                   use_shield=(method == "shielded" and not args.no_shield), use_null_sub=True, use_gate=True,
                                   use_budget=(not args.no_budget), use_time_sync=(not args.no_sync),
                                   gate_floor=args.gate_floor, seed=1234, n_steps=30)
                lat, _ = sample_shielded_local(experts, unet, noise_scheduler, scale_factor, spacing, modality,
                                               (C,D,H,W), uncond, [g_A, g_B], [P_A, P_B], shp,
                                               include_modality=True, include_body_region=False)
            elif method in ("ownership", "ownership_norm"):
                # SOFT asymmetric ownership: u = sum_k g_k*(v_c-v0). NO hard veto / shield / budget / sync.
                # Each expert keeps full strength in its own zone (g_k~1), softly suppressed in foreign zone.
                experts = [ControlNetExpert("A", ctrlA, condA), ControlNetExpert("B", ctrlB, condB)]
                shp = ShieldConfig(lambda_expert=1.0, use_shield=False, use_null_sub=False, use_gate=True,
                                   gate_floor=0.0, use_hard_veto=False, norm_own=(method == "ownership_norm"),
                                   use_budget=False, use_time_sync=False, seed=1234, n_steps=30)
                lat, _ = sample_shielded_local(experts, unet, noise_scheduler, scale_factor, spacing, modality,
                                               (C,D,H,W), uncond, [g_A, g_B], [P_A, P_B], shp,
                                               include_modality=True, include_body_region=False)
            elif method in ("uncertainty", "uncertainty_pure"):
                # UNCERTAINTY-AWARE PoE: precision weights from per-voxel guidance magnitude ||d_k||.
                # Zero extra cost vs naive (d_k = v_c - v0). Confident expert wins per-voxel; OOD expert auto-defers.
                experts = [ControlNetExpert("A", ctrlA, condA), ControlNetExpert("B", ctrlB, condB)]
                shp = ShieldConfig(lambda_expert=1.0, use_shield=False, use_null_sub=args.unc_nullsub, use_gate=False,
                                   use_hard_veto=False, norm_own=False, use_budget=False, use_time_sync=False,
                                   use_uncertainty=True, unc_power=args.unc_power, unc_smooth=args.unc_smooth,
                                   unc_self_norm=(not args.no_unc_self_norm),
                                   unc_support_mix=(0.0 if method == "uncertainty_pure" else args.unc_support_mix),
                                   seed=1234, n_steps=30)
                lat, _ = sample_shielded_local(experts, unet, noise_scheduler, scale_factor, spacing, modality,
                                               (C,D,H,W), uncond, [g_A, g_B], [P_A, P_B], shp,
                                               include_modality=True, include_body_region=False)
            elif method == "mc_adaptive":
                # T2 REGIME-ADAPTIVE: u = sum_k c_k*d_k, c_k = MC-variance outlier gate (ABSOLUTE epistemic).
                # In-domain c_k~1 -> naive additive; OOD foreign expert -> c->0 in its OOD zone -> suppressed.
                experts = [ControlNetExpert("A", ctrlA, condA), ControlNetExpert("B", ctrlB, condB)]
                shp = ShieldConfig(lambda_expert=1.0, use_shield=False, use_null_sub=False, use_gate=False,
                                   use_hard_veto=False, norm_own=False, use_budget=False, use_time_sync=False,
                                   use_mc_uncertainty=True, mc_samples=args.mc_samples, mc_eps=args.mc_eps,
                                   mc_kappa=args.mc_kappa, mc_sharp=args.mc_sharp, seed=1234, n_steps=30)
                lat, _ = sample_shielded_local(experts, unet, noise_scheduler, scale_factor, spacing, modality,
                                               (C,D,H,W), uncond, [g_A, g_B], [P_A, P_B], shp,
                                               include_modality=True, include_body_region=False, verbose=True)
            else:
                raise ValueError(method)
            if gen is None:
                gen = decode(ae, lat, scale_factor)
            gen_t = time.time() - t0
            # segment
            t0 = time.time()
            p = out_dir / f"{cid}_{(args.save_label or method)}.nii.gz"   # PERSIST for union-proxy re-scoring
            nib.save(nib.Nifti1Image(gen.astype(np.float32), affine), str(p))
            data, props = sitk_io.read_images([str(p)])
            pn = predictor.predict_single_npy_array(input_image=data, image_properties=props,
                                                    segmentation_previous_stage=None, save_or_return_probabilities=False).astype(np.int32)
            pm = remap(pn.transpose(2,1,0))
            d = dice_per_label(pm, lbl, ids_abd)
            cf = [v for v in d.values() if not np.isnan(v)]
            md = float(np.mean(cf)) if cf else float("nan")
            results[method]["per_case"].append(md)
            for sid, dv in d.items():
                if not np.isnan(dv): results[method]["per_organ"][sid].append(dv)
            print(f"    [{method:<14}] gen={gen_t:.0f}s seg={time.time()-t0:.0f}s  abd={md:.3f}")
            torch.cuda.empty_cache()

    print(f"\n========== Shielded MVP (n={len(eval_meta)}) abd Dice ==========")
    summary = {"n_cases": len(eval_meta), "params": {"lam": args.lam, "rho": args.rho, "beta_sync": args.beta_sync}, "methods": {}}
    for m, R in results.items():
        v = [x for x in R["per_case"] if not np.isnan(x)]
        mean = float(np.mean(v)) if v else float("nan"); std = float(np.std(v)) if v else float("nan")
        print(f"  {m:<14} abd = {mean:.3f} ± {std:.3f}  (n={len(v)})")
        summary["methods"][m] = {"mean": mean, "std": std, "n": len(v), "per_case": R["per_case"],
                                 "per_organ": {int(s): float(np.mean(x)) for s,x in R["per_organ"].items()}}
    # paired vs single_A
    if "single_A" in results:
        a = np.array(results["single_A"]["per_case"])
        for m in methods:
            if m == "single_A": continue
            b = np.array(results[m]["per_case"])
            mask = ~(np.isnan(a) | np.isnan(b))
            if mask.sum() > 0:
                dlt = b[mask] - a[mask]
                print(f"  Δ({m} − single_A) = {dlt.mean():+.3f} ± {dlt.std():.3f}  (n={int(mask.sum())})")
                summary["methods"][m]["delta_vs_single_A"] = float(dlt.mean())
    suffix = f"_{args.tag}" if args.tag else ""
    out_p = REP / f"eval_path2_shielded_summary{suffix}.json"
    out_p.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_p}")


if __name__ == "__main__":
    main()
