#!/usr/bin/env python3
"""Custom ControlNet trainer for medgen-fkc — bypasses MAISI's MONAI CacheDataset.

The official `maisi/scripts/train_controlnet.py` silently crashes on our dataset
(suspected fault: CacheDataset + multi-worker DataLoader fork/cache pressure).
The minimal smoke test (`scripts/minimal_smoke.py`) confirms MAISI's UNet +
ControlNet forward + L1 loss path works correctly on a single case.

This custom trainer:
  - Loads NIfTI latents + labels directly via nibabel in the main thread (no
    workers, no CacheDataset, no fork).
  - Shuffles per epoch with a fixed seed.
  - Logs loss + GPU mem + timing every iteration.
  - Saves a `controlnet_state_dict` checkpoint at end-of-epoch.
  - Supports --resume from a previous checkpoint.

Usage (detached / job runner):
  python -u scripts/train_controlnet_custom.py \
    --env-config configs/env_smoke_expert_A.json \
    --model-config configs/config_smoke_5ep.json \
    --network-config configs/config_network_rflow.json
"""

import argparse
import json
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

REPO_ROOT = Path(__file__).resolve().parents[1]
MAISI_ROOT = REPO_ROOT / "maisi"
sys.path.insert(0, str(MAISI_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from monai.networks.utils import copy_model_state
from monai.networks.schedulers import RFlowScheduler

from scripts.utils import define_instance, binarize_labels
from scripts.diff_model_setting import load_config
from src.cond_encoding import to_one_hot as _to_one_hot, N_COND_CHANNELS, to_inpaint_cond, N_INPAINT_COND_CHANNELS


# binarize_labels (imported from maisi/scripts/utils.py) is the native MAISI 8-bit encoding.
# Each integer label ID is decomposed into its 8-bit binary representation, one bit per channel.
# ID 200 (body silhouette) must be present in label volumes — see scripts/add_body_silhouette.py.


def log(msg):
    print(f"[{time.strftime('%X')}] {msg}", flush=True)


def load_latent(path: Path, device: torch.device) -> torch.Tensor:
    """NIfTI channel-last (X, Y, Z, C) -> tensor (1, C, X, Y, Z)."""
    arr = nib.load(str(path)).get_fdata().astype(np.float32)
    return torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).to(device)


def load_label(path: Path, device: torch.device) -> torch.Tensor:
    """NIfTI int volume -> tensor (1, 1, X, Y, Z)."""
    arr = nib.load(str(path)).get_fdata().astype(np.int64)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def load_ct_norm(path: Path, device: torch.device) -> torch.Tensor:
    """Raw CT NIfTI (HU) -> normalized (1,1,X,Y,Z) in [0,1] (same convention as AE: (HU+1000)/2000)."""
    arr = nib.load(str(path)).get_fdata().astype(np.float32)
    arr = np.clip((arr + 1000.0) / 2000.0, 0.0, 1.0)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def make_inpaint_cond(lbl_t: torch.Tensor, ct_t: torch.Tensor, dilate: int = 3) -> torch.Tensor:
    """Build the 45-ch inpaint conditioning [label one-hot(43) | masked-CT(1) | mask R(1)].
    R = expert's organ voxels (lbl>0 & lbl!=200) dilated; masked-CT = CT with R zeroed (the model fills R)."""
    organ = ((lbl_t > 0) & (lbl_t != 200)).float()                       # (b,1,...) expert's organs
    d = int(np.random.randint(max(1, dilate - 1), dilate + 2))           # mild random dilation jitter
    R = torch.nn.functional.max_pool3d(organ, kernel_size=2 * d + 1, stride=1, padding=d)
    R = (R > 0.5).float()
    masked_ct = ct_t * (1.0 - R)                                          # zero (air) the inpaint region
    return to_inpaint_cond(lbl_t.long(), masked_ct, R)                    # (b,45,...)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-config", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--network-config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Per-step batch (real batching, stacks N cases).")
    ap.add_argument("--grad-accumulate", type=int, default=1,
                    help="Gradient accumulation factor. Effective batch = batch_size * grad_accumulate.")
    ap.add_argument("--phase", choices=["1", "2", "mmwhs_b", "amos22_a"], default="1",
                    help="Dataset selector. All phases use 8-ch binarize_labels encoding.")
    ap.add_argument("--expert", choices=["A", "B", "oracle"], default="oracle",
                    help="Informational label only (no channel masking in binarize_labels mode).")
    ap.add_argument("--vis-every", type=int, default=0,
                    help="Save inference PNG every N epochs (0 = disabled).")
    ap.add_argument("--vis-case-idx", type=int, default=0,
                    help="Dataset index of fixed case used for vis inference.")
    ap.add_argument("--vis-steps", type=int, default=30,
                    help="Denoising steps for vis inference.")
    ap.add_argument("--cfg-dropout", type=float, default=0.1,
                    help="Classifier-free guidance dropout probability (0 = disabled).")
    ap.add_argument("--lr-schedule", choices=["poly2", "constant", "warmup-plateau"],
                    default="poly2",
                    help="poly2 = original PolynomialLR(power=2); constant = no decay; "
                         "warmup-plateau = linear warmup 5% steps then constant.")
    ap.add_argument("--warmup-frac", type=float, default=0.05,
                    help="Fraction of total steps used for warmup (warmup-plateau only).")
    ap.add_argument("--lr-override", type=float, default=None,
                    help="Override lr from config (e.g. 5e-5).")
    ap.add_argument("--epochs-override", type=int, default=None,
                    help="Override n_epochs from config.")
    ap.add_argument("--val-json", type=str, default=None,
                    help="JSON file with held-out validation entries (same format as training). "
                         "If provided, compute val_loss every --val-every epochs and save checkpoint_best.pt.")
    ap.add_argument("--val-every", type=int, default=1,
                    help="Run validation every N epochs (only when --val-json is set).")
    ap.add_argument("--val-n-samples", type=int, default=0,
                    help="Limit validation to first N entries (0 = use all). For speed during dev.")
    ap.add_argument("--n-bins", type=int, default=5,
                    help="Number of t bins for binned training+val loss tracking (0 = disabled). "
                         "For flow models, global mean loss blends signal across t; per-bin is informative.")
    ap.add_argument("--t-alpha", type=float, default=1.0,
                    help="Beta-distribution alpha for t sampling. alpha=beta=1 = uniform. "
                         "alpha=beta=2 = concentrate mid-t (Beta(2,2)). "
                         "alpha=beta=4 = stronger mid-t concentration.")
    ap.add_argument("--t-beta", type=float, default=1.0,
                    help="Beta-distribution beta for t sampling. See --t-alpha.")
    ap.add_argument("--z-jitter", type=int, default=0,
                    help="Maximum random Z-shift in latent voxels per training step (each = 4 label voxels). "
                         "0 = disabled. Typical: 2 or 3. Used to make experts FOV-shift robust for cross-dataset composition.")
    ap.add_argument("--cond-encoding", choices=["binarize", "onehot"], default="binarize",
                    help="Conditioning encoding. 'binarize' = native MAISI 8-bit (REQUIRED when loading "
                         "pretrained ControlNet). 'onehot' = 43-channel per-organ one-hot (only valid "
                         "when training from scratch; network config must set conditioning_embedding_in_channels=43).")
    ap.add_argument("--inpaint-dilate", type=int, default=3,
                    help="Dilation (voxels) of the organ region used as the inpaint mask R. Network config must set "
                         "conditioning_embedding_in_channels=45 (configs/config_network_inpaint45.json).")
    args = ap.parse_args()

    if args.cond_encoding == "onehot":
        def make_cond(lbl):
            return _to_one_hot(lbl.long())
        log(f"phase={args.phase}  expert={args.expert}  conditioning=one-hot({N_COND_CHANNELS}-ch)")
    else:
        def make_cond(lbl):
            return binarize_labels(lbl.long()).float()
        log(f"phase={args.phase}  expert={args.expert}  conditioning=binarize_labels(8-bit)")

    cfg = load_config(args.env_config, args.model_config, args.network_config)

    device = torch.device("cuda:0")
    data_root = Path(cfg.data_base_dir[0]) if isinstance(cfg.data_base_dir, list) else Path(cfg.data_base_dir)
    json_path = cfg.json_data_list[0] if isinstance(cfg.json_data_list, list) else cfg.json_data_list

    with open(json_path) as f:
        entries = json.load(f)["training"]
    log(f"loaded {len(entries)} entries from {json_path}")

    # --- Load validation entries (held-out, never in training)
    val_entries = []
    if args.val_json:
        # Try absolute, then relative to CWD, then relative to data_root
        candidates = [Path(args.val_json)]
        candidates.append(Path.cwd() / args.val_json)
        candidates.append(data_root / Path(args.val_json).name)
        val_path = None
        for c in candidates:
            if c.exists():
                val_path = c
                break
        if val_path is None:
            log(f"WARN: --val-json {args.val_json} not found (tried {candidates}); validation disabled")
        else:
            with open(val_path) as f:
                val_entries = json.load(f)["training"]
            if args.val_n_samples > 0:
                val_entries = val_entries[:args.val_n_samples]
            log(f"loaded {len(val_entries)} validation entries from {val_path}")

    # --- Build models
    log("building UNet ...")
    unet = define_instance(cfg, "diffusion_unet_def").to(device)
    sd = torch.load(cfg.trained_diffusion_path, map_location=device, weights_only=False)
    unet.load_state_dict(sd["unet_state_dict"], strict=False)
    scale_factor = sd["scale_factor"].to(device) if hasattr(sd["scale_factor"], "to") else torch.tensor(sd["scale_factor"]).to(device)
    unet.eval()
    for p in unet.parameters():
        p.requires_grad_(False)
    log(f"UNet loaded; scale_factor={scale_factor.item():.4f}; GPU {torch.cuda.memory_allocated()/1e9:.2f}GB")

    log("building ControlNet ...")
    controlnet = define_instance(cfg, "controlnet_def").to(device)
    copy_model_state(controlnet, unet.state_dict())
    if cfg.existing_ckpt_filepath and Path(cfg.existing_ckpt_filepath).exists():
        ctrl_sd = torch.load(cfg.existing_ckpt_filepath, map_location=device, weights_only=False)
        src = ctrl_sd["controlnet_state_dict"]
        tgt = controlnet.state_dict()
        # Warm-start from a from-scratch generation expert. The ONLY architectural difference is the
        # cond-embedding first conv: source has 43 input channels (one-hot labels), the inpaint model
        # has 45 (= 43 labels + masked-CT + mask). Copy the 43 trained channels into the first 43 and
        # zero-init the 2 new (masked-CT, mask) channels so the model starts as the gen expert and
        # *learns* to use the image context. All other 230 params match exactly.
        CONV_IN = "controlnet_cond_embedding.conv_in.conv.weight"
        new_sd, n_copy, n_expand, n_keep = {}, 0, 0, 0
        for k, v in tgt.items():
            sv = src.get(k, None)
            if sv is not None and sv.shape == v.shape:
                new_sd[k] = sv; n_copy += 1
            elif (sv is not None and k == CONV_IN and sv.dim() == 5 and v.dim() == 5
                  and sv.shape[0] == v.shape[0] and sv.shape[2:] == v.shape[2:] and sv.shape[1] < v.shape[1]):
                w = v.clone().zero_()
                w[:, :sv.shape[1]] = sv
                new_sd[k] = w; n_expand += 1
            else:
                new_sd[k] = v; n_keep += 1  # keep current init (UNet-copied body / fresh)
        controlnet.load_state_dict(new_sd, strict=True)
        log(f"ControlNet warm-started from {cfg.existing_ckpt_filepath}: "
            f"{n_copy} copied, {n_expand} channel-expanded (43->45 cond conv_in), {n_keep} kept-init")
    else:
        log("ControlNet trained from scratch (no pretrained ckpt)")
    controlnet.train()
    log(f"ControlNet ready; GPU {torch.cuda.memory_allocated()/1e9:.2f}GB")

    noise_scheduler = define_instance(cfg, "noise_scheduler")

    # --- Optimizer
    n_epochs = args.epochs_override if args.epochs_override else cfg.controlnet_train["n_epochs"]
    lr = args.lr_override if args.lr_override else cfg.controlnet_train["lr"]
    _bsz_for_lr = max(1, int(args.batch_size))
    _gacc_for_lr = max(1, int(args.grad_accumulate))
    _microbatches_per_epoch = (len(entries) + _bsz_for_lr - 1) // _bsz_for_lr
    total_steps = max(1, n_epochs * ((_microbatches_per_epoch + _gacc_for_lr - 1) // _gacc_for_lr))
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=lr)
    if args.lr_schedule == "poly2":
        lr_sched = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)
    elif args.lr_schedule == "constant":
        lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    elif args.lr_schedule == "warmup-plateau":
        n_warm = max(1, int(args.warmup_frac * total_steps))
        def _wp(step):
            if step < n_warm:
                return (step + 1) / n_warm
            return 1.0
        lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_wp)
    else:
        raise ValueError(args.lr_schedule)
    log(f"lr_schedule={args.lr_schedule}  warmup_frac={args.warmup_frac}  total_steps={total_steps}")
    scaler = GradScaler("cuda")
    log(f"optimizer ready: lr={lr}, n_epochs={n_epochs}, eff_batch={_bsz_for_lr * _gacc_for_lr}, total_opt_steps={total_steps}")

    # --- Resume
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    resumed_step = 0
    if args.resume:
        ck_path = output_dir / "checkpoint_latest.pt"
        if ck_path.exists():
            ck = torch.load(ck_path, map_location=device, weights_only=False)
            controlnet.load_state_dict(ck["controlnet_state_dict"])
            optimizer.load_state_dict(ck["optimizer_state_dict"])
            start_epoch = ck["epoch"] + 1
            resumed_step = ck.get("step", 0)
            # Re-init LR scheduler so extending n_epochs works correctly, but keep
            # the saved scheduler epoch instead of double-decaying from resumed LR.
            # (Honor --lr-schedule for the fresh scheduler.)
            if args.lr_schedule == "poly2":
                lr_sched = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)
            elif args.lr_schedule == "constant":
                lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
            elif args.lr_schedule == "warmup-plateau":
                n_warm = max(1, int(args.warmup_frac * total_steps))
                def _wp_resume(step):
                    if step < n_warm:
                        return (step + 1) / n_warm
                    return 1.0
                lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_wp_resume)
            sched_state = ck.get("lr_scheduler_state_dict")
            if sched_state is not None:
                lr_sched.load_state_dict(sched_state)
                if getattr(lr_sched, "total_iters", total_steps) != total_steps:
                    lr_sched.total_iters = total_steps
            else:
                for _ in range(resumed_step):
                    lr_sched.step()
            log(f"resumed from {ck_path} at epoch {start_epoch} (step {resumed_step})")

    # --- Vis setup (periodic inference PNG)
    ae_vis = None
    vis_cond = vis_spacing = vis_modality = vis_dir = None
    if args.vis_every > 0:
        log("Loading AutoEncoder for vis (CPU, norm_float16=False) ...")
        ae_vis = define_instance(cfg, "autoencoder_def")  # keep on CPU
        ae_sd_vis = torch.load(cfg.trained_autoencoder_path, map_location="cpu", weights_only=False)
        if isinstance(ae_sd_vis, dict) and "unet_state_dict" in ae_sd_vis:
            ae_sd_vis = ae_sd_vis["unet_state_dict"]
        ae_vis.load_state_dict(ae_sd_vis)
        ae_vis.eval()
        for p in ae_vis.parameters():
            p.requires_grad_(False)
        for m in ae_vis.modules():
            if hasattr(m, "norm_float16"):
                m.norm_float16 = False
        log("AutoEncoder vis ready (CPU)")

        vis_idx = min(args.vis_case_idx, len(entries) - 1)
        vis_entry = entries[vis_idx]
        vis_lbl = load_label(data_root / vis_entry["label"], device)
        vis_cond = make_cond(vis_lbl)
        vis_spacing = torch.tensor([vis_entry["spacing"]], device=device, dtype=torch.float32) * 100.0
        vis_modality = torch.ones((1,), device=device, dtype=torch.long)
        vis_dir = output_dir / "vis"
        vis_dir.mkdir(exist_ok=True)
        log(f"vis case idx={vis_idx}: {vis_entry['label']}")

    # --- Train loop
    rng = random.Random(1234)
    indices = list(range(len(entries)))
    t_start = time.time()
    step = resumed_step if args.resume else 0

    # Bin tracking for flow models — t-binned loss is more informative than global mean.
    NB = max(0, int(args.n_bins))
    T_max = noise_scheduler.num_train_timesteps
    bin_edges = np.linspace(0, 1, NB + 1) if NB > 0 else None

    # Auto-stop watchdog
    NAN_WINDOW = 100
    DIVERGE_THRESHOLD = 5.0
    SKIP_LIMIT = 50
    consecutive_skip = 0
    BSZ = max(1, int(args.batch_size))
    GACC = max(1, int(args.grad_accumulate))
    eff_bsz = BSZ * GACC
    log(f"batch_size={BSZ}  grad_accumulate={GACC}  effective_batch={eff_bsz}  cfg_dropout={args.cfg_dropout}")

    recent_losses = []
    for epoch in range(start_epoch, n_epochs):
        rng.shuffle(indices)
        epoch_losses = []
        # Per-bin accumulators for training loss
        bin_loss_sum = [0.0] * NB if NB > 0 else None
        bin_loss_count = [0] * NB if NB > 0 else None
        t_epoch = time.time()

        # Iterate batches of size BSZ; group GACC such batches into one optimizer step
        n = len(indices)
        i = 0
        accum_count = 0
        accum_target = GACC
        optimizer.zero_grad(set_to_none=True)

        while i < n:
            batch_start = i
            batch_indices = indices[i:i + BSZ]
            i += len(batch_indices)
            if accum_count == 0:
                remaining_microbatches = (n - batch_start + BSZ - 1) // BSZ
                accum_target = min(GACC, remaining_microbatches)

            # Load and stack batch
            img_list, lbl_list, ct_list, spacing_list, valid_entries = [], [], [], [], []
            for bi in batch_indices:
                entry = entries[bi]
                try:
                    im = load_latent(data_root / entry["image"], device)
                    lb = load_label(data_root / entry["label"], device)
                    ct = load_ct_norm(data_root / "images" / Path(entry["label"]).name, device)  # raw CT for masked-image cond
                    img_list.append(im); lbl_list.append(lb); ct_list.append(ct)
                    spacing_list.append(entry["spacing"]); valid_entries.append(entry)
                except Exception as e:
                    print(f"  [skip {entry['image']}] {type(e).__name__}: {e}", flush=True)
            if not img_list:
                continue

            img_t = torch.cat(img_list, dim=0)         # (b, 4, D, H, W) — latent
            lbl_t = torch.cat(lbl_list, dim=0)         # (b, 1, D, H, W) — full-res label
            ct_t = torch.cat(ct_list, dim=0)           # (b, 1, D, H, W) — full-res CT (normalized) for masked-image cond
            b = img_t.shape[0]

            # Z-jitter augmentation: shift latent + label by same Z in physical space.
            # Latent Z dim is 1/4 of label Z dim, so latent shift k → label shift k*4.
            if args.z_jitter > 0:
                z_shift_lat = int(np.random.randint(-args.z_jitter, args.z_jitter + 1))
                if z_shift_lat != 0:
                    img_t = torch.roll(img_t, shifts=z_shift_lat, dims=-3)
                    lbl_t = torch.roll(lbl_t, shifts=z_shift_lat * 4, dims=-3)
                    ct_t = torch.roll(ct_t, shifts=z_shift_lat * 4, dims=-3)

            images_scaled = img_t * scale_factor
            noise = torch.randn_like(images_scaled)
            # t sampling: beta(alpha, beta). alpha=beta=1 == uniform (scheduler default).
            if args.t_alpha == 1.0 and args.t_beta == 1.0:
                if isinstance(noise_scheduler, RFlowScheduler):
                    timesteps = noise_scheduler.sample_timesteps(images_scaled)
                else:
                    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (b,), device=device).long()
            else:
                t_frac = np.random.beta(args.t_alpha, args.t_beta, size=b).astype(np.float32)
                if isinstance(noise_scheduler, RFlowScheduler):
                    timesteps = torch.from_numpy(t_frac * float(T_max)).to(device).float()
                else:
                    timesteps = torch.from_numpy((t_frac * float(T_max)).astype(np.int64)).to(device).long()
            noisy_latent = noise_scheduler.add_noise(original_samples=images_scaled, noise=noise, timesteps=timesteps)

            controlnet_cond = make_inpaint_cond(lbl_t, ct_t, dilate=args.inpaint_dilate)  # (b,45,...)
            if args.cfg_dropout > 0:
                drop_mask = torch.rand(b, device=device) < args.cfg_dropout
                controlnet_cond[drop_mask, :N_COND_CHANNELS] = 0.0   # cfg: drop LABEL channels only; keep masked-CT+mask context
            spacing_tensor = torch.tensor(spacing_list, device=device, dtype=torch.float32) * 100.0  # (b, 3)
            modality_tensor = torch.ones((b,), device=device, dtype=torch.long)                     # (b,)

            with autocast("cuda"):
                down, mid = controlnet(
                    x=noisy_latent, timesteps=timesteps,
                    controlnet_cond=controlnet_cond, class_labels=modality_tensor,
                )
                model_output = unet(
                    x=noisy_latent, timesteps=timesteps, spacing_tensor=spacing_tensor,
                    down_block_additional_residuals=down,
                    mid_block_additional_residual=mid,
                    class_labels=modality_tensor,
                )
                if isinstance(noise_scheduler, RFlowScheduler):
                    target = images_scaled - noise
                else:
                    target = noise
                # Divide by the current accumulation window so accumulated grads
                # represent the mean, including the final short window in an epoch.
                loss = F.l1_loss(model_output.float(), target.float()) / accum_target

            loss_val = float(loss.detach().cpu()) * accum_target  # un-normalize for logging
            # Per-bin training loss tracking
            if NB > 0 and bin_loss_sum is not None:
                ts_cpu = timesteps.detach().float().cpu().numpy() / float(T_max)
                for tf in ts_cpu:
                    bi = min(NB - 1, max(0, int(tf * NB)))
                    bin_loss_sum[bi] += loss_val
                    bin_loss_count[bi] += 1
            if not (loss_val == loss_val) or loss_val == float("inf") or loss_val == float("-inf"):
                print(f"  [skip batch] loss is NaN/Inf -- entries: {[e['image'] for e in valid_entries]}", flush=True)
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                del loss
                consecutive_skip += 1
                if consecutive_skip >= SKIP_LIMIT:
                    log(f"ABORT: {consecutive_skip} consecutive batches skipped")
                    sys.exit(2)
                continue

            scaler.scale(loss).backward()
            accum_count += 1

            if accum_count >= accum_target:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(controlnet.parameters(), max_norm=1.0)
                if not torch.isfinite(grad_norm):
                    print(f"  [skip step] grad_norm is NaN/Inf", flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    accum_count = 0
                    consecutive_skip += 1
                    if consecutive_skip >= SKIP_LIMIT:
                        log(f"ABORT: {consecutive_skip} consecutive grad-skips")
                        sys.exit(2)
                    continue
                scaler.step(optimizer)
                scaler.update()
                lr_sched.step()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                step += 1
                consecutive_skip = 0
            epoch_losses.append(loss_val)
            recent_losses.append(loss_val)
            if len(recent_losses) > NAN_WINDOW:
                recent_losses.pop(0)
            if len(recent_losses) >= NAN_WINDOW:
                window_mean = sum(recent_losses) / len(recent_losses)
                if window_mean > DIVERGE_THRESHOLD:
                    log(f"ABORT: mean loss over last {NAN_WINDOW} steps = {window_mean:.3f} > {DIVERGE_THRESHOLD}")
                    sys.exit(3)

            # Free per-case tensors
            del img_t, lbl_t, controlnet_cond, noisy_latent, noise, model_output, down, mid

            if step > 0 and step % 5 == 0 and accum_count == 0:
                elapsed = time.time() - t_start
                rate = step / max(elapsed, 1e-6)
                mem = torch.cuda.memory_allocated() / 1e9
                log(f"epoch {epoch}/{n_epochs} step {step}/{total_steps} loss {loss_val:.4f} "
                    f"lr {lr_sched.get_last_lr()[0]:.2e} rate {rate:.2f} it/s GPU {mem:.2f}GB")

        epoch_mean = np.mean(epoch_losses) if epoch_losses else float("nan")
        log(f"=== epoch {epoch} done: mean_loss {epoch_mean:.4f} time {time.time()-t_epoch:.0f}s ===")

        # Per-bin training-loss breakdown
        train_bin_means = None
        if NB > 0 and bin_loss_count is not None:
            train_bin_means = [bin_loss_sum[i] / bin_loss_count[i] if bin_loss_count[i] > 0 else float("nan")
                               for i in range(NB)]
            bins_str = "  ".join([f"[{bin_edges[i]:.2f},{bin_edges[i+1]:.2f}]={train_bin_means[i]:.3f}"
                                  for i in range(NB)])
            log(f"  train_bins: {bins_str}")

        # Save checkpoint
        ck_path = output_dir / "checkpoint_latest.pt"
        torch.save({
            "controlnet_state_dict": controlnet.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_sched.state_dict(),
            "epoch": epoch,
            "step": step,
            "mean_loss": epoch_mean,
        }, ck_path)
        log(f"saved {ck_path}")

        # --- Validation (val_loss) + best-checkpoint tracking
        if val_entries and (epoch + 1) % args.val_every == 0:
            controlnet.eval()
            val_losses = []
            val_bin_losses = [[] for _ in range(NB)] if NB > 0 else None
            t_val = time.time()
            val_rng = np.random.default_rng(1234 + epoch)
            with torch.no_grad():
                for ei, ve in enumerate(val_entries):
                    try:
                        v_img = load_latent(data_root / ve["image"], device)
                        v_lbl = load_label(data_root / ve["label"], device)
                    except Exception as e:
                        print(f"  [val skip {ve['image']}] {type(e).__name__}: {e}", flush=True)
                        continue
                    v_scaled = v_img * scale_factor
                    # 45-ch inpaint cond (NOT plain one-hot): load the raw CT like training and build the
                    # masked-image conditioning, else the 45-ch cond_in conv gets a 43-ch input and crashes.
                    v_ct = load_ct_norm(data_root / "images" / Path(ve["label"]).name, device)
                    v_cond = make_inpaint_cond(v_lbl, v_ct, dilate=args.inpaint_dilate)
                    v_spacing = torch.tensor([ve["spacing"]], device=device, dtype=torch.float32) * 100.0
                    v_mod = torch.ones((1,), device=device, dtype=torch.long)

                    # Binned val: cycle through bins so each case contributes to all bins evenly
                    if NB > 0:
                        bi = ei % NB
                        lo, hi = bin_edges[bi], bin_edges[bi + 1]
                        t_frac = float(val_rng.uniform(lo, hi))
                        if isinstance(noise_scheduler, RFlowScheduler):
                            v_t = torch.tensor([t_frac * T_max], device=device, dtype=torch.float32)
                        else:
                            v_t = torch.tensor([int(t_frac * T_max)], device=device, dtype=torch.long)
                    else:
                        if isinstance(noise_scheduler, RFlowScheduler):
                            v_t = noise_scheduler.sample_timesteps(v_scaled)
                        else:
                            v_t = torch.randint(0, noise_scheduler.num_train_timesteps, (1,), device=device).long()

                    v_noise = torch.randn_like(v_scaled)
                    v_noisy = noise_scheduler.add_noise(original_samples=v_scaled, noise=v_noise, timesteps=v_t)
                    with autocast("cuda"):
                        v_down, v_mid = controlnet(
                            x=v_noisy, timesteps=v_t,
                            controlnet_cond=v_cond, class_labels=v_mod,
                        )
                        v_out = unet(
                            x=v_noisy, timesteps=v_t, spacing_tensor=v_spacing,
                            down_block_additional_residuals=v_down,
                            mid_block_additional_residual=v_mid,
                            class_labels=v_mod,
                        )
                        v_target = v_scaled - v_noise if isinstance(noise_scheduler, RFlowScheduler) else v_noise
                        v_loss_t = F.l1_loss(v_out.float(), v_target.float())
                    v_loss_f = float(v_loss_t.detach().cpu())
                    val_losses.append(v_loss_f)
                    if NB > 0:
                        val_bin_losses[ei % NB].append(v_loss_f)
                    del v_img, v_lbl, v_cond, v_noisy, v_noise, v_out, v_down, v_mid
            val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
            log(f"=== epoch {epoch} val_loss={val_loss:.4f} (n={len(val_losses)} samples, {time.time()-t_val:.0f}s) ===")
            if NB > 0 and val_bin_losses is not None:
                val_bin_means = [float(np.mean(b)) if b else float("nan") for b in val_bin_losses]
                bins_str = "  ".join([f"[{bin_edges[i]:.2f},{bin_edges[i+1]:.2f}]={val_bin_means[i]:.3f}(n={len(val_bin_losses[i])})"
                                      for i in range(NB)])
                log(f"  val_bins: {bins_str}")

            # Track best
            if not hasattr(main, "_best_val_loss"):
                main._best_val_loss = float("inf")
                main._best_val_epoch = -1
            if val_loss < main._best_val_loss:
                main._best_val_loss = val_loss
                main._best_val_epoch = epoch
                best_path = output_dir / "checkpoint_best.pt"
                torch.save({
                    "controlnet_state_dict": controlnet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_sched.state_dict(),
                    "epoch": epoch,
                    "step": step,
                    "mean_loss": epoch_mean,
                    "val_loss": val_loss,
                }, best_path)
                log(f"  *** new best val_loss={val_loss:.4f} at epoch {epoch}; saved {best_path}")
            controlnet.train()

        # --- Periodic vis inference
        if ae_vis is not None and args.vis_every > 0 and (epoch + 1) % args.vis_every == 0:
            log(f"running vis inference (epoch {epoch + 1}) ...")
            unet.eval()
            controlnet.eval()
            try:
                with torch.no_grad():
                    sched_vis = deepcopy(noise_scheduler)
                    sched_vis.set_timesteps(
                        num_inference_steps=args.vis_steps,
                        input_img_size_numel=64 ** 3,
                    )
                    z_vis = torch.randn((1, 4, 64, 64, 64), device=device)
                    # RFlowScheduler.step needs next_timestep for correct ODE integration
                    vis_timesteps = sched_vis.timesteps.tolist()
                    vis_next_ts = vis_timesteps[1:] + [0]
                    for t_vis, next_t_vis in zip(vis_timesteps, vis_next_ts):
                        t_b = torch.tensor([t_vis], device=device)
                        with autocast("cuda"):
                            down_vis, mid_vis = controlnet(
                                x=z_vis, timesteps=t_b,
                                controlnet_cond=vis_cond,
                                class_labels=vis_modality,
                            )
                            v_vis = unet(
                                x=z_vis, timesteps=t_b,
                                spacing_tensor=vis_spacing,
                                down_block_additional_residuals=down_vis,
                                mid_block_additional_residual=mid_vis,
                                class_labels=vis_modality,
                            )
                        _step_out = sched_vis.step(v_vis, t_vis, z_vis, next_timestep=next_t_vis)
                        z_vis = _step_out[0] if isinstance(_step_out, (list, tuple)) else _step_out.prev_sample
                    ct_arr = ae_vis.decode_stage_2_outputs(
                        z_vis.cpu() / scale_factor.cpu()
                    ).squeeze().numpy()
                    # Denormalize [0,1] -> HU, then window for display
                    ct_arr = np.clip(ct_arr, 0.0, 1.0)
                    ct_arr = ct_arr * 2000.0 - 1000.0
                    _wl, _ww = 40.0, 400.0
                    ct_arr = np.clip((ct_arr - _wl + _ww / 2.0) / _ww, 0.0, 1.0)

                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                d, h, w = ct_arr.shape
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(ct_arr[d // 2], cmap="gray", vmin=0, vmax=1)
                axes[0].set_title(f"Axial z={d // 2}")
                axes[0].axis("off")
                axes[1].imshow(ct_arr[:, h // 2, :], cmap="gray", vmin=0, vmax=1)
                axes[1].set_title(f"Coronal y={h // 2}")
                axes[1].axis("off")
                axes[2].imshow(ct_arr[:, :, w // 2], cmap="gray", vmin=0, vmax=1)
                axes[2].set_title(f"Sagittal x={w // 2}")
                axes[2].axis("off")
                plt.suptitle(f"epoch {epoch + 1}  loss {epoch_mean:.4f}")
                plt.tight_layout()
                vis_path = vis_dir / f"epoch_{epoch + 1:04d}.png"
                plt.savefig(str(vis_path), dpi=80, bbox_inches="tight")
                plt.close(fig)
                log(f"vis saved: {vis_path}")
                del ct_arr, z_vis
            except Exception as vis_exc:
                log(f"vis FAILED (non-fatal): {vis_exc}")
            finally:
                unet.eval()
                controlnet.train()

    log(f"training done: total {time.time()-t_start:.0f}s, {step} steps")


if __name__ == "__main__":
    main()
