#!/usr/bin/env python3
"""Overfit smoke test: ControlNet fine-tune on tiny subset with constant LR.

Goal: prove the fine-tune pipeline CAN drive loss to near-zero on a small subset.
If yes → pipeline OK, full-dataset plateau is just data complexity.
If no  → pipeline bug (data loading, model state, loss, gradient flow, ...).

Differs from train_controlnet_custom.py:
  - Constant LR (no PolynomialLR decay)
  - Per-step loss printed every step (not every 5)
  - Per-epoch mean loss tracked AND per-case loss tracked (to see if specific cases stuck)
  - Larger initial lr (1e-4 default vs 1e-5)
  - Saves loss curve to JSON for plotting

Usage (detached / job runner):
  python3 scripts/train_controlnet_overfit.py \
      --env-config configs/env_iso15_overfit20.json \
      --model-config maisi/configs/config_overfit_100ep_lr1e4.json \
      --network-config maisi/configs/config_network_rflow.json
"""
import argparse, json, os, random, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

MAISI_ROOT = Path(__file__).resolve().parents[1] / "maisi"
sys.path.insert(0, str(MAISI_ROOT))

from monai.networks.utils import copy_model_state
from monai.networks.schedulers import RFlowScheduler

from scripts.utils import define_instance, binarize_labels
from scripts.diff_model_setting import load_config


def log(msg): print(f"[{time.strftime('%X')}] {msg}", flush=True)


def load_latent(path: Path, device):
    arr = nib.load(str(path)).get_fdata().astype(np.float32)
    return torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).to(device)

def load_label(path: Path, device):
    arr = nib.load(str(path)).get_fdata().astype(np.int64)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-config", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--network-config", required=True)
    ap.add_argument("--lr-override", type=float, default=None)
    ap.add_argument("--output-dir-override", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.env_config, args.model_config, args.network_config)
    if args.lr_override is not None:
        cfg.controlnet_train["lr"] = args.lr_override
    if args.output_dir_override is not None:
        cfg.output_dir = args.output_dir_override
    device = torch.device("cuda:0")
    data_root = Path(cfg.data_base_dir[0]) if isinstance(cfg.data_base_dir, list) else Path(cfg.data_base_dir)
    json_path = cfg.json_data_list[0] if isinstance(cfg.json_data_list, list) else cfg.json_data_list

    with open(json_path) as f:
        entries = json.load(f)["training"]
    log(f"loaded {len(entries)} entries (OVERFIT mode)")

    # Build models
    log("building UNet (frozen) ...")
    unet = define_instance(cfg, "diffusion_unet_def").to(device)
    sd = torch.load(cfg.trained_diffusion_path, map_location=device, weights_only=False)
    unet.load_state_dict(sd["unet_state_dict"], strict=False)
    scale_factor = sd["scale_factor"].to(device) if hasattr(sd["scale_factor"], "to") else torch.tensor(sd["scale_factor"]).to(device)
    unet.eval()
    for p in unet.parameters(): p.requires_grad_(False)
    log(f"UNet ready, scale_factor={scale_factor.item():.4f}")

    log("building ControlNet ...")
    controlnet = define_instance(cfg, "controlnet_def").to(device)
    copy_model_state(controlnet, unet.state_dict())
    if cfg.existing_ckpt_filepath and Path(cfg.existing_ckpt_filepath).exists():
        ck = torch.load(cfg.existing_ckpt_filepath, map_location=device, weights_only=False)
        controlnet.load_state_dict(ck["controlnet_state_dict"], strict=False)
        log(f"ControlNet warm-start: {cfg.existing_ckpt_filepath}")
    controlnet.train()
    n_params = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
    log(f"ControlNet trainable params: {n_params/1e6:.1f}M")

    noise_scheduler = define_instance(cfg, "noise_scheduler")

    n_epochs = cfg.controlnet_train["n_epochs"]
    lr = cfg.controlnet_train["lr"]
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=lr)
    # CONSTANT LR (no scheduler) for overfit test
    scaler = GradScaler("cuda")
    log(f"optimizer: AdamW lr={lr} (CONSTANT, no decay)  n_epochs={n_epochs}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Loss tracking
    epoch_losses_record = []     # mean loss per epoch
    per_case_losses_record = {Path(e["image"]).stem.replace("_emb.nii", "").replace(".nii", ""): [] for e in entries}

    rng = random.Random(1234)
    indices = list(range(len(entries)))
    t_start = time.time()
    step = 0

    for epoch in range(n_epochs):
        rng.shuffle(indices)
        epoch_losses = []
        t_epoch = time.time()
        for idx in indices:
            entry = entries[idx]
            case_key = Path(entry["image"]).stem.replace("_emb.nii", "").replace(".nii", "")
            try:
                img_t = load_latent(data_root / entry["image"], device)
                lbl_t = load_label(data_root / entry["label"], device)
            except Exception as e:
                log(f"  SKIP {entry['image']}: {type(e).__name__}: {e}")
                continue

            images_scaled = img_t * scale_factor
            noise = torch.randn_like(images_scaled)
            timesteps = noise_scheduler.sample_timesteps(images_scaled)
            noisy_latent = noise_scheduler.add_noise(original_samples=images_scaled, noise=noise, timesteps=timesteps)

            controlnet_cond = binarize_labels(lbl_t).float()
            spacing_tensor = torch.FloatTensor(entry["spacing"]).unsqueeze(0).to(device) * 100.0
            modality_tensor = torch.tensor([1], device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
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
                target = images_scaled - noise
                loss = F.l1_loss(model_output.float(), target.float())

            lv = float(loss.detach().cpu())
            if not (lv == lv):  # NaN
                log(f"  NaN loss at step {step} (skipping)")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(controlnet.parameters(), max_norm=1.0)
            if torch.isfinite(gn):
                scaler.step(optimizer)
            scaler.update()
            step += 1
            epoch_losses.append(lv)
            per_case_losses_record[case_key].append(lv)

        epoch_mean = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        epoch_min = float(np.min(epoch_losses)) if epoch_losses else float("nan")
        epoch_losses_record.append({"epoch": epoch, "mean": epoch_mean, "min": epoch_min, "time_s": time.time() - t_epoch})
        log(f"epoch {epoch:3d}/{n_epochs}  mean={epoch_mean:.4f}  min={epoch_min:.4f}  step={step}  t={time.time()-t_epoch:.0f}s")

        # Save loss curve every epoch
        with open(output_dir / "loss_curve.json", "w") as f:
            json.dump({
                "config": {"lr": lr, "n_epochs": n_epochs, "n_subset": len(entries)},
                "per_epoch": epoch_losses_record,
                "per_case_last_5_means": {k: float(np.mean(v[-5:])) if len(v) >= 5 else None for k, v in per_case_losses_record.items()},
            }, f, indent=2)

        # Save ckpt every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == n_epochs - 1:
            torch.save({
                "controlnet_state_dict": controlnet.state_dict(),
                "epoch": epoch,
                "step": step,
            }, output_dir / f"checkpoint_ep{epoch:03d}.pt")
            log(f"saved checkpoint_ep{epoch:03d}.pt")

    log(f"OVERFIT TEST DONE. total {time.time()-t_start:.0f}s, {step} steps")
    log(f"loss curve: {epoch_losses_record[0]['mean']:.4f} (ep 0) → {epoch_losses_record[-1]['mean']:.4f} (ep {n_epochs-1})")


if __name__ == "__main__":
    main()
