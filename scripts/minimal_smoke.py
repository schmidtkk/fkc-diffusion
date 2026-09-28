#!/usr/bin/env python3
"""Minimal MAISI ControlNet smoke test — bypass MONAI CacheDataset.

Goal: identify where the silent crash happens. Loads ONE training case manually,
runs ONE forward+backward iteration, prints loss + memory. Verbose at every step.

If this works, the bug is in MAISI's DataLoader pipeline.
If this fails, we see exactly which line dies.
"""

import json
import os
import sys
import time
from pathlib import Path

print(f"[smoke] start at {time.strftime('%X')}", flush=True)

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

MAISI_ROOT = Path(__file__).resolve().parents[1] / "maisi"
sys.path.insert(0, str(MAISI_ROOT))

from scripts.utils import define_instance, binarize_labels
from scripts.diff_model_setting import load_config

print(f"[smoke] imports OK ({time.strftime('%X')})", flush=True)
print(f"[smoke] CUDA available: {torch.cuda.is_available()}", flush=True)

device = torch.device(f"cuda:0")
print(f"[smoke] using device {device}", flush=True)

# 1) Load configs
args = load_config(
    str(MAISI_ROOT / "configs/env_smoke_expert_A.json"),
    str(MAISI_ROOT / "configs/config_smoke_5ep.json"),
    str(MAISI_ROOT / "configs/config_network_rflow.json"),
)
print(f"[smoke] config loaded; trained_diffusion_path={args.trained_diffusion_path}", flush=True)

# 2) Build UNet
print(f"[smoke] building UNet...", flush=True)
unet = define_instance(args, "diffusion_unet_def").to(device)
sd = torch.load(args.trained_diffusion_path, map_location=device, weights_only=False)
unet.load_state_dict(sd["unet_state_dict"], strict=False)
scale_factor = sd["scale_factor"].to(device) if hasattr(sd["scale_factor"], "to") else torch.tensor(sd["scale_factor"]).to(device)
unet.eval()
for p in unet.parameters():
    p.requires_grad_(False)
print(f"[smoke] UNet loaded, scale_factor={scale_factor.item():.4f}; GPU mem {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

# 3) Build ControlNet (initialize from UNet weights + load pretrained)
print(f"[smoke] building ControlNet...", flush=True)
controlnet = define_instance(args, "controlnet_def").to(device)
from monai.networks.utils import copy_model_state
copy_model_state(controlnet, unet.state_dict())
ctrl_sd = torch.load(args.existing_ckpt_filepath, map_location=device, weights_only=False)
controlnet.load_state_dict(ctrl_sd["controlnet_state_dict"], strict=False)
print(f"[smoke] ControlNet loaded; GPU mem {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

# 4) Build noise scheduler
noise_scheduler = define_instance(args, "noise_scheduler")
print(f"[smoke] noise scheduler built: {type(noise_scheduler).__name__}", flush=True)

# 5) Load one training entry MANUALLY (no MONAI loader)
data_root = Path(__file__).resolve().parents[1] / "data/maisi_controlnet"
with open(data_root / "expert_A_abd_dataset.json") as f:
    dataset = json.load(f)["training"]
entry = dataset[0]
print(f"[smoke] using entry: {entry}", flush=True)

image_path = data_root / entry["image"]
label_path = data_root / entry["label"]

# Load image (latent: shape (X, Y, Z, C=4) per channel-last from prep)
print(f"[smoke] loading image latent from {image_path}", flush=True)
img_nii = nib.load(str(image_path))
img_arr = img_nii.get_fdata().astype(np.float32)
print(f"[smoke]   image shape: {img_arr.shape} (channel-last expected: (X, Y, Z, 4))", flush=True)
# Convert to (1, C, X, Y, Z)
img_t = torch.from_numpy(img_arr).permute(3, 0, 1, 2).unsqueeze(0).to(device)
print(f"[smoke]   image tensor shape: {img_t.shape}, mem now {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

# Load label (int volume at full resolution)
print(f"[smoke] loading label from {label_path}", flush=True)
lbl_nii = nib.load(str(label_path))
lbl_arr = lbl_nii.get_fdata().astype(np.int64)
print(f"[smoke]   label shape: {lbl_arr.shape}", flush=True)
lbl_t = torch.from_numpy(lbl_arr).unsqueeze(0).unsqueeze(0).to(device)
print(f"[smoke]   label tensor shape: {lbl_t.shape}, mem now {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

# 6) Binarize labels
print(f"[smoke] binarizing labels...", flush=True)
controlnet_cond = binarize_labels(lbl_t).float()
print(f"[smoke]   controlnet_cond shape: {controlnet_cond.shape}, mem {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

# 7) Sample timestep + noise
B = 1
timesteps = torch.tensor([500], device=device, dtype=torch.long)
noise = torch.randn_like(img_t)
print(f"[smoke] noise + timesteps prepared", flush=True)

# 8) Apply noise schedule
images_scaled = img_t * scale_factor
noisy_latent = noise_scheduler.add_noise(original_samples=images_scaled, noise=noise, timesteps=timesteps)
print(f"[smoke] noise added, noisy_latent shape: {noisy_latent.shape}", flush=True)

# 9) Modality + spacing tensors
spacing_tensor = torch.FloatTensor(entry["spacing"]).unsqueeze(0).to(device) * 100.0
modality_tensor = torch.tensor([1], device=device, dtype=torch.long)  # 1 = ct
print(f"[smoke] meta tensors built", flush=True)

# 10) ControlNet forward
print(f"[smoke] ControlNet forward...", flush=True)
t0 = time.time()
with torch.amp.autocast("cuda"):
    down, mid = controlnet(
        x=noisy_latent,
        timesteps=timesteps,
        controlnet_cond=controlnet_cond,
        class_labels=modality_tensor,
    )
print(f"[smoke]   ControlNet forward done in {time.time()-t0:.2f}s; mid shape {mid.shape}, down has {len(down)} blocks", flush=True)
print(f"[smoke]   GPU mem after controlnet: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

# 11) UNet forward
print(f"[smoke] UNet forward (with controlnet residuals)...", flush=True)
t0 = time.time()
with torch.amp.autocast("cuda"):
    model_output = unet(
        x=noisy_latent,
        timesteps=timesteps,
        spacing_tensor=spacing_tensor,
        down_block_additional_residuals=down,
        mid_block_additional_residual=mid,
        class_labels=modality_tensor,
    )
print(f"[smoke]   UNet forward done in {time.time()-t0:.2f}s; output shape {model_output.shape}", flush=True)
print(f"[smoke]   GPU mem after UNet: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

# 12) Loss
target = images_scaled - noise  # rectified flow target
loss = F.l1_loss(model_output.float(), target.float())
print(f"[smoke] loss = {loss.item():.4f}", flush=True)

print(f"\n[smoke] SUCCESS — full forward path works on case {entry['image']}", flush=True)
print(f"[smoke] peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
