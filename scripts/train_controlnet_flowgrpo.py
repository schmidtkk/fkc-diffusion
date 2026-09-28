#!/usr/bin/env python3
"""Flow-GRPO online-RL fine-tuning for medgen-fkc MAISI ControlNet.

This script deliberately does not start GPU/model loading by default.  Use
``--dry-run`` for the CPU synthetic shape check used in CI/dev, or pass
``--run-real`` explicitly in the external training system to execute real RL.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import nibabel as nib
import numpy as np
import torch
from torch.amp import GradScaler, autocast

REPO_ROOT = Path(__file__).resolve().parents[1]
MAISI_ROOT = REPO_ROOT / "maisi"
sys.path.insert(0, str(MAISI_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from src.flow_grpo_sampler import (  # noqa: E402
    group_relative_advantage,
    ppo_clipped_loss,
    sde_rollout,
    step_logprob,
)
from src.rl_reward import dice_cf_reward, latent_to_hu, load_reward_segmenter  # noqa: E402

CORE_MAP = {1: 1, 3: 2, 5: 3, 14: 4, 6: 5, 37: 6, 36: 7, 35: 8, 34: 9, 33: 10}


def log(msg: str) -> None:
    print(f"[{time.strftime('%X')}] {msg}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Flow-GRPO fine-tuning for MAISI ControlNet")
    ap.add_argument("--env-config", default=None)
    ap.add_argument("--model-config", default=None)
    ap.add_argument("--network-config", default=None)
    ap.add_argument("--cond-encoding", choices=["binarize", "onehot"], default="binarize")
    ap.add_argument("--reward-ckpt", default=None, help="SegResNet state_dict checkpoint for Dice_CF reward")
    ap.add_argument("--out-dir", default="outputs/flowgrpo")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--rollout-steps", type=int, default=10)
    ap.add_argument("--sde-sigma", type=float, default=0.3)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--kl-beta", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--adv-mask", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dry-run", action="store_true", help="CPU tiny tensor rollout+loss check; no real model loading")
    ap.add_argument("--run-real", action="store_true", help="Explicit opt-in required for real GPU model loading/training")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--latent-shape", default="4,64,64,64", help="C,D,H,W latent shape")
    ap.add_argument("--max-conditions", type=int, default=0, help="Limit conditions per epoch; 0 = all")
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--continual-method", choices=["none", "ewc", "lwf"], default="none")
    ap.add_argument("--continual-lambda", type=float, default=0.0)
    ap.add_argument("--continual-anchor", default=None)
    return ap


class TinyScheduler:
    num_train_timesteps = 100

    def __init__(self, n_steps: int):
        self.timesteps = [float(t) for t in torch.linspace(99, 0, n_steps + 1).tolist()]

    def step(self, model_output, timestep, sample, next_timestep=None):
        dt = (float(timestep) - float(next_timestep)) / self.num_train_timesteps
        return sample + model_output * dt, sample


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.05))

    def velocity(self, x_t, t, cond, spacing, modality, unet, controlnet, scheduler):
        del t, cond, spacing, modality, unet, scheduler
        return x_t * 0.0 + controlnet.weight


def dry_run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    policy = TinyPolicy()
    scheduler = TinyScheduler(args.rollout_steps)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    cond = torch.zeros(1, 1, 2, 2, 2)
    rollout = sde_rollout(
        None,
        policy,
        scheduler,
        1.0,
        cond,
        None,
        None,
        sigma=args.sde_sigma,
        n_steps=args.rollout_steps,
        group_size=args.group_size,
        generator=generator,
        device="cpu",
        latent_shape=(1, 2, 2, 2),
        velocity_fn=policy.velocity,
    )
    rewards = -rollout["x0"].flatten(1).pow(2).mean(dim=1)
    adv = group_relative_advantage(rewards, args.adv_mask)
    losses = []
    ratios = []
    for tr in rollout["traj"]:
        logp_new = step_logprob(
            None,
            policy,
            scheduler,
            tr["x_t"],
            tr["x_next"],
            tr["t"],
            tr["next_t"],
            args.sde_sigma,
            cond,
            None,
            None,
            velocity_fn=policy.velocity,
        )
        loss, ratio = ppo_clipped_loss(logp_new, tr["logp_old"], adv, args.clip_eps)
        losses.append(loss)
        ratios.append(ratio)
    total_loss = torch.stack(losses).mean()
    total_loss.backward()
    ratio_all = torch.cat(ratios)
    print(
        "DRY_RUN PASS "
        f"traj_steps={len(rollout['traj'])} x0_shape={tuple(rollout['x0'].shape)} "
        f"loss={float(total_loss.detach()):.6f} ratio_mean={float(ratio_all.mean()):.6f} "
        f"grad={float(policy.weight.grad.detach()):.6f}",
        flush=True,
    )


def parse_latent_shape(text: str) -> Tuple[int, int, int, int]:
    vals = tuple(int(v) for v in text.split(","))
    if len(vals) != 4:
        raise ValueError("--latent-shape must be C,D,H,W")
    return vals


def load_latent(path: Path, device: torch.device, scale_factor: torch.Tensor) -> torch.Tensor:
    arr = nib.load(str(path)).get_fdata().astype(np.float32)
    return torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).to(device) * scale_factor


def load_label(path: Path, device: torch.device) -> torch.Tensor:
    arr = nib.load(str(path)).get_fdata().astype(np.int64)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def resolve_path(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base / p


def present_core_ids(label: torch.Tensor) -> List[int]:
    values = set(int(v) for v in torch.unique(label).detach().cpu().tolist())
    return sorted(core for label_id, core in CORE_MAP.items() if label_id in values)


def make_condition_encoder(args: argparse.Namespace):
    from scripts.utils import binarize_labels
    from src.cond_encoding import to_one_hot as _to_one_hot

    if args.cond_encoding == "onehot":
        def encode(lbl: torch.Tensor) -> torch.Tensor:
            return _to_one_hot(lbl.long()).float()
        return encode

    def encode(lbl: torch.Tensor) -> torch.Tensor:
        return binarize_labels(lbl.long()).float()
    return encode


def load_real_components(args: argparse.Namespace):
    from monai.networks.schedulers import RFlowScheduler
    from monai.networks.utils import copy_model_state
    from scripts.diff_model_setting import load_config
    from scripts.utils import define_instance

    if not (args.env_config and args.model_config and args.network_config):
        raise ValueError("--env-config, --model-config, and --network-config are required with --run-real")
    if not args.reward_ckpt:
        raise ValueError("--reward-ckpt is required with --run-real")

    cfg = load_config(args.env_config, args.model_config, args.network_config)
    device = torch.device(args.device)
    log("building frozen UNet ...")
    unet = define_instance(cfg, "diffusion_unet_def").to(device)
    sd = torch.load(cfg.trained_diffusion_path, map_location=device, weights_only=False)
    unet.load_state_dict(sd["unet_state_dict"], strict=False)
    scale_factor = sd["scale_factor"].to(device) if hasattr(sd["scale_factor"], "to") else torch.tensor(sd["scale_factor"], device=device)
    unet.eval().requires_grad_(False)

    log("building frozen VAE ...")
    ae = define_instance(cfg, "autoencoder_def").to(device)
    ae_sd = torch.load(cfg.trained_autoencoder_path, map_location=device, weights_only=False)
    if isinstance(ae_sd, dict) and "unet_state_dict" in ae_sd:
        ae_sd = ae_sd["unet_state_dict"]
    ae.load_state_dict(ae_sd)
    ae.eval().requires_grad_(False)
    for m in ae.modules():
        if hasattr(m, "norm_float16"):
            m.norm_float16 = False

    log("building trainable ControlNet ...")
    controlnet = define_instance(cfg, "controlnet_def").to(device)
    copy_model_state(controlnet, unet.state_dict())
    if cfg.existing_ckpt_filepath and Path(cfg.existing_ckpt_filepath).exists():
        ck = torch.load(cfg.existing_ckpt_filepath, map_location=device, weights_only=False)
        controlnet.load_state_dict(ck["controlnet_state_dict"], strict=False)
        log(f"loaded ControlNet warm start {cfg.existing_ckpt_filepath}")
    controlnet.train().requires_grad_(True)

    ref_controlnet = None
    if args.kl_beta > 0:
        ref_controlnet = deepcopy(controlnet).eval().requires_grad_(False)

    scheduler = define_instance(cfg, "noise_scheduler")
    try:
        scheduler.set_timesteps(num_inference_steps=args.rollout_steps, input_img_size_numel=64 ** 3)
    except TypeError:
        scheduler.set_timesteps(num_inference_steps=args.rollout_steps)
    if not isinstance(scheduler, RFlowScheduler):
        log(f"WARN: scheduler is {type(scheduler).__name__}; expected MONAI RFlowScheduler-compatible step")

    reward_net = load_reward_segmenter(args.reward_ckpt, device)
    data_root = Path(cfg.data_base_dir[0]) if isinstance(cfg.data_base_dir, list) else Path(cfg.data_base_dir)
    json_path = cfg.json_data_list[0] if isinstance(cfg.json_data_list, list) else cfg.json_data_list
    with open(json_path) as f:
        entries = json.load(f)["training"]
    if args.max_conditions > 0:
        entries = entries[: args.max_conditions]
    return cfg, device, unet, ae, controlnet, ref_controlnet, scheduler, reward_net, data_root, entries, scale_factor


def save_checkpoint(controlnet: torch.nn.Module, out_dir: Path, epoch: int, iteration: int, extra: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {"controlnet_state_dict": controlnet.state_dict(), "epoch": epoch, "iter": iteration, **extra}
    torch.save(ckpt, out_dir / f"checkpoint_flowgrpo_e{epoch:03d}_i{iteration:06d}.pt")
    torch.save(ckpt, out_dir / "checkpoint_latest.pt")


def run_real(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    latent_shape = parse_latent_shape(args.latent_shape)
    (
        cfg,
        device,
        unet,
        ae,
        controlnet,
        ref_controlnet,
        scheduler,
        reward_net,
        data_root,
        entries,
        scale_factor,
    ) = load_real_components(args)
    encode_cond = make_condition_encoder(args)
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=device.type == "cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    global_iter = 0

    for epoch in range(args.epochs):
        random.shuffle(entries)
        for entry in entries[: args.iters if args.iters > 0 else len(entries)]:
            global_iter += 1
            label_path = resolve_path(data_root, entry.get("label") or entry.get("label_path"))
            label = load_label(label_path, device)
            cond = encode_cond(label)
            sp = entry.get("spacing", [1.5, 1.5, 1.5])
            spacing = torch.tensor([sp] * args.group_size, device=device, dtype=torch.float32) * 100.0
            modality = torch.ones((args.group_size,), device=device, dtype=torch.long)
            rollout = sde_rollout(
                unet,
                controlnet,
                scheduler,
                scale_factor,
                cond,
                spacing,
                modality,
                sigma=args.sde_sigma,
                n_steps=args.rollout_steps,
                group_size=args.group_size,
                generator=generator,
                device=device,
                latent_shape=latent_shape,
            )

            rewards = []
            core_ids = present_core_ids(label)
            with torch.no_grad():
                for latent in rollout["x0"]:
                    hu = latent_to_hu(ae, latent.unsqueeze(0), scale_factor)
                    rewards.append(dice_cf_reward(hu, label, core_ids, segmenter=reward_net))
            reward_t = torch.tensor(rewards, device=device, dtype=torch.float32)
            advantage = group_relative_advantage(reward_t, args.adv_mask)

            optimizer.zero_grad(set_to_none=True)
            step_loss_values = []
            ratios = []
            n_traj = len(rollout["traj"])
            for step_idx, tr in enumerate(rollout["traj"]):
                with autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logp_new = step_logprob(
                        unet, controlnet, scheduler, tr["x_t"], tr["x_next"], tr["t"], tr["next_t"],
                        args.sde_sigma, cond, spacing, modality,
                    )
                    loss, ratio = ppo_clipped_loss(logp_new, tr["logp_old"], advantage, args.clip_eps)
                    if args.kl_beta > 0 and ref_controlnet is not None:
                        with torch.no_grad():
                            logp_ref = step_logprob(
                                unet, ref_controlnet, scheduler, tr["x_t"], tr["x_next"], tr["t"], tr["next_t"],
                                args.sde_sigma, cond, spacing, modality,
                            )
                        loss = loss + args.kl_beta * (logp_ref - logp_new).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite Flow-GRPO loss at iter {global_iter}, step {step_idx}: {loss.detach()}"
                        )
                    step_loss = loss / n_traj
                scaler.scale(step_loss).backward()
                step_loss_values.append(loss.detach())
                ratios.append(ratio)
            total_loss = torch.stack(step_loss_values).mean()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(controlnet.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            ratio_all = torch.cat(ratios)
            mem = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else 0.0
            log(
                f"iter={global_iter} mean_reward={float(reward_t.mean()):.4f} "
                f"loss={float(total_loss.detach()):.6f} ratio={float(ratio_all.mean()):.4f} gpu_mem_gb={mem:.2f}"
            )
            if global_iter % args.save_every == 0:
                save_checkpoint(
                    controlnet,
                    out_dir,
                    epoch,
                    global_iter,
                    {"mean_reward": float(reward_t.mean().detach().cpu()), "args": vars(args)},
                )
            if args.iters > 0 and global_iter >= args.iters * args.epochs:
                return


def main() -> None:
    args = build_argparser().parse_args()
    if args.dry_run:
        dry_run(args)
        return
    if not args.run_real:
        print("Flow-GRPO trainer configured. Pass --dry-run for CPU shape check or --run-real in the external training system.", flush=True)
        return
    run_real(args)


if __name__ == "__main__":
    main()
