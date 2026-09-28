"""Flow-GRPO rollout and log-probability helpers for MAISI ControlNet.

The functions here are intentionally small and side-effect free so the Gaussian
math can be unit-tested on CPU with stub models.  Real MAISI usage goes through
``velocity_from_controlnet``; tests may pass ``velocity_fn`` to avoid loading GPU
models.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from src.fkc_poe_sampler import _unet_velocity

Tensor = torch.Tensor
VelocityFn = Callable[[Tensor, Any, Any, Any, Any, Any, Any, Any], Tensor]


def _as_float_timestep(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().flatten()[0].item())
    return float(value)


def timestep_dt(t: Any, next_t: Any, scheduler: Any) -> float:
    """Return positive rectified-flow dt in training-timestep units."""
    num_train = float(getattr(scheduler, "num_train_timesteps", 1000))
    dt = (_as_float_timestep(t) - _as_float_timestep(next_t)) / num_train
    if dt <= 0:
        raise ValueError(f"Expected t > next_t for denoising, got t={t} next_t={next_t}")
    return dt


def gaussian_logprob(x_next: Tensor, mean: Tensor, var: float | Tensor) -> Tensor:
    """Per-sample diagonal Gaussian log-density summed over non-batch dims."""
    if not torch.is_tensor(var):
        var_t = torch.tensor(var, dtype=mean.dtype, device=mean.device)
    else:
        var_t = var.to(device=mean.device, dtype=mean.dtype)
    log_terms = ((x_next - mean) ** 2) / var_t + torch.log(2.0 * torch.pi * var_t)
    return -0.5 * log_terms.flatten(1).sum(dim=1)


def group_relative_advantage(rewards: Tensor, adv_mask: bool = False, eps: float = 1e-8) -> Tensor:
    """Normalize rewards within a rollout group and optionally mask below-mean samples."""
    rewards = rewards.float()
    mean = rewards.mean()
    adv = (rewards - mean) / (rewards.std(unbiased=False) + eps)
    if adv_mask:
        adv = torch.where(rewards >= mean, adv, torch.zeros_like(adv))
    return adv


def ppo_clipped_loss(logp_new: Tensor, logp_old: Tensor, advantage: Tensor, clip_eps: float) -> Tuple[Tensor, Tensor]:
    """Return PPO clipped policy loss and detached importance ratios."""
    advantage = advantage.to(device=logp_new.device, dtype=logp_new.dtype)
    ratio = torch.exp(logp_new - logp_old.to(device=logp_new.device, dtype=logp_new.dtype))
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    return -torch.minimum(unclipped, clipped).mean(), ratio.detach()


def velocity_from_controlnet(
    unet: torch.nn.Module,
    controlnet: torch.nn.Module,
    x_t: Tensor,
    t: Any,
    cond: Tensor,
    spacing: Optional[Tensor],
    modality: Optional[Tensor],
    include_modality: bool = True,
    include_body_region: bool = False,
    top_region_index_tensor: Optional[Tensor] = None,
    bottom_region_index_tensor: Optional[Tensor] = None,
) -> Tensor:
    """Run MAISI ControlNet residuals and frozen UNet velocity for one step.

    MAISI configs in this repo generally use spacing and optional modality.  Body
    region tensors are accepted for future configs but default off to match the
    existing FKC-PoE sampler usage.
    """
    batch = x_t.shape[0]
    timestep = _as_float_timestep(t)
    timestep_tensor = torch.full((batch,), timestep, device=x_t.device, dtype=x_t.dtype)
    cond_batch = cond.to(device=x_t.device, dtype=x_t.dtype)
    if cond_batch.shape[0] == 1 and batch != 1:
        cond_batch = cond_batch.expand(batch, *cond_batch.shape[1:])

    controlnet_inputs = {
        "x": x_t,
        "timesteps": timestep_tensor,
        "controlnet_cond": cond_batch,
    }
    if include_modality and modality is not None:
        controlnet_inputs["class_labels"] = modality.to(device=x_t.device)
    down, mid = controlnet(**controlnet_inputs)

    if spacing is None:
        spacing = torch.ones((batch, 3), device=x_t.device, dtype=x_t.dtype)
    elif spacing.shape[0] == 1 and batch != 1:
        spacing = spacing.expand(batch, *spacing.shape[1:])
    if modality is None:
        modality = torch.zeros((batch,), device=x_t.device, dtype=torch.long)
    elif modality.shape[0] == 1 and batch != 1:
        modality = modality.expand(batch)
    if top_region_index_tensor is None:
        top_region_index_tensor = torch.zeros((batch,), device=x_t.device, dtype=torch.long)
    if bottom_region_index_tensor is None:
        bottom_region_index_tensor = torch.zeros((batch,), device=x_t.device, dtype=torch.long)

    return _unet_velocity(
        x_t,
        timestep,
        spacing.to(device=x_t.device, dtype=x_t.dtype),
        modality.to(device=x_t.device),
        top_region_index_tensor.to(device=x_t.device),
        bottom_region_index_tensor.to(device=x_t.device),
        down,
        mid,
        unet,
        include_modality and modality is not None,
        include_body_region,
    )


def scheduler_mean(scheduler: Any, velocity: Tensor, t: Any, x_t: Tensor, next_t: Any) -> Tensor:
    """Compute rectified-flow deterministic mean using MONAI scheduler.step."""
    mean, _ = scheduler.step(velocity, t, x_t, next_timestep=next_t)
    return mean


def step_logprob(
    unet: Optional[torch.nn.Module],
    controlnet: Optional[torch.nn.Module],
    scheduler: Any,
    x_t: Tensor,
    x_next: Tensor,
    t: Any,
    next_t: Any,
    sigma: float,
    cond: Optional[Tensor],
    spacing: Optional[Tensor],
    modality: Optional[Tensor],
    velocity_fn: Optional[VelocityFn] = None,
    **velocity_kwargs: Any,
) -> Tensor:
    """Gaussian log-probability of one SDE transition, with grad through mean."""
    if sigma <= 0:
        raise ValueError("sigma must be > 0 for SDE log-probability")
    if velocity_fn is None:
        if unet is None or controlnet is None or cond is None:
            raise ValueError("unet, controlnet, and cond are required without velocity_fn")
        velocity = velocity_from_controlnet(
            unet, controlnet, x_t, t, cond, spacing, modality, **velocity_kwargs
        )
    else:
        velocity = velocity_fn(x_t, t, cond, spacing, modality, unet, controlnet, scheduler)
    mean = scheduler_mean(scheduler, velocity, t, x_t, next_t)
    var = (sigma ** 2) * timestep_dt(t, next_t, scheduler)
    return gaussian_logprob(x_next, mean, var)


def _select_rollout_timesteps(scheduler: Any, n_steps: int) -> Sequence[Any]:
    timesteps = list(getattr(scheduler, "timesteps", []))
    if len(timesteps) < n_steps + 1:
        try:
            scheduler.set_timesteps(num_inference_steps=n_steps, input_img_size_numel=64 ** 3)
        except TypeError:
            scheduler.set_timesteps(num_inference_steps=n_steps)
        timesteps = list(getattr(scheduler, "timesteps", []))
    if len(timesteps) < n_steps + 1:
        raise ValueError(f"scheduler produced {len(timesteps)} timesteps, need at least {n_steps + 1}")
    return timesteps[: n_steps + 1]


def sde_rollout(
    unet: Optional[torch.nn.Module],
    controlnet: Optional[torch.nn.Module],
    scheduler: Any,
    scale_factor: Any,
    cond: Tensor,
    spacing: Optional[Tensor],
    modality: Optional[Tensor],
    sigma: float,
    n_steps: int,
    group_size: int,
    generator: Optional[torch.Generator],
    device: torch.device | str,
    latent_shape: Tuple[int, int, int, int] = (4, 64, 64, 64),
    velocity_fn: Optional[VelocityFn] = None,
    **velocity_kwargs: Any,
) -> Dict[str, Any]:
    """Sample a group of SDE trajectories under no_grad and store transition data."""
    del scale_factor  # Sampling is in scaled latent space; decode handles scale_factor later.
    device = torch.device(device)
    if sigma <= 0:
        raise ValueError("sigma must be > 0 for SDE rollout")
    if n_steps <= 0 or group_size <= 0:
        raise ValueError("n_steps and group_size must be positive")

    timesteps = _select_rollout_timesteps(scheduler, n_steps)
    latents = torch.randn((group_size, *latent_shape), generator=generator, device=device)
    cond = cond.to(device)
    if spacing is not None:
        spacing = spacing.to(device)
    if modality is not None:
        modality = modality.to(device)

    traj: List[Dict[str, Tensor | Any]] = []
    with torch.no_grad():
        for step_idx in range(n_steps):
            t = timesteps[step_idx]
            next_t = timesteps[step_idx + 1]
            if velocity_fn is None:
                velocity = velocity_from_controlnet(
                    unet, controlnet, latents, t, cond, spacing, modality, **velocity_kwargs
                )
            else:
                velocity = velocity_fn(latents, t, cond, spacing, modality, unet, controlnet, scheduler)
            mean = scheduler_mean(scheduler, velocity, t, latents, next_t)
            dt = timestep_dt(t, next_t, scheduler)
            noise = torch.randn(mean.shape, generator=generator, device=device, dtype=mean.dtype)
            x_next = mean + sigma * math.sqrt(dt) * noise
            logp_old = gaussian_logprob(x_next, mean, (sigma ** 2) * dt).detach()
            traj.append({
                "x_t": latents.detach().clone(),
                "x_next": x_next.detach().clone(),
                "t": t,
                "next_t": next_t,
                "logp_old": logp_old,
            })
            latents = x_next
    return {"traj": traj, "x0": latents.detach().clone()}
