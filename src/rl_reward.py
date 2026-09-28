"""Reward utilities for Flow-GRPO ControlNet fine-tuning."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet

PATCH = (96, 96, 96)
CORE_MAP = {1: 1, 3: 2, 5: 3, 14: 4, 6: 5, 37: 6, 36: 7, 35: 8, 34: 9, 33: 10}
N_CLASSES = 11

_REWARD_NET: Optional[torch.nn.Module] = None


def load_reward_segmenter(ckpt: str, device: torch.device | str) -> SegResNet:
    """Load the fast SegResNet reward model checkpoint."""
    device = torch.device(device)
    net = SegResNet(spatial_dims=3, in_channels=1, out_channels=N_CLASSES, init_filters=16).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state)
    net.eval()
    return net


def set_reward_segmenter(net: Optional[torch.nn.Module]) -> None:
    """Install a process-local reward net for dice_cf_reward calls."""
    global _REWARD_NET
    _REWARD_NET = net


def latent_to_hu(ae: torch.nn.Module, latent: torch.Tensor, scale_factor: torch.Tensor | float) -> torch.Tensor:
    """Decode MAISI latent to HU using the repo's eval convention."""
    if not torch.is_tensor(scale_factor):
        scale_factor = torch.tensor(scale_factor, device=latent.device, dtype=latent.dtype)
    lat = latent / scale_factor.to(device=latent.device, dtype=latent.dtype)
    ct = torch.clip(ae.decode_stage_2_outputs(lat), 0.0, 1.0)
    return ct * 2000.0 - 1000.0


def _normalize_hu(hu_ct: torch.Tensor) -> torch.Tensor:
    return torch.clamp((hu_ct + 1000.0) / 2000.0, 0.0, 1.0)


def remap_label_ids_to_core(label_ids: torch.Tensor) -> torch.Tensor:
    """Map TotalSegmentator IDs to contiguous CORE class IDs used by reward net."""
    out = torch.zeros_like(label_ids, dtype=torch.long)
    for label_id, core_id in CORE_MAP.items():
        out = torch.where(label_ids.long() == label_id, torch.full_like(out, core_id), out)
    return out


def _hard_dice(pred: torch.Tensor, target: torch.Tensor, cls: int) -> torch.Tensor:
    p = pred == cls
    g = target == cls
    denom = p.sum(dtype=torch.float32) + g.sum(dtype=torch.float32)
    if float(denom.item()) == 0.0:
        return torch.tensor(float("nan"), device=pred.device)
    return 2.0 * (p & g).sum(dtype=torch.float32) / (denom + 1e-8)


def dice_cf_reward(
    hu_ct: torch.Tensor,
    cond_label_ids_volume: torch.Tensor,
    present_core_ids: Optional[Sequence[int]] = None,
    segmenter: Optional[torch.nn.Module] = None,
) -> float:
    """Terminal label-adherence reward from SegResNet predictions vs condition mask."""
    net = segmenter if segmenter is not None else _REWARD_NET
    if net is None:
        raise RuntimeError("Reward segmenter is not loaded; call load_reward_segmenter or set_reward_segmenter")
    device = next(net.parameters()).device
    x = hu_ct.to(device=device, dtype=torch.float32)
    if x.dim() == 4:
        x = x.unsqueeze(1)
    if x.shape[1] != 1:
        raise ValueError(f"Expected CT tensor with one channel, got shape {tuple(x.shape)}")
    x = _normalize_hu(x)
    with torch.no_grad():
        pred = sliding_window_inference(x, PATCH, 4, net, overlap=0.25).argmax(1)

    target = cond_label_ids_volume.to(device=device)
    if target.dim() == 5 and target.shape[1] == 1:
        target = target[:, 0]
    elif target.dim() == 3:
        target = target.unsqueeze(0)
    target_core = remap_label_ids_to_core(target)

    if present_core_ids is None:
        present_core_ids = sorted(int(v) for v in torch.unique(target_core).detach().cpu().tolist() if int(v) > 0)
    dices = [_hard_dice(pred, target_core, int(cls)) for cls in present_core_ids]
    valid = [d for d in dices if not torch.isnan(d)]
    if not valid:
        return 0.0
    return float(torch.stack(valid).mean().detach().cpu().item())
