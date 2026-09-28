#!/usr/bin/env python
"""Probe aggregator output distribution on one TS test case.

Reports per-channel prob min/max/mean to diagnose why dice ~= 0.
"""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.aggregator import AggregatorConfig, PartialMaskAggregator  # noqa: E402


def main():
    ckpt_path = REPO / "checkpoints/aggregator_v0/aggregator_ep04.pt"
    label_partition = json.loads((REPO / "data/label_partition.json").read_text())
    expert_a = list(label_partition["expert_A_abdominal"])
    expert_b = list(label_partition["expert_B_spine"])
    union_ids = sorted(set(expert_a) | set(expert_b))

    ck = torch.load(str(ckpt_path), map_location="cuda:0", weights_only=False)
    cfg_dict = ck["cfg"]
    cfg = AggregatorConfig(
        n_union_channels=cfg_dict["n_union_channels"],
        base_features=tuple(cfg_dict["base_features"]),
        norm=cfg_dict.get("norm", "instance"),
        act=cfg_dict.get("act", "leakyrelu"),
        dropout=cfg_dict.get("dropout", 0.0),
        use_presence_mask=cfg_dict.get("use_presence_mask", True),
    )
    net = PartialMaskAggregator(cfg).to("cuda:0")
    net.load_state_dict(ck["model_state"])
    net.eval()
    scored_ids = list(ck["scored_ids"])
    id_to_idx = {sid: i for i, sid in enumerate(scored_ids)}
    K = len(scored_ids)
    print(f"K={K}, params={sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")

    ds = json.loads((REPO / "data/phase1_ts_iso15/expert_U_dataset.json").read_text())
    entries = ds["training"]
    entry = entries[-1]  # case_1067 (last test case)
    lpath = REPO / "data/phase1_ts_iso15" / entry["label"]
    lbl = torch.from_numpy(nib.load(str(lpath)).get_fdata().astype(np.int64)).to("cuda:0")
    print(f"label shape: {lbl.shape}, ids present: {sorted(torch.unique(lbl).cpu().tolist())[:20]}...")

    # Build partials: A keeps half its ids, B keeps half its ids
    rng = np.random.default_rng(1234)
    keep_a = rng.random(len(expert_a)) >= 0.3
    keep_b = rng.random(len(expert_b)) >= 0.3
    kept_a = [sid for sid, k in zip(expert_a, keep_a) if k]
    kept_b = [sid for sid, k in zip(expert_b, keep_b) if k]
    pa = torch.zeros_like(lbl)
    pb = torch.zeros_like(lbl)
    for sid in kept_a:
        pa[lbl == sid] = sid
    for sid in kept_b:
        pb[lbl == sid] = sid

    # One-hot
    partial_oh = torch.zeros(1, K, *lbl.shape, device="cuda:0")
    presence = torch.zeros(1, K, *lbl.shape, device="cuda:0")
    for sid in scored_ids:
        idx = id_to_idx[sid]
        in_a = pa == sid
        in_b = pb == sid
        partial_oh[0, idx] = torch.maximum(in_a.float(), in_b.float())
        if in_a.any() or in_b.any():
            presence[0, idx] = 1.0

    # Aggregator forward
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prob = net.posterior_mean(partial_oh, presence)
    prob = prob.float()
    print(f"prob shape: {prob.shape}")
    print(f"prob min/max/mean (global): {prob.min().item():.4f} / {prob.max().item():.4f} / {prob.mean().item():.4f}")
    print()
    print(f"{'idx':<4} {'sid':<5} {'expert':<7} {'observed':<10} {'p_max':<8} {'p_mean':<8} {'p_>0.5_voxels':<14} {'gt_voxels':<10}")
    for idx in range(K):
        sid = scored_ids[idx]
        ch_prob = prob[0, idx]
        is_a = sid in expert_a
        observed = (sid in kept_a) if is_a else (sid in kept_b)
        gt_vox = int((lbl == sid).sum().item())
        p_above = int((ch_prob >= 0.5).sum().item())
        print(f"{idx:<4} {sid:<5} {'A' if is_a else 'B':<7} {str(observed):<10} {ch_prob.max().item():<8.4f} {ch_prob.mean().item():<8.4f} {p_above:<14} {gt_vox:<10}")


if __name__ == "__main__":
    main()
