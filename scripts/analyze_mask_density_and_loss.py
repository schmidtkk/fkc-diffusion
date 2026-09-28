#!/usr/bin/env python3
"""Mask density analysis + ControlNet L1 loss correlation report.

Compares:
  - KiTS demo dataset (MAISI combined labels, 93 classes, dense)
  - TotalSegmentator Expert A (4-ch one-hot, sparse)

For TS dataset: runs ControlNet forward pass with real vs null mask,
records L1 loss vs mask fg_ratio to test whether conditioning is effective.

Output: reports/mask_density_loss_report.html
"""
import base64
import io
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAISI_ROOT   = PROJECT_ROOT / "maisi"
sys.path.insert(0, str(MAISI_ROOT))

from scripts.utils import define_instance
from scripts.diff_model_setting import load_config
from monai.networks.utils import copy_model_state
from monai.networks.schedulers import RFlowScheduler

# ─────────────────────────────── constants ───────────────────────────────────

KITS_ROOT  = PROJECT_ROOT / "maisi/dataset/C4KC-KiTS_subset"
TS_ROOT    = PROJECT_ROOT / "data/maisi_controlnet_iso10"
CKPT_PATH  = PROJECT_ROOT / "checkpoints/expert_A_long/checkpoint_latest.pt"
ENV_CFG    = PROJECT_ROOT / "configs/env_iso10_expert_A_long.json"
MODEL_CFG  = MAISI_ROOT   / "configs/config_maisi_controlnet_train_rflow-ct.json"
NET_CFG    = PROJECT_ROOT / "configs/config_network_rflow_4ch.json"
OUT_HTML   = PROJECT_ROOT / "reports/mask_density_loss_report.html"

CHANNEL_MAP = {1: 0, 3: 1, 5: 2, 14: 3}   # liver, spleen, kidney_R, kidney_L
CH_NAMES    = ["liver", "spleen", "kidney_R", "kidney_L"]
N_LOSS_CASES = 30   # TS cases to probe for L1 loss

# MAISI label vocabulary (subset shown in table)
MAISI_LABELS = {
    1:"spleen", 2:"kidney_R", 3:"kidney_L", 4:"gallbladder", 5:"liver",
    6:"stomach", 7:"aorta", 8:"IVC", 9:"pancreas", 10:"adrenal_R",
    11:"adrenal_L", 12:"lung_UL_L", 13:"lung_LL_L", 14:"lung_UL_R",
    15:"lung_ML_R", 16:"lung_LL_R", 17:"esophagus", 18:"trachea",
    19:"thyroid", 22:"brain", 25:"kidney_cyst_R", 26:"kidney_cyst_L",
    27:"spinal_cord", 28:"vert_C1", 29:"vert_C2", 30:"vert_C3",
    31:"vert_C4", 32:"vert_C5", 33:"vert_C6", 34:"vert_C7",
    35:"vert_T1", 36:"vert_T2", 37:"vert_T3", 38:"vert_T4",
    39:"vert_T5", 40:"vert_T6", 41:"vert_T7", 42:"vert_T8",
    43:"vert_T9", 44:"vert_T10", 45:"vert_T11", 46:"vert_T12",
    47:"vert_L1", 48:"vert_L2", 49:"vert_L3", 50:"vert_L4",
    51:"vert_L5", 52:"sacrum", 57:"clavicula_L", 58:"clavicula_R",
    59:"scapula_L", 60:"scapula_R", 61:"humerus_L", 62:"humerus_R",
    63:"femur_L", 64:"femur_R", 65:"hip_L", 66:"hip_R",
    79:"iliac_artery_L", 80:"iliac_artery_R", 81:"iliac_vena_L", 82:"iliac_vena_R",
    83:"portal_vein", 84:"celiac_trunk", 93:"costal_cartilages",
    94:"autochthon_L", 95:"autochthon_R", 96:"iliopsoas_L", 97:"iliopsoas_R",
    98:"glut_max_L", 99:"glut_max_R", 100:"glut_med_L", 101:"glut_med_R",
    102:"glut_min_L", 103:"glut_min_R",
    104:"heart", 105:"lung_L", 106:"lung_R",
    107:"pulm_artery", 108:"atrium_append_L",
    114:"rib_L1", 115:"rib_R1",
    121:"sternum", 122:"vertebrae_body",
    127:"rib_L_all", 132:"rib_R_all",
    200:"body_region (MAISI special)",
}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────── Section 1: KiTS density ─────────────────────────

def compute_kits_density():
    log("Computing KiTS mask density ...")
    files_orig = sorted(KITS_ROOT.glob("*/mask_zs*.nii.gz"))
    files_comb = sorted(KITS_ROOT.glob("*/mask_combined_label*.nii.gz"))

    def stats_for(files):
        per_case = []
        agg = defaultdict(int)
        total_vox = 0
        for f in files:
            arr = nib.load(f).get_fdata(dtype=np.float32).astype(np.int32)
            n = arr.size
            total_vox += n
            u, c = np.unique(arr, return_counts=True)
            for ui, ci in zip(u.tolist(), c.tolist()):
                agg[ui] += ci
            fg_ids = [int(x) for x in u if x != 0]
            bg = int(c[u == 0][0]) if 0 in u else 0
            fg_ratio = 1 - bg / n
            per_case.append({
                "case": f.parent.name,
                "shape": list(arr.shape),
                "n_fg_classes": len(fg_ids),
                "fg_ratio": fg_ratio,
                "fg_ids": sorted(fg_ids),
            })
        fg_labels = {k: v for k, v in agg.items() if k != 0}
        per_label = {k: v / total_vox for k, v in sorted(fg_labels.items())}
        return per_case, per_label, total_vox, agg[0] / total_vox

    orig, orig_per_label, orig_total, orig_bg = stats_for(files_orig)
    comb, comb_per_label, comb_total, comb_bg = stats_for(files_comb)
    return {
        "orig": {"per_case": orig, "per_label": orig_per_label,
                 "total_vox": orig_total, "bg_ratio": orig_bg,
                 "n_files": len(files_orig)},
        "comb": {"per_case": comb, "per_label": comb_per_label,
                 "total_vox": comb_total, "bg_ratio": comb_bg,
                 "n_files": len(files_comb)},
    }


# ─────────────────────── Section 2: TS (Expert A) density ────────────────────

def compute_ts_density(entries, n=None):
    log("Computing TotalSegmentator Expert A mask density ...")
    if n is not None:
        entries = entries[:n]
    per_case = []
    for entry in entries:
        lbl_path = TS_ROOT / entry["label"]
        arr = nib.load(lbl_path).get_fdata(dtype=np.float32).astype(np.int32)
        n_vox = arr.size
        # Count each organ (MAISI IDs in CHANNEL_MAP keys)
        ch_fg = {}
        for maisi_id, ch in CHANNEL_MAP.items():
            cnt = int((arr == maisi_id).sum())
            ch_fg[CH_NAMES[ch]] = cnt / n_vox
        total_fg = sum(ch_fg.values())
        u, _ = np.unique(arr, return_counts=True)
        per_case.append({
            "case": Path(entry["label"]).stem.replace(".nii", ""),
            "shape": list(arr.shape),
            "fg_ratio": total_fg,
            "ch_fg": ch_fg,
            "n_fg_classes": int(sum(1 for v in ch_fg.values() if v > 0)),
        })
    return per_case


# ─────────────────────── Section 3: L1 loss probe ────────────────────────────

def to_one_hot(lbl_t, channel_map, device):
    B, spatial = lbl_t.shape[0], lbl_t.shape[2:]
    out = torch.zeros((B, 4) + spatial, dtype=torch.float16, device=device)
    for maisi_id, ch in channel_map.items():
        out[:, ch] = (lbl_t[:, 0] == maisi_id).to(torch.float16)
    return out


def load_latent(path, device):
    arr = nib.load(path).get_fdata(dtype=np.float32)
    # latents stored channel-last (D,H,W,4) → permute to (1,4,D,H,W)
    t = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)
    return t.to(device, dtype=torch.float16)


def load_label(path, device):
    arr = nib.load(path).get_fdata(dtype=np.float32).astype(np.int16)
    return torch.from_numpy(arr[None, None]).to(device)  # (1,1,D,H,W)


@torch.no_grad()
def compute_l1_loss_probe(entries, device, scale_factor, unet, controlnet,
                           noise_scheduler):
    log(f"Running L1 loss probe on {len(entries)} cases ...")
    results = []
    for i, entry in enumerate(entries):
        try:
            latent = load_latent(TS_ROOT / entry["image"], device)
            label  = load_label(TS_ROOT / entry["label"], device)
        except Exception as e:
            log(f"  [skip] {entry['image']}: {e}")
            continue

        images_scaled = latent * scale_factor
        noise = torch.randn_like(images_scaled, dtype=torch.float32).to(torch.float16)
        # Use a fixed timestep (t=0.5) for reproducibility
        timesteps = torch.tensor([0.5], device=device)
        noisy_latent = noise_scheduler.add_noise(
            images_scaled.float(), noise.float(), timesteps
        ).to(torch.float16)
        target = (images_scaled - noise).float()

        mask_real = to_one_hot(label, CHANNEL_MAP, device)
        mask_null = torch.zeros_like(mask_real)

        sp_val = entry.get("spacing", [1.0, 1.0, 1.0])
        if isinstance(sp_val, (int, float)):
            sp_val = [float(sp_val)] * 3
        sp_t   = torch.tensor([sp_val], device=device, dtype=torch.float32) * 100.0
        mod_t  = torch.ones((1,), device=device, dtype=torch.long)

        def forward(mask):
            with autocast("cuda"):
                down, mid = controlnet(x=noisy_latent, timesteps=timesteps,
                                       controlnet_cond=mask, class_labels=mod_t)
                out = unet(x=noisy_latent, timesteps=timesteps, spacing_tensor=sp_t,
                           down_block_additional_residuals=down,
                           mid_block_additional_residual=mid, class_labels=mod_t)
            return F.l1_loss(out.float(), target).item()

        l1_real = forward(mask_real)
        l1_null = forward(mask_null)

        fg_ratio = float(mask_real.float().mean())
        ch_fg = {CH_NAMES[ch]: float((mask_real[0, ch] > 0).float().mean())
                 for ch in range(4)}
        n_active = int((mask_real.sum(dim=(0, 2, 3, 4)) > 0).sum())

        results.append({
            "case": Path(entry["label"]).stem.replace(".nii", ""),
            "fg_ratio": fg_ratio,
            "n_active_ch": n_active,
            "ch_fg": ch_fg,
            "l1_real": l1_real,
            "l1_null": l1_null,
            "l1_diff": l1_null - l1_real,   # positive = real mask HELPS
        })
        log(f"  [{i+1}/{len(entries)}] {results[-1]['case']}: "
            f"fg={fg_ratio:.4f}, l1_real={l1_real:.4f}, l1_null={l1_null:.4f}, "
            f"diff={results[-1]['l1_diff']:+.4f}")

    return results


# ─────────────────────────────── SVG charts ──────────────────────────────────

def svg_bar(data_dict, title, xlabel, ylabel, color="#4c8abe", width=600, height=300):
    """Simple horizontal bar chart as inline SVG."""
    items = sorted(data_dict.items(), key=lambda x: -x[1])[:30]
    if not items:
        return "<p>No data</p>"
    max_val = max(v for _, v in items)
    bar_h = max(10, int((height - 60) / len(items)))
    height = bar_h * len(items) + 60
    label_w = 160
    bar_area = width - label_w - 20
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="font-family:monospace;font-size:11px">',
        f'<text x="{width//2}" y="20" text-anchor="middle" font-weight="bold">{title}</text>',
    ]
    for i, (lbl, val) in enumerate(items):
        y = 30 + i * bar_h
        bar_w = int(val / max_val * bar_area)
        lines.append(
            f'<text x="{label_w - 4}" y="{y + bar_h//2 + 4}" text-anchor="end" '
            f'fill="#333">{str(lbl)[:22]}</text>'
        )
        lines.append(
            f'<rect x="{label_w}" y="{y + 2}" width="{bar_w}" height="{bar_h - 4}" '
            f'fill="{color}" opacity="0.8"/>'
        )
        lines.append(
            f'<text x="{label_w + bar_w + 4}" y="{y + bar_h//2 + 4}" '
            f'fill="#333">{val:.4%}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines)


def svg_scatter(xs, ys, labels, title, xlabel, ylabel,
                color_real="#2c7bb6", color_null="#d73027",
                width=700, height=420):
    """Scatter plot comparing real vs null L1 loss."""
    if not xs:
        return "<p>No data</p>"
    margin = {"l": 70, "r": 30, "t": 50, "b": 60}
    pw = width - margin["l"] - margin["r"]
    ph = height - margin["t"] - margin["b"]

    xmin, xmax = 0.0, max(xs) * 1.1 + 1e-9
    yall = ys["real"] + ys["null"]
    ymin, ymax = min(yall) * 0.98, max(yall) * 1.02

    def tx(x): return margin["l"] + (x - xmin) / (xmax - xmin) * pw
    def ty(y): return margin["t"] + ph - (y - ymin) / (ymax - ymin) * ph

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="font-family:monospace;font-size:11px">',
        f'<text x="{width//2}" y="22" text-anchor="middle" font-size="13" '
        f'font-weight="bold">{title}</text>',
        # axes
        f'<line x1="{margin["l"]}" y1="{margin["t"]}" x2="{margin["l"]}" '
        f'y2="{margin["t"] + ph}" stroke="#aaa"/>',
        f'<line x1="{margin["l"]}" y1="{margin["t"] + ph}" '
        f'x2="{margin["l"] + pw}" y2="{margin["t"] + ph}" stroke="#aaa"/>',
        f'<text x="{width//2}" y="{height - 8}" text-anchor="middle">{xlabel}</text>',
        f'<text x="14" y="{height//2}" text-anchor="middle" '
        f'transform="rotate(-90,14,{height//2})">{ylabel}</text>',
    ]
    # gridlines + y ticks
    for yt in np.linspace(ymin, ymax, 5):
        y_px = ty(yt)
        lines.append(f'<line x1="{margin["l"]}" y1="{y_px:.1f}" '
                     f'x2="{margin["l"]+pw}" y2="{y_px:.1f}" stroke="#eee"/>')
        lines.append(f'<text x="{margin["l"]-4}" y="{y_px+4:.1f}" '
                     f'text-anchor="end">{yt:.3f}</text>')
    # x ticks
    for xt in np.linspace(xmin, xmax, 6):
        x_px = tx(xt)
        lines.append(f'<text x="{x_px:.1f}" y="{margin["t"]+ph+16}" '
                     f'text-anchor="middle">{xt:.3%}</text>')

    # points
    for xi, (yr, yn, lbl) in enumerate(zip(ys["real"], ys["null"], labels)):
        x_px, yr_px, yn_px = tx(xs[xi]), ty(yr), ty(yn)
        lines.append(f'<circle cx="{x_px:.1f}" cy="{yr_px:.1f}" r="5" '
                     f'fill="{color_real}" opacity="0.75"><title>{lbl}\nfg={xs[xi]:.4%}\nl1_real={yr:.4f}</title></circle>')
        lines.append(f'<circle cx="{x_px:.1f}" cy="{yn_px:.1f}" r="5" '
                     f'fill="{color_null}" opacity="0.75"><title>{lbl}\nfg={xs[xi]:.4%}\nl1_null={yn:.4f}</title></circle>')
        # line connecting real to null
        lines.append(f'<line x1="{x_px:.1f}" y1="{yr_px:.1f}" '
                     f'x2="{x_px:.1f}" y2="{yn_px:.1f}" stroke="#999" stroke-width="1" opacity="0.4"/>')

    # legend
    ly = margin["t"] + 10
    lines.append(f'<circle cx="{margin["l"]+pw-120}" cy="{ly}" r="5" fill="{color_real}"/>')
    lines.append(f'<text x="{margin["l"]+pw-112}" y="{ly+4}">real mask</text>')
    ly += 18
    lines.append(f'<circle cx="{margin["l"]+pw-120}" cy="{ly}" r="5" fill="{color_null}"/>')
    lines.append(f'<text x="{margin["l"]+pw-112}" y="{ly+4}">null mask</text>')

    lines.append("</svg>")
    return "\n".join(lines)


def svg_histogram(values, title, xlabel, color="#4c8abe", bins=12, width=500, height=280):
    if not values:
        return "<p>No data</p>"
    arr = np.array(values)
    counts, edges = np.histogram(arr, bins=bins)
    margin = {"l": 50, "r": 20, "t": 40, "b": 50}
    pw = width - margin["l"] - margin["r"]
    ph = height - margin["t"] - margin["b"]
    bar_w = pw / bins
    max_c = max(counts)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="font-family:monospace;font-size:11px">',
        f'<text x="{width//2}" y="20" text-anchor="middle" font-weight="bold">{title}</text>',
        f'<line x1="{margin["l"]}" y1="{margin["t"]}" x2="{margin["l"]}" '
        f'y2="{margin["t"]+ph}" stroke="#aaa"/>',
        f'<line x1="{margin["l"]}" y1="{margin["t"]+ph}" '
        f'x2="{margin["l"]+pw}" y2="{margin["t"]+ph}" stroke="#aaa"/>',
        f'<text x="{width//2}" y="{height-6}" text-anchor="middle">{xlabel}</text>',
    ]
    for i, (cnt, lo, hi) in enumerate(zip(counts, edges[:-1], edges[1:])):
        bh = int(cnt / max_c * ph)
        x = margin["l"] + i * bar_w
        y = margin["t"] + ph - bh
        lines.append(f'<rect x="{x:.1f}" y="{y}" width="{bar_w - 1:.1f}" height="{bh}" '
                     f'fill="{color}" opacity="0.8"/>')
        if i % 2 == 0:
            lines.append(f'<text x="{x + bar_w/2:.1f}" y="{margin["t"]+ph+14}" '
                         f'text-anchor="middle">{lo:.2%}</text>')
    lines.append("</svg>")
    return "\n".join(lines)


# ─────────────────────────────── HTML builder ────────────────────────────────

def build_html(kits_stats, ts_per_case, loss_results, ep53_note):
    # ── KiTS density ──
    kits_fg_ratios = [c["fg_ratio"] for c in kits_stats["comb"]["per_case"]]
    kits_n_classes = [c["n_fg_classes"] for c in kits_stats["comb"]["per_case"]]

    hist_kits_fg    = svg_histogram(kits_fg_ratios, "KiTS (MAISI combined): fg_ratio per case",
                                    "fg_ratio", color="#e07b39")
    hist_kits_nclass = svg_histogram(kits_n_classes, "KiTS: #FG classes per case",
                                     "#classes", color="#e07b39", bins=8)

    top_kits_labels = dict(sorted(kits_stats["comb"]["per_label"].items(),
                                  key=lambda x: -x[1])[:30])
    bar_kits = svg_bar({f"label {k} ({MAISI_LABELS.get(k,'?')})": v
                        for k, v in top_kits_labels.items()},
                       "KiTS MAISI combined: top-30 labels by volume",
                       "label", "% of total volume", color="#e07b39")

    # ── TS density ──
    ts_fg_ratios = [c["fg_ratio"] for c in ts_per_case]
    ts_n_classes = [c["n_fg_classes"] for c in ts_per_case]
    hist_ts_fg   = svg_histogram(ts_fg_ratios, "TotalSeg Expert A: fg_ratio per case",
                                 "fg_ratio", color="#4c8abe")
    hist_ts_nclass = svg_histogram(ts_n_classes, "TotalSeg: #active channels per case",
                                   "#channels", color="#4c8abe", bins=5)

    # ── L1 loss scatter ──
    xs     = [r["fg_ratio"] for r in loss_results]
    y_real = [r["l1_real"]  for r in loss_results]
    y_null = [r["l1_null"]  for r in loss_results]
    labels = [r["case"]     for r in loss_results]
    scatter = svg_scatter(xs, {"real": y_real, "null": y_null}, labels,
                          "ControlNet L1 loss: real mask vs null mask vs fg_ratio",
                          "mask fg_ratio", "L1 loss")

    # ── L1 diff histogram ──
    diffs = [r["l1_diff"] for r in loss_results]
    hist_diff = svg_histogram(diffs, "l1_null - l1_real (positive = mask helps)",
                              "Δ L1", color="#74c476", bins=10)

    # ── Summary table ──
    mean_real = np.mean(y_real) if y_real else 0
    mean_null = np.mean(y_null) if y_null else 0
    mean_diff = np.mean(diffs)  if diffs  else 0
    corr_label = "n/a"
    if len(xs) > 3:
        corr = float(np.corrcoef(xs, diffs)[0, 1])
        corr_label = f"{corr:+.3f}"

    # ── MAISI label table ──
    label_rows = "\n".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in sorted(MAISI_LABELS.items())
    )

    # ── Loss results table ──
    loss_rows = "\n".join(
        f"<tr><td>{r['case']}</td><td>{r['fg_ratio']:.4%}</td>"
        f"<td>{r['n_active_ch']}</td>"
        f"<td>{r['l1_real']:.4f}</td><td>{r['l1_null']:.4f}</td>"
        f"<td style='color:{'green' if r['l1_diff']>0.001 else 'red'}'>"
        f"{r['l1_diff']:+.4f}</td></tr>"
        for r in loss_results
    )

    kits_orig_bg  = kits_stats["orig"]["bg_ratio"]
    kits_comb_bg  = kits_stats["comb"]["bg_ratio"]
    kits_n_comb   = len(kits_stats["comb"]["per_case"])
    kits_n_orig   = len(kits_stats["orig"]["per_case"])

    ts_mean_fg = np.mean(ts_fg_ratios) if ts_fg_ratios else 0
    ts_max_fg  = max(ts_fg_ratios)     if ts_fg_ratios else 0
    ts_min_fg  = min(ts_fg_ratios)     if ts_fg_ratios else 0

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8"/>
<title>Mask Density & ControlNet L1 Loss Analysis</title>
<style>
  body {{ font-family: "Segoe UI", sans-serif; background: #f5f5f5; padding: 20px; }}
  h1 {{ color: #222; border-bottom: 3px solid #4c8abe; padding-bottom: 8px; }}
  h2 {{ color: #4c8abe; margin-top: 30px; }}
  h3 {{ color: #666; }}
  .card {{ background: white; border-radius: 8px; padding: 20px; margin: 16px 0;
           box-shadow: 0 2px 6px rgba(0,0,0,0.1); }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                 gap: 12px; }}
  .stat-box {{ background: #f0f6ff; border: 1px solid #c0d8f0; border-radius: 6px;
               padding: 12px; text-align: center; }}
  .stat-val {{ font-size: 2em; font-weight: bold; color: #2c7bb6; }}
  .stat-label {{ font-size: 0.85em; color: #555; margin-top: 4px; }}
  .warn {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px;
           padding: 10px; margin: 8px 0; }}
  .good {{ background: #d4edda; border: 1px solid #28a745; border-radius: 6px;
           padding: 10px; margin: 8px 0; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.88em; }}
  th {{ background: #4c8abe; color: white; padding: 6px 10px; }}
  td {{ padding: 4px 10px; border-bottom: 1px solid #ddd; }}
  tr:nth-child(even) {{ background: #f9f9f9; }}
  .charts {{ display: flex; flex-wrap: wrap; gap: 16px; }}
  .note {{ color: #666; font-size: 0.9em; font-style: italic; }}
  .finding {{ background: #e8f4f8; border-left: 4px solid #4c8abe;
              padding: 10px 14px; margin: 8px 0; border-radius: 0 6px 6px 0; }}
</style>
</head>
<body>
<h1>Mask Density & ControlNet L1 Loss — Dataset Comparison</h1>
<p class="note">Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} | Expert A checkpoint: ep53 | {N_LOSS_CASES} TS cases probed</p>

<div class="card">
<h2>核心问题：不同数据集类别数不同，MAISI v2 如何统一训练？</h2>
<div class="finding">
<strong>MAISI v2 原始预训练方案（binary encoding）：</strong>
将 integer label_id 按 binarize_labels 分解为 8-bit binary pattern（固定 8 channels），
任何 label_id（1~132）都映射到相同格式。不同数据集只是有/没有某些 label_id，
条件格式完全一致，可以 mixed-batch 训练。
</div>
<div class="finding">
<strong>我们的 Expert A 方案（4-ch one-hot）：</strong>
每 channel 对应 1 个器官，缺失 = 全 0。4 channels 固定对应
{" / ".join([f"ch{i}={n}" for i, n in enumerate(CH_NAMES)])}。
不同数据集只要能提供这 4 个器官的 mask 即可，缺的 channel 置零。
</div>
</div>

<!-- ═══════════════════════════ Section 1: KiTS ═══════════════════════════ -->
<div class="card">
<h2>§1. KiTS Demo Dataset（MAISI C4KC-KiTS_subset，30 cases）</h2>
<h3>1a. 原始 KiTS 标注（mask_zs — 仅肾脏+肿瘤）</h3>
<div class="stats-grid">
  <div class="stat-box"><div class="stat-val">2</div><div class="stat-label">#FG classes (kidney + tumor)</div></div>
  <div class="stat-box"><div class="stat-val">{kits_orig_bg:.1%}</div><div class="stat-label">Background ratio</div></div>
  <div class="stat-box"><div class="stat-val">{kits_n_orig}</div><div class="stat-label">Cases</div></div>
  <div class="stat-box"><div class="stat-val">label 1 & 2</div><div class="stat-label">kidney / tumor</div></div>
</div>
<div class="warn">⚠ 原始 KiTS mask 只有 2 类，fg_ratio 仅 0.2~0.8%，极度稀疏。这是原始任务标注，不是 MAISI 输入的 mask。</div>

<h3>1b. MAISI combined_label（mask_combined_label_wbdm — MAISI 自动扩充标注）</h3>
<div class="stats-grid">
  <div class="stat-box"><div class="stat-val">93</div><div class="stat-label">#FG classes (aggregate)</div></div>
  <div class="stat-box"><div class="stat-val">{kits_comb_bg:.1%}</div><div class="stat-label">Background ratio</div></div>
  <div class="stat-box"><div class="stat-val">{kits_n_comb}</div><div class="stat-label">Cases</div></div>
  <div class="stat-box"><div class="stat-val">41~90</div><div class="stat-label">FG classes per case</div></div>
  <div class="stat-box"><div class="stat-val">1~132+200</div><div class="stat-label">Label ID range</div></div>
  <div class="stat-box"><div class="stat-val">15.9%</div><div class="stat-label">label 200 (body region) share</div></div>
</div>
<div class="good">✅ MAISI 训练时实际使用的是 combined_label：自动扩充了 93 个器官类别，fg_ratio 6~42%，远比原始 KiTS 标注密集。</div>

<div class="charts">
  {hist_kits_fg}
  {hist_kits_nclass}
</div>
{bar_kits}

<h3>1c. KiTS Per-case 统计</h3>
<table>
<tr><th>Case</th><th>Shape</th><th>#FG classes</th><th>fg_ratio</th><th>Label IDs</th></tr>
{"".join(
    f"<tr><td>{c['case']}</td><td>{c['shape']}</td>"
    f"<td>{c['n_fg_classes']}</td><td>{c['fg_ratio']:.2%}</td>"
    f"<td style='font-size:0.78em'>{str(c['fg_ids'][:10])}{'...' if len(c['fg_ids'])>10 else ''}</td></tr>"
    for c in kits_stats["comb"]["per_case"]
)}
</table>
</div>

<!-- ═══════════════════════════ Section 2: TotalSeg ═══════════════════════════ -->
<div class="card">
<h2>§2. TotalSegmentator Expert A（iso10 384³，4-ch one-hot）</h2>
<div class="stats-grid">
  <div class="stat-box"><div class="stat-val">4</div><div class="stat-label">#channels (liver/spleen/kdnR/kdnL)</div></div>
  <div class="stat-box"><div class="stat-val">{ts_mean_fg:.2%}</div><div class="stat-label">Mean fg_ratio</div></div>
  <div class="stat-box"><div class="stat-val">{ts_min_fg:.2%}</div><div class="stat-label">Min fg_ratio</div></div>
  <div class="stat-box"><div class="stat-val">{ts_max_fg:.2%}</div><div class="stat-label">Max fg_ratio</div></div>
</div>
<div class="warn">⚠ TS Expert A mask 仅有 4 个器官，平均 fg_ratio {ts_mean_fg:.2%}（极度稀疏）。这是 conditioning failure 的主要原因之一。</div>

<div class="charts">
  {hist_ts_fg}
  {hist_ts_nclass}
</div>

<h3>Per-case 统计（前30）</h3>
<table>
<tr><th>Case</th><th>fg_ratio</th><th>active_ch</th><th>liver</th><th>spleen</th><th>kidney_R</th><th>kidney_L</th></tr>
{"".join(
    f"<tr><td>{c['case']}</td><td>{c['fg_ratio']:.4%}</td><td>{c['n_fg_classes']}</td>"
    + "".join(f"<td>{c['ch_fg'].get(n, 0):.4%}</td>" for n in CH_NAMES)
    + "</tr>"
    for c in ts_per_case[:30]
)}
</table>
</div>

<!-- ═══════════════════════════ Section 3: L1 Loss ═══════════════════════════ -->
<div class="card">
<h2>§3. ControlNet L1 Loss vs Mask Density（Expert A ep53）</h2>
<p>{ep53_note}</p>
<div class="stats-grid">
  <div class="stat-box"><div class="stat-val">{mean_real:.4f}</div><div class="stat-label">Mean L1 (real mask)</div></div>
  <div class="stat-box"><div class="stat-val">{mean_null:.4f}</div><div class="stat-label">Mean L1 (null mask)</div></div>
  <div class="stat-box"><div class="stat-val" style="color:{'green' if mean_diff>0.001 else 'red'}">{mean_diff:+.4f}</div><div class="stat-label">Mean Δ L1 (null−real)</div></div>
  <div class="stat-box"><div class="stat-val">{corr_label}</div><div class="stat-label">Corr(fg_ratio, Δ L1)</div></div>
</div>

{"<div class='good'>✅ 实验结果：real mask L1 < null mask L1，条件信息有效压低了 loss。</div>" if mean_diff > 0.005 else
 "<div class='warn'>⚠ 实验结果：real mask 与 null mask L1 差距极小，ControlNet 对 mask 内容不敏感——conditioning 失效。</div>"}

{scatter}

<div class="charts">
  {hist_diff}
</div>

<div class="finding">
<strong>分析：</strong> 如果 conditioning 有效，预期在 fg_ratio 较高的 case 中，Δ L1 应更大（mask 信息更丰富，帮助越多）。
如果 Corr(fg_ratio, ΔL1) ≈ 0 且 ΔL1 ≈ 0，说明 ControlNet 添加的是 mask-independent 残差，即 unconditional bias。
</div>

<h3>详细结果表</h3>
<table>
<tr><th>Case</th><th>fg_ratio</th><th>active_ch</th><th>L1 real</th><th>L1 null</th><th>Δ (null−real)</th></tr>
{loss_rows}
</table>
</div>

<!-- ═══════════════════════════ Section 4: MAISI Label Table ══════════════════ -->
<div class="card">
<h2>§4. MAISI v2 Label Vocabulary（部分）</h2>
<p>MAISI v2 完整词表包含 <strong>132 个解剖结构</strong>（label_id 1~132，含 gaps），
加上特殊 <strong>label 200</strong>（body region / "other"）。
C4KC-KiTS 数据集中出现 <strong>93 个 FG label</strong>，最大 label_id=200。</p>
<table>
<tr><th>Label ID</th><th>Organ / Structure</th></tr>
{label_rows}
</table>
</div>

</body>
</html>"""
    return html


# ─────────────────────────────────── main ────────────────────────────────────

def main():
    random.seed(1234)

    # ── Section 1: KiTS density ──
    kits_stats = compute_kits_density()

    # ── Section 2: TS density (sample 50 cases) ──
    with open(TS_ROOT / "expert_A_853cases.json") as f:
        all_entries = json.load(f)["training"]
    # Sample spread by fg_ratio after loading a few; just use sequential first 50
    sample_entries_density = all_entries[:50]
    ts_per_case = compute_ts_density(sample_entries_density)

    # ── Sections 3: L1 loss probe ──
    log("Loading models for L1 probe ...")
    device = torch.device("cuda:0")

    cfg = load_config(str(ENV_CFG), str(MODEL_CFG), str(NET_CFG))

    unet = define_instance(cfg, "diffusion_unet_def").to(device)
    sd   = torch.load(cfg.trained_diffusion_path, map_location=device, weights_only=False)
    unet.load_state_dict(sd["unet_state_dict"], strict=False)
    scale_factor = float(sd["scale_factor"])
    unet.eval()
    for p in unet.parameters(): p.requires_grad_(False)
    log(f"UNet loaded; scale_factor={scale_factor:.4f}")

    controlnet = define_instance(cfg, "controlnet_def").to(device)
    copy_model_state(controlnet, unet.state_dict())
    ck = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    controlnet.load_state_dict(ck["controlnet_state_dict"], strict=False)
    ep53_note = f"ControlNet ep{ck.get('epoch', '?')} checkpoint loaded from {CKPT_PATH}"
    controlnet.eval()
    for p in controlnet.parameters(): p.requires_grad_(False)
    log(ep53_note)

    with open(NET_CFG) as f:
        net_cfg_raw = json.load(f)
    ns_cfg = net_cfg_raw.get("noise_scheduler", {})
    noise_scheduler = RFlowScheduler(
        num_train_timesteps=ns_cfg.get("num_train_timesteps", 1000),
        use_discrete_timesteps=ns_cfg.get("use_discrete_timesteps", False),
        use_timestep_transform=ns_cfg.get("use_timestep_transform", True),
        sample_method=ns_cfg.get("sample_method", "uniform"),
        scale=ns_cfg.get("scale", 1.4),
    )

    # Pick N_LOSS_CASES spread across all 853 entries
    step = max(1, len(all_entries) // N_LOSS_CASES)
    probe_entries = all_entries[::step][:N_LOSS_CASES]
    loss_results = compute_l1_loss_probe(probe_entries, device, scale_factor,
                                         unet, controlnet, noise_scheduler)

    # ── Build HTML ──
    log("Building HTML report ...")
    html = build_html(kits_stats, ts_per_case, loss_results, ep53_note)
    OUT_HTML.parent.mkdir(exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    log(f"Report written: {OUT_HTML}  ({OUT_HTML.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
