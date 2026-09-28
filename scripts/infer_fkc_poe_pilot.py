#!/usr/bin/env python3
"""FKC-PoE pilot inference: Expert A (TS abdominal) + Expert B (MMWHS cardiac).

Runs 4 generation modes per test case:
  A_only     — single Expert A ControlNet, no composition
  B_only     — single Expert B ControlNet, no composition
  poe_naive  — CFG-PoE, no FKC weights
  poe_fkc    — CFG-PoE + FKC weights (beta=1, K=4)

Expert A: amos22_expert_A_r5   (8-ch binarize_labels, trained on AMOS22 abdominal).
Expert B: mmwhs_expert_B_v3    (8-ch binarize_labels, trained on MMWHS cardiac).

Each expert independently conditions on its own mask type.  Spatial coherence
is approximate in this pilot (no composite volumes): Expert A's mask covers
abdominal anatomy while Expert B's mask covers cardiac anatomy; the FKC weights
measure velocity agreement to produce a single coherent trajectory.

Usage:
  python -u scripts/infer_fkc_poe_pilot.py --gpu 1
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import nibabel as nib
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAISI_ROOT = PROJECT_ROOT / "maisi"
sys.path.insert(0, str(MAISI_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(MAISI_ROOT)

from scripts.utils import define_instance, binarize_labels
from scripts.diff_model_setting import load_config
from fkc_poe_sampler import ControlNetExpert, SamplerConfig, sample_fkc_poe_n, snis_weighted_average

# ---------------------------------------------------------------------------
# Label IDs for foreground visualization only (not used for conditioning)
# ---------------------------------------------------------------------------

AMOS22_IDS = list(range(1, 16))   # IDs 1-15
MMWHS_IDS  = list(range(101, 108))  # IDs 101-107

AMOS22_LABEL_NAMES = {
    1: "spleen", 2: "right_kidney", 3: "left_kidney", 4: "gall_bladder",
    5: "esophagus", 6: "liver", 7: "stomach", 8: "aorta",
    9: "IVC", 10: "pancreas", 11: "R_adrenal", 12: "L_adrenal",
    13: "duodenum", 14: "bladder", 15: "prostate_uterus"
}
MMWHS_LABEL_NAMES = {
    101: "LV_myo", 102: "L_atrium", 103: "L_ventricle",
    104: "aorta_asc", 105: "R_ventricle", 106: "pulm_artery", 107: "R_atrium"
}

# binarize_labels always outputs 8 channels regardless of dataset
N_CH_A = 8
N_CH_B = 8

# Checkpoints (correct protocol: binarize_labels 8-bit + pretrained ControlNet)
CKPT_A = PROJECT_ROOT / "checkpoints/amos22_expert_A_r5/checkpoint_latest.pt"
CKPT_B = PROJECT_ROOT / "checkpoints/mmwhs_expert_B_v3/checkpoint_latest.pt"

# Data
AMOS22_LABEL_DIR = PROJECT_ROOT / "data/phase2_amos22_iso15/labels"
MMWHS_LABEL_DIR  = PROJECT_ROOT / "data/phase2_mmwhs_iso15/labels"

SPACING_ISO15 = [1.5, 1.5, 1.5]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_label(path, device):
    arr = nib.load(str(path)).get_fdata().astype(np.int64)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def build_models(device: torch.device):
    """Load shared UNet, AutoEncoder, and noise scheduler."""
    cfg = load_config(
        str(MAISI_ROOT / "configs/environment_rflow-ct_local.json"),
        str(MAISI_ROOT / "configs/config_maisi_diff_model_rflow-ct.json"),
        str(MAISI_ROOT / "configs/config_network_rflow.json"),
    )

    log("Loading UNet ...")
    unet = define_instance(cfg, "diffusion_unet_def").to(device)
    sd = torch.load(cfg.trained_diffusion_path, map_location=device, weights_only=False)
    unet.load_state_dict(sd["unet_state_dict"], strict=False)
    scale_factor = float(sd["scale_factor"].item() if hasattr(sd["scale_factor"], "item") else sd["scale_factor"])
    unet.eval()
    for p in unet.parameters():
        p.requires_grad_(False)
    log(f"UNet ready; scale_factor={scale_factor:.4f}")

    log("Loading AutoEncoder (CPU) ...")
    ae = define_instance(cfg, "autoencoder_def")  # keep on CPU — decode is 8 GB at 256^3
    ae_sd = torch.load(cfg.trained_autoencoder_path, map_location="cpu", weights_only=False)
    if isinstance(ae_sd, dict) and "unet_state_dict" in ae_sd:
        ae_sd = ae_sd["unet_state_dict"]
    ae.load_state_dict(ae_sd)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    # norm_float16=True would cast to fp16 inside autocast; on CPU without autocast this
    # mismatches fp32 Conv biases → NaN activations. Disable globally.
    for m in ae.modules():
        if hasattr(m, "norm_float16"):
            m.norm_float16 = False
    log("AutoEncoder ready (CPU, norm_float16=False)")

    noise_scheduler = define_instance(cfg, "noise_scheduler")
    return unet, ae, noise_scheduler, scale_factor


def load_controlnet(network_cfg_path: str, ckpt_path: Path, unet, device: torch.device):
    """Build ControlNet from dedicated network config + load checkpoint."""
    from monai.networks.utils import copy_model_state
    cfg = load_config(
        str(MAISI_ROOT / "configs/environment_rflow-ct_local.json"),
        str(MAISI_ROOT / "configs/config_maisi_diff_model_rflow-ct.json"),
        network_cfg_path,
    )
    cn = define_instance(cfg, "controlnet_def").to(device)
    copy_model_state(cn, unet.state_dict())
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        cn.load_state_dict(ck["controlnet_state_dict"], strict=True)
        log(f"  loaded {ckpt_path.name}")
    else:
        log(f"  WARNING: {ckpt_path} not found — using copy_model_state weights (untrained)")
    cn.eval()
    for p in cn.parameters():
        p.requires_grad_(False)
    return cn


def decode_latent(ae, latents: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Decode MAISI latent on CPU to avoid GPU OOM at 256^3 scale."""
    with torch.no_grad():
        ct = ae.decode_stage_2_outputs(latents.cpu() / scale_factor)
    return ct


def _ct_disp(sl, vmin=-1.0, vmax=1.0):
    return np.clip(sl, vmin, vmax)


def _overlay_contours(ax, la_sl, lb_sl):
    if la_sl is not None and la_sl.any():
        ax.contour(la_sl > 0, levels=[0.5], colors=["red"], linewidths=1.0, alpha=0.85)
    if lb_sl is not None and lb_sl.any():
        ax.contour(lb_sl > 0, levels=[0.5], colors=["cyan"], linewidths=1.0, alpha=0.85)


def _organ_centroid(mask):
    """Return (cx, cy, cz) centroid of non-zero voxels, or volume center if empty."""
    if mask is None or not mask.any():
        s = mask.shape if mask is not None else (256, 256, 256)
        return s[0] // 2, s[1] // 2, s[2] // 2
    pts = np.argwhere(mask > 0)
    return tuple(pts.mean(axis=0).astype(int))


def _buf_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def render_all_modes_plane_montage(cts: dict, title: str, axis: str = "axial",
                                    lbl_a=None, lbl_b=None, n_slices=12) -> str:
    """All modes in a rows=modes × cols=slices grid for the given axis."""
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        modes = list(cts.keys())
        ct0 = next(iter(cts.values()))
        X, Y, Z = ct0.shape
        if axis == "axial":
            fg_ax = (0, 1); dim_size = Z
            def get_sl(ct, i): return ct[:, :, i].T
            label_char = "z"
        elif axis == "coronal":
            fg_ax = (0, 2); dim_size = Y
            def get_sl(ct, i): return ct[:, i, :].T
            label_char = "y"
        else:
            fg_ax = (1, 2); dim_size = X
            def get_sl(ct, i): return ct[i, :, :].T
            label_char = "x"

        fg = np.zeros(ct0.shape, dtype=bool)
        if lbl_a is not None: fg |= lbl_a > 0
        if lbl_b is not None: fg |= lbl_b > 0
        if fg.any():
            pos = np.where(fg.any(axis=fg_ax))[0]
            lo, hi = int(pos.min()), int(pos.max())
        else:
            lo, hi = dim_size // 4, 3 * dim_size // 4
        indices = np.linspace(lo, hi, n_slices, dtype=int)

        fig, axes = plt.subplots(len(modes), n_slices,
                                  figsize=(3.2 * n_slices, 4.5 * len(modes)))
        fig.suptitle(title, fontsize=13)
        for row, mode in enumerate(modes):
            ct_np = cts[mode]
            for col, idx in enumerate(indices):
                ax = axes[row, col]
                ax.imshow(_ct_disp(get_sl(ct_np, idx)), cmap="gray",
                          vmin=-1, vmax=1, aspect="auto", origin="lower")
                _overlay_contours(ax,
                                  get_sl(lbl_a, idx) if lbl_a is not None else None,
                                  get_sl(lbl_b, idx) if lbl_b is not None else None)
                if col == 0:
                    ax.set_ylabel(mode, fontsize=9, rotation=0, ha="right", labelpad=55)
                if row == 0:
                    ax.set_title(f"{label_char}={idx}", fontsize=8)
                ax.axis("off")
        plt.tight_layout()
        return _buf_to_b64(fig)
    except Exception as e:
        log(f"  [montage {axis} warning] {e}")
        return ""


def render_all_modes_triplane(cts: dict, title: str, anchor,
                               lbl_a=None, lbl_b=None) -> str:
    """All modes in a rows=modes × 3-plane grid at the given (cx,cy,cz) anchor."""
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cx, cy, cz = anchor
        modes = list(cts.keys())
        plane_names = [f"Axial z={cz}", f"Coronal y={cy}", f"Sagittal x={cx}"]
        fig, axes = plt.subplots(len(modes), 3, figsize=(14, 4.5 * len(modes)))
        fig.suptitle(title, fontsize=11)
        for row, mode in enumerate(modes):
            ct_np = cts[mode]
            slices = [
                (ct_np[:, :, cz].T,
                 lbl_a[:, :, cz].T if lbl_a is not None else None,
                 lbl_b[:, :, cz].T if lbl_b is not None else None),
                (ct_np[:, cy, :].T,
                 lbl_a[:, cy, :].T if lbl_a is not None else None,
                 lbl_b[:, cy, :].T if lbl_b is not None else None),
                (ct_np[cx, :, :].T,
                 lbl_a[cx, :, :].T if lbl_a is not None else None,
                 lbl_b[cx, :, :].T if lbl_b is not None else None),
            ]
            for col, (sl, la_sl, lb_sl) in enumerate(slices):
                ax = axes[row, col]
                ax.imshow(_ct_disp(sl), cmap="gray", vmin=-1, vmax=1,
                          aspect="auto", origin="lower")
                _overlay_contours(ax, la_sl, lb_sl)
                if col == 0:
                    ax.set_ylabel(mode, fontsize=9, rotation=0, ha="right", labelpad=55)
                if row == 0:
                    ax.set_title(plane_names[col], fontsize=9)
                ax.axis("off")
        plt.tight_layout()
        return _buf_to_b64(fig)
    except Exception as e:
        log(f"  [triplane-all warning] {e}")
        return ""


def render_multipos_comparison(cts: dict, title: str, z_positions: list,
                                lbl_a=None, lbl_b=None) -> str:
    """All modes at multiple z-positions. Rows=modes, columns=z-positions."""
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        modes = list(cts.keys())
        n_pos = len(z_positions)
        fig, axes = plt.subplots(len(modes), n_pos,
                                  figsize=(3.5 * n_pos, 4.5 * len(modes)))
        fig.suptitle(title, fontsize=11)
        for row, mode in enumerate(modes):
            ct_np = cts[mode]
            for col, z in enumerate(z_positions):
                ax = axes[row, col]
                ax.imshow(_ct_disp(ct_np[:, :, z].T), cmap="gray",
                          vmin=-1, vmax=1, aspect="auto", origin="lower")
                _overlay_contours(ax,
                                  lbl_a[:, :, z].T if lbl_a is not None else None,
                                  lbl_b[:, :, z].T if lbl_b is not None else None)
                if col == 0:
                    ax.set_ylabel(mode, fontsize=9, rotation=0, ha="right", labelpad=55)
                if row == 0:
                    ax.set_title(f"z={z}", fontsize=9)
                ax.axis("off")
        plt.tight_layout()
        return _buf_to_b64(fig)
    except Exception as e:
        log(f"  [multipos warning] {e}")
        return ""


def render_diff_map(ct_ref: np.ndarray, ct_other: np.ndarray,
                    title: str, z: int) -> str:
    """Absolute difference |ct_ref - ct_other| at axial slice z."""
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        diff = np.abs(ct_ref[:, :, z] - ct_other[:, :, z]).T
        fig, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.imshow(diff, cmap="hot", vmin=0, vmax=0.5, aspect="auto", origin="lower")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        return _buf_to_b64(fig)
    except Exception as e:
        log(f"  [diff warning] {e}")
        return ""


def render_slices_base64(ct_np: np.ndarray, title: str, lbl_a: np.ndarray = None,
                         lbl_b: np.ndarray = None) -> str:
    """Thin wrapper kept for mask-preview call only (blank CT + label contours)."""
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        X, Y, Z = ct_np.shape
        cx, cy, cz = _organ_centroid(lbl_a if (lbl_a is not None and lbl_a.any()) else lbl_b)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title, fontsize=11)
        for ax, (vname, sl, la_sl, lb_sl) in zip(axes, [
            (f"axial z={cz}",    ct_np[:, :, cz].T,
             lbl_a[:, :, cz].T if lbl_a is not None else None,
             lbl_b[:, :, cz].T if lbl_b is not None else None),
            (f"coronal y={cy}",  ct_np[:, cy, :].T,
             lbl_a[:, cy, :].T if lbl_a is not None else None,
             lbl_b[:, cy, :].T if lbl_b is not None else None),
            (f"sagittal x={cx}", ct_np[cx, :, :].T,
             lbl_a[cx, :, :].T if lbl_a is not None else None,
             lbl_b[cx, :, :].T if lbl_b is not None else None),
        ]):
            ax.imshow(_ct_disp(sl), cmap="gray", vmin=-1, vmax=1,
                      aspect="auto", origin="lower")
            _overlay_contours(ax, la_sl, lb_sl)
            ax.set_title(vname, fontsize=9)
            ax.axis("off")
        return _buf_to_b64(fig)
    except Exception as e:
        log(f"  [render warning] {e}")
        return ""


def render_ess_curve_base64(ess_curves: dict, logw_ranges: dict) -> str:
    """Render ESS curves for multiple modes, return base64 PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        colors = {"poe_naive": "steelblue", "poe_fkc": "darkorange"}

        for mode, ess in ess_curves.items():
            c = colors.get(mode, "gray")
            ax1.plot(ess, label=mode, color=c, linewidth=1.5)
        ax1.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="K=1 baseline")
        ax1.set_xlabel("Denoising step")
        ax1.set_ylabel("ESS")
        ax1.set_title("Effective Sample Size per step")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        for mode, (lo_curve, hi_curve) in logw_ranges.items():
            c = colors.get(mode, "gray")
            steps = range(len(lo_curve))
            ax2.fill_between(steps, lo_curve, hi_curve, alpha=0.3, color=c, label=f"{mode} log_w range")
            ax2.plot(steps, [(h + l) / 2 for h, l in zip(hi_curve, lo_curve)],
                     color=c, linewidth=1.5)
        ax2.set_xlabel("Denoising step")
        ax2.set_ylabel("log weight")
        ax2.set_title("FK log-weight spread")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception as e:
        log(f"  [ESS render warning] {e}")
        return ""


def render_vip_curve_base64(vip_curves: dict) -> str:
    """Render velocity inner-product curves (FK signal strength), return base64 PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 3))
        colors = {"poe_naive": "steelblue", "poe_fkc": "darkorange"}
        for mode, curve in vip_curves.items():
            c = colors.get(mode, "gray")
            ax.plot(curve, label=mode, color=c, linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.set_xlabel("Denoising step")
        ax.set_ylabel("d log w / step  (FK signal)")
        ax.set_title("Per-step FK weight increment  ⟨v_A−v_un, v_B−v_un⟩/D")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception as e:
        return ""


def run_single_expert(unet, ae, noise_scheduler, scale_factor, controlnet, cond_t,
                      n_channels, K, n_steps, spacing, device, seed=1234):
    """Run standard single-expert generation (degenerate FKC with 1 expert)."""
    dummy_expert = ControlNetExpert(name="solo", controlnet=controlnet, controlnet_cond=cond_t)
    uncond = torch.zeros_like(cond_t)
    cfg = SamplerConfig(
        K_SMC=K, n_steps=n_steps, composition="cfg_poe",
        use_fkc_weights=False, fkc_eta=0.0, seed=seed,
    )
    # Get latent shape from a reference latent
    latent_shape = (4, 64, 64, 64)  # ISO15: 256/4 = 64
    sp_t = torch.tensor([spacing], device=device, dtype=torch.float32).expand(K, -1).clone() * 100.0
    mod_t = torch.ones((K,), device=device, dtype=torch.long)
    particles, log_w, diag = sample_fkc_poe_n(
        experts=[dummy_expert],
        diffusion_unet=unet,
        noise_scheduler=noise_scheduler,
        scale_factor=scale_factor,
        spacing_tensor=sp_t,
        modality_tensor=mod_t,
        latent_shape=latent_shape,
        config=cfg,
        include_modality=True,
        include_body_region=False,
        uncond_controlnet_cond=uncond,
    )
    final_lat = snis_weighted_average(particles, log_w).unsqueeze(0)
    ct = decode_latent(ae, final_lat, scale_factor)
    return ct.detach().cpu().numpy().squeeze(), diag


def run_composition(unet, ae, noise_scheduler, scale_factor,
                    controlnet_A, cond_A, controlnet_B, cond_B,
                    K, n_steps, spacing, device, use_fkc=True, seed=1234):
    """Run two-expert CFG-PoE FKC composition."""
    expert_A = ControlNetExpert(name="A", controlnet=controlnet_A, controlnet_cond=cond_A)
    expert_B = ControlNetExpert(name="B", controlnet=controlnet_B, controlnet_cond=cond_B)
    # Uncond uses Expert A's ControlNet with zero A-mask (approximates prior)
    uncond_A = torch.zeros_like(cond_A)
    cfg = SamplerConfig(
        K_SMC=K, n_steps=n_steps, composition="cfg_poe",
        use_fkc_weights=use_fkc, fkc_eta=0.05 if use_fkc else 0.0, seed=seed,
    )
    latent_shape = (4, 64, 64, 64)
    sp_t = torch.tensor([spacing], device=device, dtype=torch.float32).expand(K, -1).clone() * 100.0
    mod_t = torch.ones((K,), device=device, dtype=torch.long)
    particles, log_w, diag = sample_fkc_poe_n(
        experts=[expert_A, expert_B],
        diffusion_unet=unet,
        noise_scheduler=noise_scheduler,
        scale_factor=scale_factor,
        spacing_tensor=sp_t,
        modality_tensor=mod_t,
        latent_shape=latent_shape,
        config=cfg,
        include_modality=True,
        include_body_region=False,
        uncond_controlnet_cond=uncond_A,
    )
    final_lat = snis_weighted_average(particles, log_w).unsqueeze(0)
    ct = decode_latent(ae, final_lat, scale_factor)
    return ct.detach().cpu().numpy().squeeze(), diag


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>FKC-PoE Pilot Report</title>
<style>
  body {{ font-family: monospace; margin: 20px; background: #f8f8f8; }}
  h1 {{ color: #333; }}
  h2 {{ color: #555; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
  h3 {{ color: #444; margin: 8px 0 4px; }}
  .section {{ background: white; border: 1px solid #ddd; border-radius: 4px;
              padding: 16px; margin: 12px 0; overflow-x: auto; }}
  .grid2 {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }}
  .grid4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }}
  img {{ max-width: 100%; border: 1px solid #eee; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td, th {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
  th {{ background: #eee; }}
  .label {{ font-size: 11px; color: #666; margin-top: 4px; text-align: center; }}
  .note {{ font-size: 12px; color: #888; font-style: italic; }}
</style>
</head>
<body>
<h1>FKC-PoE Pilot Report</h1>
<p>Generated: {timestamp}<br>
   Expert A: <code>{ckpt_A}</code><br>
   Expert B: <code>{ckpt_B}</code><br>
   AMOS22 case: <code>{ts_case}</code> &nbsp;|&nbsp;
   MMWHS case: <code>{mmwhs_case}</code></p>

<div class="section">
<h2>§1 Conditioning Masks</h2>
<p><b style="color:red">Red</b> = Expert A labels (AMOS22 abdominal IDs 1–15).
   <b style="color:cyan;background:#333;padding:0 3px">Cyan</b> = Expert B labels (MMWHS cardiac IDs 101–107).
   Triplane at respective organ centroids.</p>
<img src="data:image/png;base64,{mask_preview}" alt="mask preview">
</div>

<div class="section">
<h2>§2 Axial Montage — all 4 modes × 12 slices</h2>
<p class="note">Rows = generation modes. Columns = 12 evenly-spaced axial slices spanning organ union z-range (inferior → superior).
Red = Expert A (abdominal), Cyan = Expert B (cardiac).</p>
<img src="data:image/png;base64,{montage_axial}" alt="axial montage all modes">
</div>

<div class="section">
<h2>§3 Coronal Montage — all 4 modes × 10 slices</h2>
<p class="note">Rows = generation modes. Columns = 10 evenly-spaced coronal slices spanning organ y-range (posterior → anterior).</p>
<img src="data:image/png;base64,{montage_coronal}" alt="coronal montage all modes">
</div>

<div class="section">
<h2>§4 Sagittal Montage — all 4 modes × 10 slices</h2>
<p class="note">Rows = generation modes. Columns = 10 evenly-spaced sagittal slices spanning organ x-range (right → left).</p>
<img src="data:image/png;base64,{montage_sagittal}" alt="sagittal montage all modes">
</div>

<div class="section">
<h2>§5 Multi-Slice Axial Grid — all 4 modes × 8 z-positions</h2>
<p class="note">Rows = modes (A_only / B_only / poe_naive / poe_fkc). Columns = 8 equally-spaced z-positions across the full organ range.
Read down each column to compare modes at the same anatomy level.</p>
<img src="data:image/png;base64,{multipos_grid}" alt="multi-position comparison grid">
</div>

<div class="section">
<h2>§6 All-Modes Triplane — Abdominal Anchor (Expert A centroid)</h2>
<p class="note">Rows = modes. Columns = axial / coronal / sagittal. All 4 modes at identical planes centred on the abdominal organ cloud.</p>
<img src="data:image/png;base64,{triplane_all_abd}" alt="all-modes triplane abdominal">
</div>

<div class="section">
<h2>§7 All-Modes Triplane — Cardiac Anchor (Expert B centroid)</h2>
<p class="note">Rows = modes. Columns = axial / coronal / sagittal. All 4 modes at identical planes centred on the cardiac organ cloud.</p>
<img src="data:image/png;base64,{triplane_all_card}" alt="all-modes triplane cardiac">
</div>

<div class="section">
<h2>§8 Difference Maps — what composition adds over A_only</h2>
<p class="note">Absolute difference maps at the cardiac centroid axial slice (z = cardiac anchor).
Shows where Expert B's conditioning changes the CT relative to Expert A alone. Colormap: hot, range [0, 0.5].</p>
<div class="grid2">
  <div><img src="data:image/png;base64,{diff_bonly_vs_aonly}" alt="B_only - A_only">
    <p class="label">|B_only − A_only|</p></div>
  <div><img src="data:image/png;base64,{diff_naive_vs_aonly}" alt="poe_naive - A_only">
    <p class="label">|poe_naive − A_only|</p></div>
  <div><img src="data:image/png;base64,{diff_fkc_vs_aonly}" alt="poe_fkc - A_only">
    <p class="label">|poe_fkc − A_only|</p></div>
  <div><img src="data:image/png;base64,{diff_fkc_vs_naive}" alt="poe_fkc - poe_naive">
    <p class="label">|poe_fkc − poe_naive| &nbsp;(FKC correction vs naive PoE)</p></div>
</div>
</div>

<div class="section">
<h2>§9 FKC Diagnostics</h2>
<img src="data:image/png;base64,{ess_curves}" alt="ESS curves">
<br>
<img src="data:image/png;base64,{vip_curves}" alt="FK increment curves">
</div>

<div class="section">
<h2>§10 Run Summary</h2>
<table>
<tr><th>Mode</th><th>Wallclock (s)</th><th>ESS min</th><th>ESS max</th><th>log_w spread</th></tr>
{summary_rows}
</table>
</div>

<div class="section">
<h2>§11 Notes</h2>
<ul>
  <li>Expert A (AMOS22, 8-ch binarize_labels) trained on 200 abdominal cases:
  IDs 1–15 (spleen, kidneys, liver, stomach, aorta, IVC, pancreas, etc.) + ID 200 body silhouette.</li>
  <li>Expert B (MMWHS, 8-ch binarize_labels) trained on 18 cardiac cases:
  IDs 101–107 (LV_myo, L_atrium, L_ventricle, aorta_asc, R_ventricle, pulm_artery, R_atrium) + ID 200.</li>
  <li>Both experts use MAISI native 8-bit binary encoding and were fine-tuned from pretrained controlnet_3d_rflow-ct.pt.</li>
  <li>FKC signal (velocity inner product) near zero = experts make orthogonal requests (expected for spatially non-overlapping abdominal vs cardiac anatomy).</li>
  <li>ESS=K=2 throughout = uniform particle weights; FKC resampling provides no pressure in this pilot (K too small for spatial separation to create meaningful divergence).</li>
  <li>CT values in model output are normalised to approximately [−1, 1]; difference maps with values &gt;0.05 are visually significant.</li>
</ul>
</div>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--amos22-case", default="amos22_amos_0001",
                    help="AMOS22 case stem for Expert A conditioning.")
    ap.add_argument("--mmwhs-case", default="mmwhs_1001",
                    help="MMWHS case stem for Expert B conditioning.")
    ap.add_argument("--K", type=int, default=4, help="Number of SMC particles.")
    ap.add_argument("--n-steps", type=int, default=30, help="Denoising steps.")
    ap.add_argument("--out-dir", default="outputs/fkc_poe_pilot")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build base models (shared UNet + AE)
    unet, ae, noise_scheduler, scale_factor = build_models(device)

    # ---- Load Expert A ControlNet (8-ch binarize_labels, AMOS22)
    log("Loading Expert A ControlNet (8-ch, AMOS22) ...")
    controlnet_A = load_controlnet(
        str(PROJECT_ROOT / "configs/config_network_amos22_15ch.json"),
        CKPT_A, unet, device,
    )

    # ---- Load Expert B ControlNet (8-ch binarize_labels, MMWHS)
    log("Loading Expert B ControlNet (8-ch, MMWHS) ...")
    controlnet_B = load_controlnet(
        str(PROJECT_ROOT / "configs/config_network_mmwhs_7ch.json"),
        CKPT_B, unet, device,
    )

    log(f"GPU mem after model load: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")

    # ---- Load conditioning masks
    amos22_lbl_path = AMOS22_LABEL_DIR / f"{args.amos22_case}.nii.gz"
    mmwhs_lbl_path  = MMWHS_LABEL_DIR  / f"{args.mmwhs_case}.nii.gz"
    if not amos22_lbl_path.exists():
        log(f"ERROR: AMOS22 label not found: {amos22_lbl_path}")
        sys.exit(1)
    if not mmwhs_lbl_path.exists():
        log(f"ERROR: MMWHS label not found: {mmwhs_lbl_path}")
        sys.exit(1)

    log(f"Loading AMOS22 label: {amos22_lbl_path}")
    amos22_lbl_t = load_label(amos22_lbl_path, device)   # (1,1,256,256,256)
    cond_A = binarize_labels(amos22_lbl_t.long()).float()  # (1,8,256,256,256)

    log(f"Loading MMWHS label: {mmwhs_lbl_path}")
    mmwhs_lbl_t = load_label(mmwhs_lbl_path, device)  # (1,1,256,256,256)
    cond_B = binarize_labels(mmwhs_lbl_t.long()).float()  # (1,8,256,256,256)

    amos22_lbl_np = amos22_lbl_t.squeeze().cpu().numpy().astype(np.int32)
    mmwhs_lbl_np  = mmwhs_lbl_t.squeeze().cpu().numpy().astype(np.int32)
    amos22_fg = np.isin(amos22_lbl_np, AMOS22_IDS)
    mmwhs_fg  = np.isin(mmwhs_lbl_np, MMWHS_IDS)

    log(f"Expert A mask voxels: {int(amos22_fg.sum())}  Expert B mask voxels: {int(mmwhs_fg.sum())}")

    # ---- Render mask preview with blank CT (show label overlays)
    blank_ct = np.zeros((256, 256, 256), dtype=np.float32)
    mask_preview_b64 = render_slices_base64(
        blank_ct, f"Conditioning masks — A={args.amos22_case} (red)  B={args.mmwhs_case} (cyan)",
        lbl_a=amos22_fg.astype(np.float32), lbl_b=mmwhs_fg.astype(np.float32)
    )

    # ---- Run generations
    t0_total = time.time()
    results = {}
    ess_curves = {}
    logw_ranges = {}
    vip_curves = {}

    modes = [
        ("A_only",    lambda s: run_single_expert(unet, ae, noise_scheduler, scale_factor,
                                                  controlnet_A, cond_A, N_CH_A,
                                                  args.K, args.n_steps, SPACING_ISO15,
                                                  device, seed=s)),
        ("B_only",    lambda s: run_single_expert(unet, ae, noise_scheduler, scale_factor,
                                                  controlnet_B, cond_B, N_CH_B,
                                                  args.K, args.n_steps, SPACING_ISO15,
                                                  device, seed=s)),
        ("poe_naive", lambda s: run_composition(unet, ae, noise_scheduler, scale_factor,
                                                controlnet_A, cond_A, controlnet_B, cond_B,
                                                args.K, args.n_steps, SPACING_ISO15,
                                                device, use_fkc=False, seed=s)),
        ("poe_fkc",   lambda s: run_composition(unet, ae, noise_scheduler, scale_factor,
                                                controlnet_A, cond_A, controlnet_B, cond_B,
                                                args.K, args.n_steps, SPACING_ISO15,
                                                device, use_fkc=True, seed=s)),
    ]

    for mode_name, runner in modes:
        log(f"\n--- Running mode: {mode_name} ---")
        t0 = time.time()
        ct_np, diag = runner(1234)
        dt = time.time() - t0
        log(f"  done in {dt:.0f}s | ess: {min(diag.ess_curve):.2f}–{max(diag.ess_curve):.2f}")

        # Save NIfTI
        affine = np.diag([1.5, 1.5, 1.5, 1.0])
        nib.save(nib.Nifti1Image(ct_np, affine), str(out_dir / f"{mode_name}.nii.gz"))

        results[mode_name] = {"dt": dt, "diag": diag, "ct_np": ct_np}
        ess_curves[mode_name] = diag.ess_curve
        logw_ranges[mode_name] = (diag.logw_min_curve, diag.logw_max_curve)
        vip_curves[mode_name] = diag.velocity_inner_product_curve

    log(f"\nAll modes done in {time.time()-t0_total:.0f}s total")

    # ---- Build visualizations
    log("Rendering visualizations ...")
    cts = {m: results[m]["ct_np"] for m in results}
    abd_anchor  = _organ_centroid(amos22_fg)
    card_anchor = _organ_centroid(mmwhs_fg)
    log(f"  abd_anchor={abd_anchor}  card_anchor={card_anchor}")

    lbl_a_f = amos22_fg.astype(np.float32)
    lbl_b_f = mmwhs_fg.astype(np.float32)

    # §2 All-modes axial montage (12 slices)
    log("  §2 axial montage (12 slices × 4 modes) ...")
    montage_axial = render_all_modes_plane_montage(
        cts, "Axial — all 4 modes × 12 slices (organ z-range)", axis="axial",
        lbl_a=lbl_a_f, lbl_b=lbl_b_f, n_slices=12,
    )

    # §3 All-modes coronal montage (10 slices)
    log("  §3 coronal montage (10 slices × 4 modes) ...")
    montage_coronal = render_all_modes_plane_montage(
        cts, "Coronal — all 4 modes × 10 slices (organ y-range)", axis="coronal",
        lbl_a=lbl_a_f, lbl_b=lbl_b_f, n_slices=10,
    )

    # §4 All-modes sagittal montage (10 slices)
    log("  §4 sagittal montage (10 slices × 4 modes) ...")
    montage_sagittal = render_all_modes_plane_montage(
        cts, "Sagittal — all 4 modes × 10 slices (organ x-range)", axis="sagittal",
        lbl_a=lbl_a_f, lbl_b=lbl_b_f, n_slices=10,
    )

    # §5 Multi-slice axial grid (8 z-positions across full organ range)
    log("  §5 multi-position comparison (4 modes × 8 z) ...")
    fg_union = amos22_fg | mmwhs_fg
    if fg_union.any():
        z_all = np.where(fg_union.any(axis=(0, 1)))[0]
        z_positions = list(np.linspace(int(z_all.min()), int(z_all.max()), 8, dtype=int))
    else:
        z_positions = list(np.linspace(40, 215, 8, dtype=int))
    multipos_grid = render_multipos_comparison(
        cts, "Multi-slice axial grid — all 4 modes × 8 z-positions",
        z_positions, lbl_a=lbl_a_f, lbl_b=lbl_b_f,
    )

    # §6 All-modes triplane at abdominal anchor
    log("  §6 triplane (abdominal anchor, all modes) ...")
    triplane_all_abd = render_all_modes_triplane(
        cts, f"All-modes triplane — abdominal anchor {abd_anchor}",
        abd_anchor, lbl_a=lbl_a_f, lbl_b=lbl_b_f,
    )

    # §7 All-modes triplane at cardiac anchor
    log("  §7 triplane (cardiac anchor, all modes) ...")
    triplane_all_card = render_all_modes_triplane(
        cts, f"All-modes triplane — cardiac anchor {card_anchor}",
        card_anchor, lbl_a=lbl_a_f, lbl_b=lbl_b_f,
    )

    # §8 Difference maps at cardiac centroid slice
    log("  §8 difference maps ...")
    z_diff = card_anchor[2]
    diff_bonly_vs_aonly  = render_diff_map(cts["A_only"], cts["B_only"],
                                           "|B_only − A_only|",     z_diff)
    diff_naive_vs_aonly  = render_diff_map(cts["A_only"], cts["poe_naive"],
                                           "|poe_naive − A_only|",  z_diff)
    diff_fkc_vs_aonly    = render_diff_map(cts["A_only"], cts["poe_fkc"],
                                           "|poe_fkc − A_only|",    z_diff)
    diff_fkc_vs_naive    = render_diff_map(cts["poe_naive"], cts["poe_fkc"],
                                           "|poe_fkc − poe_naive|", z_diff)

    # §9 FKC diagnostics
    log("  §9 FKC curves ...")
    ess_b64 = render_ess_curve_base64(
        {k: v for k, v in ess_curves.items() if k in ("poe_naive", "poe_fkc")},
        {k: v for k, v in logw_ranges.items() if k in ("poe_naive", "poe_fkc")},
    )
    vip_b64 = render_vip_curve_base64(
        {k: v for k, v in vip_curves.items() if k in ("poe_naive", "poe_fkc")}
    )

    # §10 Summary table
    summary_rows = "\n".join(
        f"<tr><td>{m}</td><td>{results[m]['dt']:.0f}</td>"
        f"<td>{min(results[m]['diag'].ess_curve):.2f}</td>"
        f"<td>{max(results[m]['diag'].ess_curve):.2f}</td>"
        f"<td>[{min(results[m]['diag'].logw_min_curve):.2f}, "
        f"{max(results[m]['diag'].logw_max_curve):.2f}]</td></tr>"
        for m in results
    )

    html = HTML_TEMPLATE.format(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        ckpt_A=str(CKPT_A),
        ckpt_B=str(CKPT_B),
        ts_case=args.amos22_case,
        mmwhs_case=args.mmwhs_case,
        mask_preview=mask_preview_b64,
        montage_axial=montage_axial,
        montage_coronal=montage_coronal,
        montage_sagittal=montage_sagittal,
        multipos_grid=multipos_grid,
        triplane_all_abd=triplane_all_abd,
        triplane_all_card=triplane_all_card,
        diff_bonly_vs_aonly=diff_bonly_vs_aonly,
        diff_naive_vs_aonly=diff_naive_vs_aonly,
        diff_fkc_vs_aonly=diff_fkc_vs_aonly,
        diff_fkc_vs_naive=diff_fkc_vs_naive,
        ess_curves=ess_b64,
        vip_curves=vip_b64,
        summary_rows=summary_rows,
    )

    report_path = out_dir / "fkc_poe_pilot_report.html"
    with open(report_path, "w") as f:
        f.write(html)
    log(f"Report written: {report_path}")

    # Save JSON summary for downstream use
    summary = {
        "ts_case": args.amos22_case,
        "mmwhs_case": args.mmwhs_case,
        "K": args.K,
        "n_steps": args.n_steps,
        "modes": {
            m: {
                "wallclock_s": results[m]["dt"],
                "ess_min": float(min(results[m]["diag"].ess_curve)),
                "ess_max": float(max(results[m]["diag"].ess_curve)),
                "ess_last": float(results[m]["diag"].ess_curve[-1]),
                "vip_mean": float(np.mean(results[m]["diag"].velocity_inner_product_curve)),
                "logw_spread": float(max(results[m]["diag"].logw_max_curve) -
                                     min(results[m]["diag"].logw_min_curve)),
                "n_resample_events": len(results[m]["diag"].resample_events),
            }
            for m in results
        }
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Summary JSON written: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
