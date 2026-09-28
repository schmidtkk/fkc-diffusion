#!/usr/bin/env python3
"""Compute PSNR / SSIM metrics on FKC-PoE inference outputs.

Reads NIfTIs from an output directory (e.g. outputs/fkc_poe_amos22_mmwhs_r6/)
and computes:
  1. Cross-mode PSNR / SSIM matrix (all 4 modes against each other)
  2. Each generated mode vs. reference CT (AMOS22 + MMWHS), normalized to [-1,1]
  3. Intensity statistics per mode
  4. Appends a §12 Metrics section to the HTML report in that directory

Usage:
  python scripts/compute_fkc_metrics.py --out-dir outputs/fkc_poe_amos22_mmwhs_r6
"""

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REF_AMOS22 = PROJECT_ROOT / "data/phase2_amos22_iso15/images/amos22_amos_0001.nii.gz"
REF_MMWHS  = PROJECT_ROOT / "data/phase2_mmwhs_iso15/images/mmwhs_1001.nii.gz"
MODES = ["A_only", "B_only", "poe_naive", "poe_fkc"]

DATA_RANGE = 1.0   # generated CTs are in approximately [0, 1] (MAISI AE output after [0,1] HU norm)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_reference_hu(path: Path, target_shape=(256, 256, 256)) -> np.ndarray:
    """Load reference CT, normalize HU to [0, 1] to match MAISI AE decode output."""
    arr = nib.load(str(path)).get_fdata().astype(np.float32)
    arr = np.clip(arr, -1000.0, 1000.0)
    arr = (arr + 1000.0) / 2000.0   # → [0, 1] matching MAISI training normalization
    # Crop or pad to target_shape (center-crop each axis)
    for ax in range(3):
        sz = arr.shape[ax]
        tsz = target_shape[ax]
        if sz > tsz:
            start = (sz - tsz) // 2
            idx = [slice(None)] * 3
            idx[ax] = slice(start, start + tsz)
            arr = arr[tuple(idx)]
        elif sz < tsz:
            pad = [(0, 0)] * 3
            pad_before = (tsz - sz) // 2
            pad[ax] = (pad_before, tsz - sz - pad_before)
            arr = np.pad(arr, pad, mode="constant", constant_values=0.0)
    return arr


def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(psnr_fn(a, b, data_range=DATA_RANGE))


def compute_ssim(a: np.ndarray, b: np.ndarray, n_slices: int = 64) -> float:
    """3D SSIM on center n_slices axial slices (full 256^3 is slow)."""
    z_mid = a.shape[2] // 2
    z0 = max(0, z_mid - n_slices // 2)
    z1 = min(a.shape[2], z0 + n_slices)
    return float(ssim_fn(a[:, :, z0:z1], b[:, :, z0:z1], data_range=DATA_RANGE))


def intensity_stats(arr: np.ndarray) -> dict:
    return {
        "mean": float(arr.mean()),
        "std":  float(arr.std()),
        "p5":   float(np.percentile(arr,  5)),
        "p50":  float(np.percentile(arr, 50)),
        "p95":  float(np.percentile(arr, 95)),
    }


def fmt(v: float) -> str:
    return f"{v:.3f}"


def html_metrics_section(cross_psnr, cross_ssim, ref_psnr, ref_ssim,
                          istats, modes) -> str:
    # Cross-mode PSNR table
    def psnr_table():
        header = "<tr><th>Mode</th>" + "".join(f"<th>{m}</th>" for m in modes) + "</tr>\n"
        rows = ""
        for i, m in enumerate(modes):
            cells = "".join(
                f"<td>{'—' if i==j else fmt(cross_psnr[i][j])}</td>"
                for j in range(len(modes))
            )
            rows += f"<tr><td><b>{m}</b></td>{cells}</tr>\n"
        return f"<table>{header}{rows}</table>"

    def ssim_table():
        header = "<tr><th>Mode</th>" + "".join(f"<th>{m}</th>" for m in modes) + "</tr>\n"
        rows = ""
        for i, m in enumerate(modes):
            cells = "".join(
                f"<td>{'—' if i==j else fmt(cross_ssim[i][j])}</td>"
                for j in range(len(modes))
            )
            rows += f"<tr><td><b>{m}</b></td>{cells}</tr>\n"
        return f"<table>{header}{rows}</table>"

    def ref_table():
        header = "<tr><th>Mode</th><th>PSNR vs AMOS22</th><th>PSNR vs MMWHS</th><th>SSIM vs AMOS22</th><th>SSIM vs MMWHS</th></tr>\n"
        rows = ""
        for i, m in enumerate(modes):
            rows += (
                f"<tr><td><b>{m}</b></td>"
                f"<td>{fmt(ref_psnr[i][0])} dB</td>"
                f"<td>{fmt(ref_psnr[i][1])} dB</td>"
                f"<td>{fmt(ref_ssim[i][0])}</td>"
                f"<td>{fmt(ref_ssim[i][1])}</td></tr>\n"
            )
        return f"<table>{header}{rows}</table>"

    def istats_table():
        header = "<tr><th>Mode</th><th>mean</th><th>std</th><th>p5</th><th>p50</th><th>p95</th></tr>\n"
        rows = ""
        for m in modes:
            s = istats[m]
            rows += (
                f"<tr><td><b>{m}</b></td>"
                f"<td>{fmt(s['mean'])}</td><td>{fmt(s['std'])}</td>"
                f"<td>{fmt(s['p5'])}</td><td>{fmt(s['p50'])}</td><td>{fmt(s['p95'])}</td></tr>\n"
            )
        return f"<table>{header}{rows}</table>"

    css = """
<style>
.metrics-section table { border-collapse:collapse; margin-bottom:1em; font-size:0.9em; }
.metrics-section th, .metrics-section td { border:1px solid #888; padding:5px 10px; text-align:center; }
.metrics-section th { background:#333; color:#fff; }
.metrics-section h3 { margin-top:1.2em; }
.metrics-section p.note { color:#aaa; font-size:0.85em; }
</style>"""

    return f"""
{css}
<div class="section metrics-section">
<h2>§12 Quantitative Metrics</h2>

<h3>Cross-mode PSNR (dB) — higher = more similar pair</h3>
<p class="note">Full 256³ volume comparison. Data range = 2.0 (normalized [-1, 1]).</p>
{psnr_table()}

<h3>Cross-mode SSIM — higher = more structurally similar</h3>
<p class="note">Computed on central 64 axial slices for speed.</p>
{ssim_table()}

<h3>vs. Reference CT (pixel-level, ≠ perceptual quality)</h3>
<p class="note">Reference AMOS22 = <code>amos22_amos_0001.nii.gz</code> (clip HU[-1000,1000]→[0,1]),
Reference MMWHS = <code>mmwhs_1001.nii.gz</code> (clip HU[-1000,1000]→[0,1]).
<b>Note:</b> generated CTs are new random draws conditioned on mask shape, not reconstructions
of a specific patient — low pixel-level similarity vs. reference is expected and does not imply
poor quality.  Use this table to compare modes relative to each other, not as an absolute quality score.</p>
{ref_table()}

<h3>Intensity statistics per mode (MAISI AE output ≈ [0, 1])</h3>
{istats_table()}
</div>
"""


def patch_html_report(report_path: Path, metrics_section_html: str):
    with open(report_path) as f:
        html = f.read()
    if "§12 Quantitative Metrics" in html:
        # Remove existing §12 block (between <div class="section metrics-section"> ... </div>)
        import re
        html = re.sub(
            r'\n<style>\s*\.metrics-section.*?</div>\s*\n',
            '\n', html, flags=re.DOTALL
        )
    # Insert before </body>
    html = html.replace("</body>", metrics_section_html + "\n</body>")
    with open(report_path, "w") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/fkc_poe_amos22_mmwhs_r6")
    ap.add_argument("--ssim-slices", type=int, default=64,
                    help="Number of center axial slices for SSIM (full is slow).")
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir

    # ---- Load generated volumes
    log("Loading generated NIfTIs ...")
    vols = {}
    for m in MODES:
        p = out_dir / f"{m}.nii.gz"
        if not p.exists():
            log(f"  WARNING: {p} not found — skipping mode {m}")
            continue
        vols[m] = nib.load(str(out_dir / f"{m}.nii.gz")).get_fdata().astype(np.float32)
        log(f"  {m}: shape={vols[m].shape}  range=[{vols[m].min():.3f}, {vols[m].max():.3f}]")

    modes = [m for m in MODES if m in vols]
    n = len(modes)

    # ---- Load reference CTs
    log("Loading reference CTs ...")
    target_shape = vols[modes[0]].shape
    ref_a = load_reference_hu(REF_AMOS22, target_shape)
    ref_b = load_reference_hu(REF_MMWHS,  target_shape)
    log(f"  ref_A shape={ref_a.shape}  range=[{ref_a.min():.3f}, {ref_a.max():.3f}]")
    log(f"  ref_B shape={ref_b.shape}  range=[{ref_b.min():.3f}, {ref_b.max():.3f}]")

    # ---- Cross-mode PSNR matrix
    log("Computing cross-mode PSNR ...")
    cross_psnr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                cross_psnr[i, j] = compute_psnr(vols[modes[i]], vols[modes[j]])

    # ---- Cross-mode SSIM matrix
    log(f"Computing cross-mode SSIM (center {args.ssim_slices} slices) ...")
    cross_ssim = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = compute_ssim(vols[modes[i]], vols[modes[j]], args.ssim_slices)
            cross_ssim[i, j] = v
            cross_ssim[j, i] = v

    # ---- vs. reference
    log("Computing vs-reference PSNR/SSIM ...")
    ref_psnr = np.zeros((n, 2))
    ref_ssim = np.zeros((n, 2))
    for i, m in enumerate(modes):
        ref_psnr[i, 0] = compute_psnr(vols[m], ref_a)
        ref_psnr[i, 1] = compute_psnr(vols[m], ref_b)
        ref_ssim[i, 0] = compute_ssim(vols[m], ref_a, args.ssim_slices)
        ref_ssim[i, 1] = compute_ssim(vols[m], ref_b, args.ssim_slices)

    # ---- Intensity stats
    log("Computing intensity statistics ...")
    istats = {m: intensity_stats(vols[m]) for m in modes}

    # ---- Print summary
    print("\n===== Cross-mode PSNR (dB) =====")
    header = f"{'':12s}" + "".join(f"{m:14s}" for m in modes)
    print(header)
    for i, m in enumerate(modes):
        row = f"{m:12s}" + "".join(
            f"{'—':14s}" if i == j else f"{cross_psnr[i,j]:14.3f}"
            for j in range(n)
        )
        print(row)

    print("\n===== Cross-mode SSIM =====")
    print(header)
    for i, m in enumerate(modes):
        row = f"{m:12s}" + "".join(
            f"{'—':14s}" if i == j else f"{cross_ssim[i,j]:14.4f}"
            for j in range(n)
        )
        print(row)

    print("\n===== vs. Reference =====")
    print(f"{'Mode':12s}  {'PSNR_A':>8s}  {'PSNR_B':>8s}  {'SSIM_A':>8s}  {'SSIM_B':>8s}")
    for i, m in enumerate(modes):
        print(f"{m:12s}  {ref_psnr[i,0]:8.3f}  {ref_psnr[i,1]:8.3f}  "
              f"{ref_ssim[i,0]:8.4f}  {ref_ssim[i,1]:8.4f}")

    print("\n===== Intensity stats (MAISI AE output ≈ [0,1]) =====")

    print(f"{'Mode':12s}  {'mean':>7s}  {'std':>7s}  {'p5':>7s}  {'p50':>7s}  {'p95':>7s}")
    for m in modes:
        s = istats[m]
        print(f"{m:12s}  {s['mean']:7.4f}  {s['std']:7.4f}  "
              f"{s['p5']:7.4f}  {s['p50']:7.4f}  {s['p95']:7.4f}")

    # ---- Save JSON
    metrics_json = {
        "modes": modes,
        "cross_psnr_db": cross_psnr.tolist(),
        "cross_ssim": cross_ssim.tolist(),
        "ref_psnr_db": ref_psnr.tolist(),
        "ref_ssim": ref_ssim.tolist(),
        "intensity_stats": istats,
    }
    json_path = out_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    log(f"Metrics JSON written: {json_path}")

    # ---- Patch HTML report
    report_path = out_dir / "fkc_poe_pilot_report.html"
    if report_path.exists():
        section_html = html_metrics_section(
            cross_psnr, cross_ssim, ref_psnr, ref_ssim, istats, modes
        )
        patch_html_report(report_path, section_html)
        log(f"HTML report patched: {report_path}")
    else:
        log(f"WARNING: report not found at {report_path}, skipping HTML patch")


if __name__ == "__main__":
    main()
