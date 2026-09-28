#!/usr/bin/env python
"""Re-aggregate v3 Dice_CF on n=50 with CORRECT cond mask.

Bug: existing reval uses cond_v3 = np.where(la_aligned != 0, la_aligned, lb_aligned),
which wipes spine IDs because la_aligned has body silhouette (200) everywhere inside
body. So v3 was being evaluated ONLY on abdominal organs.

Fix: use cond_v3 = max(la_aligned, lb_aligned) where they have organ IDs (not 200/0).
Equivalently: take spine ID if present, else abdominal ID, else 200 / 0.

Re-uses cached segmentations from outputs/_mixed_da_seg_cache/. Recomputes per-case
Dice_CF + paired comparison vs Oracle.
"""
from pathlib import Path
import json
import numpy as np
import nibabel as nib

REPO = Path(__file__).resolve().parents[1]
REP = REPO / "reports/verify_path1"
DATA = REPO / "data/phase1_ts_iso15"
CACHE = REPO / "outputs/_mixed_da_seg_cache"
MANIFEST = REPO / "data/nnunet_dataset_manifest.json"

manifest = json.loads(MANIFEST.read_text())
maisi_to_nn = {int(k): int(v) for k, v in manifest["maisi_to_nnunet"].items()}
evaluable_ids = sorted(maisi_to_nn.keys())

partition = json.loads((REPO / "data/label_partition.json").read_text())
expert_A_ids = set(partition["expert_A_abdominal"])
expert_B_ids = set(partition["expert_B_spine"])


def build_aligned(labels_U, keep_ids):
    out = np.full_like(labels_U, 200, dtype=np.int32)
    body = labels_U != 0
    out[~body] = 0
    for sid in keep_ids:
        out[labels_U == sid] = sid
    return out


def build_correct_cond_v3(labels_U):
    """v3 was generated with both A and B ControlNets. The true conditioning is
    the union of both expert's organ IDs. Where neither has an organ, the body
    silhouette (200) is in effect. Effectively this is labels_U + body silhouette,
    which is just labels_U with 200 in body-but-no-organ voxels.
    """
    out = np.full_like(labels_U, 200, dtype=np.int32)
    body = labels_U != 0
    out[~body] = 0
    for sid in expert_A_ids | expert_B_ids:
        out[labels_U == sid] = sid
    return out


def dice_per_label(pred, target, ids):
    out = {}
    for sid in ids:
        p = pred == sid; t = target == sid
        if not t.any() and not p.any():
            out[sid] = float("nan"); continue
        inter = int((p & t).sum()); denom = int(p.sum() + t.sum())
        out[sid] = float(2 * inter / max(denom, 1))
    return out


# Load existing summary to get case list and Oracle results (already correct)
mixed = json.loads((REP / "mixed_da_segmenter_summary_n50.json").read_text())
agg = json.loads((REP / "cases_n50.json").read_text())
cases = agg["clean_cases"]

# Rebuild v3 results with correct cond_v3
v3_results = {"v3_residual_sum": {}, "v3_cfg_poe": {}}
for case in cases:
    lu = nib.load(str(DATA / "labels_U" / f"{case}.nii.gz")).get_fdata().astype(np.int32)
    cond_v3 = build_correct_cond_v3(lu)
    ids_in_cond_v3 = [s for s in evaluable_ids if (cond_v3 == s).any()]

    for strat, slot in [("residual_sum", "v3_residual_sum"), ("cfg_poe", "v3_cfg_poe")]:
        cache_p = CACHE / f"{case}_{strat}_da.npy"
        if not cache_p.exists():
            continue
        pred = np.load(str(cache_p))
        dcf = dice_per_label(pred, cond_v3, ids_in_cond_v3)
        cf_vals = [v for v in dcf.values() if not np.isnan(v)]
        v3_results[slot][case] = {
            "dice_cf_mean": float(np.nanmean(cf_vals)) if cf_vals else float("nan"),
            "dice_cf_per_id": {str(k): float(v) for k, v in dcf.items()},
        }

# Save corrected v3 + original Oracle into a new summary
corrected = {
    "v3_residual_sum": v3_results["v3_residual_sum"],
    "v3_cfg_poe": v3_results["v3_cfg_poe"],
    "oracle": mixed["oracle"],  # Oracle was correct
}
out_p = REP / "mixed_da_segmenter_summary_n50_corrected.json"
out_p.write_text(json.dumps(corrected, indent=2))
print(f"wrote {out_p}")

# Aggregate
print(f"\n=== Corrected n=50 aggregate (cond_v3 = labels_U) ===")
for slot in ["v3_residual_sum", "v3_cfg_poe", "oracle"]:
    cfs = [v["dice_cf_mean"] for v in corrected[slot].values() if v and not np.isnan(v["dice_cf_mean"])]
    if cfs:
        print(f"  {slot:<22}  Dice_CF: {np.mean(cfs):.3f} ± {np.std(cfs):.3f}   n={len(cfs)}")

# Paired
cfg = corrected["v3_cfg_poe"]; ora = corrected["oracle"]
common = sorted(set(cfg.keys()) & set(ora.keys()))
diffs = [cfg[c]["dice_cf_mean"] - ora[c]["dice_cf_mean"] for c in common if not np.isnan(cfg[c]["dice_cf_mean"])]
wins = sum(1 for d in diffs if d > 0)
print(f"\n  paired cfg_poe vs oracle: Δ = {np.mean(diffs):+.3f} ± {np.std(diffs):.3f}")
print(f"  cfg_poe wins on {wins}/{len(diffs)} cases")

# Sign test
from math import comb
n = len(diffs)
extreme = min(wins, n - wins)
p_value = 2 * sum(comb(n, k) * 0.5**n for k in range(0, extreme + 1))
print(f"  sign-test p-value = {p_value:.2e}")
