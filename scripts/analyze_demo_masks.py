"""Analyze MAISI C4KC-KiTS demo dataset mask label statistics."""
import nibabel as nib
import numpy as np
import glob
import os
from collections import defaultdict

DATA_ROOT = str(Path(__file__).resolve().parents[1] / "maisi/dataset/C4KC-KiTS_subset")

# MAISI label ID -> name mapping (partial, key ones)
MAISI_NAMES = {
    1: "spleen", 2: "kidney_R", 3: "kidney_L", 4: "gallbladder", 5: "liver",
    6: "stomach", 7: "aorta", 8: "IVC", 9: "pancreas", 10: "adrenal_R",
    11: "adrenal_L", 12: "lung_upper_lobe_L", 13: "lung_lower_lobe_L",
    14: "lung_upper_lobe_R", 15: "lung_middle_lobe_R", 16: "lung_lower_lobe_R",
    17: "esophagus", 18: "trachea", 19: "thyroid_gland",
    55: "kidney_cyst_R", 56: "kidney_cyst_L",
    57: "vertebrae_C1", 58: "vertebrae_C2", 59: "vertebrae_C3",
    60: "vertebrae_C4", 61: "vertebrae_C5", 62: "vertebrae_C6",
    63: "vertebrae_C7", 64: "vertebrae_T1",
}

def analyze(tag, files):
    print(f"\n{'='*70}")
    print(f"Mask type: {tag}   ({len(files)} files)")
    print(f"{'='*70}")

    all_labels = defaultdict(int)
    total_vox = 0

    for f in sorted(files):
        case = os.path.basename(os.path.dirname(f))
        arr = nib.load(f).get_fdata(dtype=np.float32).astype(np.int32)
        n = arr.size
        total_vox += n
        u, c = np.unique(arr, return_counts=True)
        for ui, ci in zip(u.tolist(), c.tolist()):
            all_labels[ui] += ci
        fg = [int(ui) for ui in u if ui != 0]
        bg_vox = int(c[u == 0][0]) if 0 in u else 0
        print(f"  {case}: shape={arr.shape}  #fg_classes={len(fg)}"
              f"  fg_ratio={1 - bg_vox/n:.2%}  "
              f"  labels={sorted(fg)}")

    fg_labels = {k: v for k, v in all_labels.items() if k != 0}
    print(f"\n  --- Aggregate ---")
    print(f"  Total voxels  : {total_vox:,}")
    print(f"  Background    : {all_labels[0]/total_vox:.2%}")
    print(f"  #FG classes   : {len(fg_labels)}")
    if fg_labels:
        print(f"  Label ID range: {min(fg_labels)} ~ {max(fg_labels)}")
        print()
        print(f"  {'ID':>5}  {'Name':<25} {'Voxels':>12}  {'% of volume':>11}")
        print(f"  {'-'*5}  {'-'*25} {'-'*12}  {'-'*11}")
        for lbl in sorted(fg_labels.keys()):
            name = MAISI_NAMES.get(lbl, "---")
            cnt = fg_labels[lbl]
            print(f"  {lbl:>5}  {name:<25} {cnt:>12,}  {cnt/total_vox:>10.4%}")


files_orig = glob.glob(DATA_ROOT + "/*/mask_zs*.nii.gz")
files_comb = glob.glob(DATA_ROOT + "/*/mask_combined_label*.nii.gz")
files_pseudo = glob.glob(DATA_ROOT + "/*/mask_pseudo_label*.nii.gz")

analyze("mask_zs  (original segmentation)", files_orig)
analyze("mask_combined_label_wbdm  (MAISI combined)", files_comb)
analyze("mask_pseudo_label_wbdm  (MAISI pseudo)", files_pseudo)
