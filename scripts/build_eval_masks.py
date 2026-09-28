#!/usr/bin/env python3
"""Build the 60-case Phase 1 eval dataset (mirrors build_controlnet_datasets.py
but for the eval split, not the controlnet_train split).

Output layout (same as Phase 1.1 train data, separate dir):
  data/maisi_controlnet_eval/
    images/eval_XXXX.nii.gz            # symlink to TS raw CT
    labels_expert_A_abd/eval_XXXX.nii.gz
    labels_expert_B_spine/eval_XXXX.nii.gz
    labels_union/eval_XXXX.nii.gz
    eval_index.json                     # case_id -> {ts_case, paths}

Smaller than the 1068-case train set; 60 cases takes ~15 min.
"""

import json
import os
from pathlib import Path

import numpy as np
import nibabel as nib


MEDGEN_ROOT = Path(__file__).resolve().parents[1]
TS_ROOT = Path(os.environ.get("TS_ROOT", "/path/to/TotalSegmentator"))
OUT_ROOT = MEDGEN_ROOT / "data" / "maisi_controlnet_eval"


def main():
    with open(MEDGEN_ROOT / "data" / "label_partition.json") as f:
        partition = json.load(f)

    ts_to_maisi = partition["ts_to_maisi"]
    expert_a_ids = set(partition["expert_A_abdominal"])
    expert_b_ids = set(partition["expert_B_spine"])

    expert_a_names = [n for n, mid in ts_to_maisi.items() if mid in expert_a_ids]
    expert_b_names = [n for n, mid in ts_to_maisi.items() if mid in expert_b_ids]

    eval_cases = partition["splits"]["eval"]
    print(f"Eval cases: {len(eval_cases)}")

    images_dir = OUT_ROOT / "images"
    labels_a_dir = OUT_ROOT / "labels_expert_A_abd"
    labels_b_dir = OUT_ROOT / "labels_expert_B_spine"
    labels_u_dir = OUT_ROOT / "labels_union"
    for d in (images_dir, labels_a_dir, labels_b_dir, labels_u_dir):
        d.mkdir(parents=True, exist_ok=True)

    index = {}

    for idx, ts_case in enumerate(eval_cases):
        case_id = f"eval_{idx:04d}"
        ct_src = TS_ROOT / ts_case / "ct.nii.gz"
        ct_link = images_dir / f"{case_id}.nii.gz"
        if not ct_link.exists():
            ct_link.symlink_to(ct_src)

        targets = [
            (labels_a_dir / f"{case_id}.nii.gz", expert_a_names),
            (labels_b_dir / f"{case_id}.nii.gz", expert_b_names),
            (labels_u_dir / f"{case_id}.nii.gz", expert_a_names + expert_b_names),
        ]
        if any(not p.exists() for p, _ in targets):
            ct_nii = nib.load(str(ct_src))
            shape = ct_nii.shape
            for out_path, organ_names in targets:
                if out_path.exists():
                    continue
                label = np.zeros(shape, dtype=np.uint8)
                for organ_name in organ_names:
                    organ_path = TS_ROOT / ts_case / "segmentations" / f"{organ_name}.nii.gz"
                    if not organ_path.exists():
                        continue
                    mask = nib.load(str(organ_path)).get_fdata() > 0
                    label[mask] = ts_to_maisi[organ_name]
                out_nii = nib.Nifti1Image(label, ct_nii.affine, ct_nii.header)
                out_nii.set_data_dtype(np.uint8)
                nib.save(out_nii, str(out_path))

        index[case_id] = {
            "ts_case": ts_case,
            "image": f"images/{case_id}.nii.gz",
            "label_A": f"labels_expert_A_abd/{case_id}.nii.gz",
            "label_B": f"labels_expert_B_spine/{case_id}.nii.gz",
            "label_union": f"labels_union/{case_id}.nii.gz",
        }
        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(eval_cases)} eval cases prepared")

    with open(OUT_ROOT / "eval_index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"Wrote {OUT_ROOT}/eval_index.json with {len(index)} entries")


if __name__ == "__main__":
    main()
