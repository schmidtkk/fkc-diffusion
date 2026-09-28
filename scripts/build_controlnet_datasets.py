#!/usr/bin/env python3
"""Build MAISI ControlNet training datasets for K=2 partial-label experts.

For each of the 1068 controlnet_train cases:
  - Symlink the raw CT volume (real HU) once
  - Build TWO multi-class mask volumes with MAISI label IDs:
      Expert A: only the 15 AMOS-style abdominal organ MAISI IDs filled
      Expert B: only the 27 VerSe-style spine MAISI IDs filled
  - Both share the same image link; labels differ.

Output layout:
  data/maisi_controlnet/
    images/case_XXXX.nii.gz                # symlink to TS raw CT
    labels_expert_A_abd/case_XXXX.nii.gz   # multi-class label, abd MAISI IDs
    labels_expert_B_spine/case_XXXX.nii.gz # multi-class label, spine MAISI IDs
    expert_A_abd_dataset.json              # MONAI-style {"training": [{...}]}
    expert_B_spine_dataset.json
    expert_AB_union_dataset.json            # joint oracle (Phase 1.5) — 42 union labels
    labels_union/case_XXXX.nii.gz           # union mask for oracle
"""

import json
import os
from pathlib import Path

import numpy as np
import nibabel as nib


MEDGEN_ROOT = Path(__file__).resolve().parents[1]
TS_ROOT = Path(os.environ.get("TS_ROOT", "/path/to/TotalSegmentator"))
OUT_ROOT = MEDGEN_ROOT / "data" / "maisi_controlnet"


def main():
    with open(MEDGEN_ROOT / "data" / "label_partition.json") as f:
        partition = json.load(f)

    ts_to_maisi = partition["ts_to_maisi"]
    expert_a_ids = set(partition["expert_A_abdominal"])    # 15 MAISI IDs
    expert_b_ids = set(partition["expert_B_spine"])         # 27 MAISI IDs
    union_ids = set(partition["union"])                     # 42 MAISI IDs

    expert_a_names = [n for n, mid in ts_to_maisi.items() if mid in expert_a_ids]
    expert_b_names = [n for n, mid in ts_to_maisi.items() if mid in expert_b_ids]
    print(f"Expert A: {len(expert_a_names)} TS organ files -> MAISI IDs {sorted(expert_a_ids)}")
    print(f"Expert B: {len(expert_b_names)} TS organ files -> MAISI IDs {sorted(expert_b_ids)}")

    train_cases = partition["splits"]["controlnet_train"]
    print(f"Controlnet train cases: {len(train_cases)}")

    images_dir = OUT_ROOT / "images"
    labels_a_dir = OUT_ROOT / "labels_expert_A_abd"
    labels_b_dir = OUT_ROOT / "labels_expert_B_spine"
    labels_u_dir = OUT_ROOT / "labels_union"
    for d in (images_dir, labels_a_dir, labels_b_dir, labels_u_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_a, train_b, train_u = [], [], []

    for idx, ts_case in enumerate(train_cases):
        case_id = f"case_{idx:04d}"

        # Symlink CT (lazy, only if missing)
        ct_src = TS_ROOT / ts_case / "ct.nii.gz"
        ct_link = images_dir / f"{case_id}.nii.gz"
        if not ct_link.exists():
            ct_link.symlink_to(ct_src)

        # Build the three masks (Expert A, Expert B, Union) if missing
        targets_specs = [
            (labels_a_dir / f"{case_id}.nii.gz", expert_a_names, ts_to_maisi),
            (labels_b_dir / f"{case_id}.nii.gz", expert_b_names, ts_to_maisi),
            (labels_u_dir / f"{case_id}.nii.gz", expert_a_names + expert_b_names, ts_to_maisi),
        ]

        needs_build = any(not p.exists() for p, _, _ in targets_specs)
        if needs_build:
            ct_nii = nib.load(str(ct_src))
            shape = ct_nii.shape
            for out_path, organ_names, name_to_id in targets_specs:
                if out_path.exists():
                    continue
                label = np.zeros(shape, dtype=np.uint8)
                for organ_name in organ_names:
                    organ_path = TS_ROOT / ts_case / "segmentations" / f"{organ_name}.nii.gz"
                    if not organ_path.exists():
                        continue
                    mask = nib.load(str(organ_path)).get_fdata() > 0
                    label[mask] = name_to_id[organ_name]  # uint8 fits since MAISI IDs <= 132
                out_nii = nib.Nifti1Image(label, ct_nii.affine, ct_nii.header)
                out_nii.set_data_dtype(np.uint8)
                nib.save(out_nii, str(out_path))

        # Relative paths so JSON is portable
        img_rel = f"images/{case_id}.nii.gz"
        train_a.append({"image": img_rel, "label": f"labels_expert_A_abd/{case_id}.nii.gz"})
        train_b.append({"image": img_rel, "label": f"labels_expert_B_spine/{case_id}.nii.gz"})
        train_u.append({"image": img_rel, "label": f"labels_union/{case_id}.nii.gz"})

        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(train_cases)} cases prepared")

    # Write dataset JSONs
    for jsonpath, train_list, expert_name in [
        (OUT_ROOT / "expert_A_abd_dataset.json", train_a, "Expert A (15 abd organs)"),
        (OUT_ROOT / "expert_B_spine_dataset.json", train_b, "Expert B (27 spine labels)"),
        (OUT_ROOT / "expert_AB_union_dataset.json", train_u, "Joint Oracle (42-label union)"),
    ]:
        payload = {
            "name": expert_name,
            "description": "TS controlnet_train cases, partial MAISI-label subset",
            "training": train_list,
            "validation": [],
        }
        with open(jsonpath, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {jsonpath} ({len(train_list)} cases)")

    print(f"\nDone. Output root: {OUT_ROOT}")


if __name__ == "__main__":
    main()
