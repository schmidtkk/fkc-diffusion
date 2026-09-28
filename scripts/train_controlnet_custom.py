#!/usr/bin/env python3
"""Custom ControlNet trainer for medgen-fkc — bypasses MAISI's MONAI CacheDataset.

The official `maisi/scripts/train_controlnet.py` silently crashes on our dataset
(suspected fault: CacheDataset + multi-worker DataLoader fork/cache pressure).
The minimal smoke test (`scripts/minimal_smoke.py`) confirms MAISI's UNet +
ControlNet forward + L1 loss path works correctly on a single case.

This custom trainer:
  - Loads NIfTI latents + labels directly via nibabel in the main thread (no
    workers, no CacheDataset, no fork).
  - Shuffles per epoch with a fixed seed.
  - Logs loss + GPU mem + timing every iteration.
  - Saves a `controlnet_state_dict` checkpoint at end-of-epoch.
  - Supports --resume from a previous checkpoint.

Usage (detached / job runner):
  python -u <repository>/scripts/train_controlnet_custom.py \
    --env-config configs/env_smoke_expert_A.json \
    --model-config configs/config_smoke_5ep.json \
    --network-config configs/config_network_rflow.json
"""

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import random
import socket
import sys
import time
from copy import deepcopy
from pathlib import Path
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import nibabel as nib
import monai
import numpy as np
import scipy
import skimage
import torch
from torch.amp import GradScaler, autocast

REPO_ROOT = Path(__file__).resolve().parents[1]
MAISI_ROOT = REPO_ROOT / "maisi"
sys.path.insert(0, str(MAISI_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from monai.networks.utils import copy_model_state  # noqa: E402
from monai.networks.schedulers import RFlowScheduler  # noqa: E402
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import (  # noqa: E402
    AutoencoderKlMaisi,
)
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (  # noqa: E402
    DiffusionModelUNetMaisi,
)
from monai.apps.generation.maisi.networks.controlnet_maisi import (  # noqa: E402
    ControlNetMaisi,
)

from maisi.scripts.utils import define_instance, binarize_labels  # noqa: E402
from maisi.scripts.diff_model_setting import load_config  # noqa: E402
from src.cond_encoding import (  # noqa: E402
    to_one_hot as _to_one_hot,
    N_COND_CHANNELS,
)
from src.fov_training_proof import (  # noqa: E402
    COMPLETION_SCHEMA,
    FARM04_HOST,
    IMPORTANCE_ESTIMATOR,
    IMPORTANCE_SCHEMA,
    IMPORTANCE_SCHEMA_VERSION,
    RUN_SIDECAR_SCHEMA,
    TrainingProofError,
    atomic_write_json,
    canonical_sha256,
    signed_payload,
    source_object_sha256,
    tensor_mapping_sha256,
    validate_completion_receipt,
    validate_importance_payload,
    validate_execution_environment_record,
    validate_remote_path_contract,
    validate_repository_snapshot_record,
    validate_semantic_source_files,
    validate_signed_payload,
    validate_validation_draw_contract,
    verify_recursive_sha256sums,
)
from src.fov_latent_mask import (  # noqa: E402
    MASK_ALGORITHM_ID,
    decode_exact_affine_record,
    equal_examination_masked_l1,
    latent_mask_content_sha256,
    mask_contract_sha256,
    masked_loss_contract,
    masked_loss_contract_sha256,
    validate_binary_mask,
    zero_fill_shift,
)


SEMANTIC_REPOSITORY_RELATIVE_PATHS = {
    "trainer": "scripts/train_controlnet_custom.py",
    "proof_helper": "src/fov_training_proof.py",
    "contract_builder": "scripts/build_fov_training_contract.py",
    "maisi_utils": "maisi/scripts/utils.py",
    "maisi_diff_model_setting": "maisi/scripts/diff_model_setting.py",
    "cond_encoding": "src/cond_encoding.py",
    "latent_mask_helper": "src/fov_latent_mask.py",
}
REQUIRED_REPOSITORY_SNAPSHOT_PATHS = tuple(
    sorted(
        {
            *SEMANTIC_REPOSITORY_RELATIVE_PATHS.values(),
            "maisi/configs/config_network_rflow.json",
            "maisi/configs/modality_mapping.json",
            "configs/fov_analysis_v3.json",
            "configs/fov_continual_success_v2.json",
        }
    )
)

MASKED_LOSS_CONTRACT = masked_loss_contract()
MASKED_LOSS_CONTRACT_SHA256 = masked_loss_contract_sha256()


# binarize_labels (imported from maisi/scripts/utils.py) is the native MAISI 8-bit encoding.
# Each integer label ID is decomposed into its 8-bit binary representation, one bit per channel.
# ID 200 (body silhouette) must be present in label volumes — see scripts/add_body_silhouette.py.


def log(msg):
    print(f"[{time.strftime('%X')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    """Return a reproducible content fingerprint for a run input."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _current_process_argv() -> list[str]:
    return [
        str(Path(os.path.abspath(sys.executable))),
        "-u",
        str(Path(sys.argv[0]).resolve()),
        *sys.argv[1:],
    ]


def _observed_execution_environment() -> dict:
    executable = Path(os.path.abspath(sys.executable))
    return {
        "target_host": socket.gethostname().split(".", 1)[0].upper(),
        "python_executable": {
            "path": str(executable),
            "sha256": sha256_file(executable),
        },
        "versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "cudnn": str(torch.backends.cudnn.version()),
            "monai": str(monai.__version__),
            "numpy": str(np.__version__),
            "scipy": str(scipy.__version__),
            "nibabel": str(nib.__version__),
            "scikit_image": str(skimage.__version__),
        },
        "monai_source_sha256": {
            "monai.networks.utils.copy_model_state": source_object_sha256(
                copy_model_state
            ),
            "monai.networks.schedulers.RFlowScheduler": source_object_sha256(
                RFlowScheduler
            ),
            "monai.apps.generation.maisi.networks.autoencoderkl_maisi.AutoencoderKlMaisi": source_object_sha256(
                AutoencoderKlMaisi
            ),
            "monai.apps.generation.maisi.networks.diffusion_model_unet_maisi.DiffusionModelUNetMaisi": source_object_sha256(
                DiffusionModelUNetMaisi
            ),
            "monai.apps.generation.maisi.networks.controlnet_maisi.ControlNetMaisi": source_object_sha256(
                ControlNetMaisi
            ),
        },
    }


def _require_import_source(value, expected_relative: str, label: str) -> None:
    source = inspect.getsourcefile(value)
    expected = REPO_ROOT / expected_relative
    if source is None or Path(source).resolve() != expected.resolve():
        raise RuntimeError(
            f"Imported {label} does not come from the location-relative shared repository"
        )


def _locked_production_preflight(run_contract: dict) -> None:
    """Verify remote code, environment, and signed roots before data/GPU work."""

    if run_contract.get("production_layout") is not True:
        return
    if run_contract.get("target_host") != FARM04_HOST:
        raise RuntimeError("production run contract does not target FARM04")
    try:
        validate_remote_path_contract(run_contract.get("remote_path_contract", {}))
        validate_repository_snapshot_record(
            run_contract.get("repository_snapshot", {}),
            required_relative_paths=REQUIRED_REPOSITORY_SNAPSHOT_PATHS,
        )
        validate_execution_environment_record(
            run_contract.get("execution_environment", {}),
            observed=_observed_execution_environment(),
        )
        validate_semantic_source_files(
            REPO_ROOT,
            run_contract.get("semantic_source_files", {}),
            required_names=tuple(SEMANTIC_REPOSITORY_RELATIVE_PATHS),
        )
    except TrainingProofError as error:
        raise RuntimeError(str(error)) from error
    if Path(os.path.abspath(sys.executable)) != Path(
        str(run_contract.get("python_executable"))
    ):
        raise RuntimeError("live sys.executable differs from the planned interpreter")
    if Path(__file__).resolve() != Path(str(run_contract.get("trainer_path"))).resolve():
        raise RuntimeError("trainer is not executing from the frozen shared snapshot")
    for value, relative, label in (
        (binarize_labels, "maisi/scripts/utils.py", "binarize_labels"),
        (define_instance, "maisi/scripts/utils.py", "define_instance"),
        (
            load_config,
            "maisi/scripts/diff_model_setting.py",
            "load_config",
        ),
        (_to_one_hot, "src/cond_encoding.py", "to_one_hot"),
        (
            validate_signed_payload,
            "src/fov_training_proof.py",
            "training proof helper",
        ),
        (
            equal_examination_masked_l1,
            "src/fov_latent_mask.py",
            "RF73 latent-mask helper",
        ),
    ):
        _require_import_source(value, relative, label)
    signed_bundles = run_contract.get("runtime_signed_bundles")
    if not isinstance(signed_bundles, dict) or not signed_bundles:
        raise RuntimeError("production run lacks runtime signed-bundle bindings")
    for name, record in signed_bundles.items():
        if not isinstance(record, dict):
            raise RuntimeError(f"runtime signed-bundle record is malformed: {name}")
        try:
            verify_recursive_sha256sums(
                Path(str(record.get("root"))),
                expected_sha256s_sha256=str(record.get("sha256s_sha256")),
                allowed_unchecksummed=record.get("allowed_unchecksummed", []),
                require_sorted=record.get("require_sorted") is True,
            )
        except TrainingProofError as error:
            raise RuntimeError(f"runtime signed bundle {name}: {error}") from error
    for value in sys.argv[1:]:
        if value == "/data" or value.startswith("/data/"):
            raise RuntimeError("production argv contains a local-only /data path")


def state_dict_sha256(state_dict) -> str:
    """Hash names, shapes, dtypes, and bytes of a model state dictionary."""
    h = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("ascii"))
        h.update(str(tensor.dtype).encode("ascii"))
        h.update(tensor.numpy().tobytes(order="C"))
    return h.hexdigest()


def _load_json_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain one JSON object: {path}")
    return value


def _atomic_torch_save(payload, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_resume_checkpoint_bindings(
    checkpoint: dict, *, run_fingerprints: dict
) -> None:
    """Reject legacy or drifted checkpoints before restoring any state."""

    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("run_fingerprints") != run_fingerprints
        or checkpoint.get("latent_mask_contract_sha256")
        != mask_contract_sha256()
        or checkpoint.get("latent_mask_helper_sha256")
        != run_fingerprints.get("latent_mask_helper_sha256")
        or checkpoint.get("masked_loss_contract") != MASKED_LOSS_CONTRACT
        or checkpoint.get("masked_loss_contract_sha256")
        != MASKED_LOSS_CONTRACT_SHA256
    ):
        raise RuntimeError(
            "resume checkpoint lacks the exact active RF73 masked-loss binding"
        )


def _validate_locked_run_sidecar(
    path: Path, *, expected_hash: str, actual_cli_args: list[str]
) -> dict:
    payload = _load_json_object(path, "run-contract sidecar")
    try:
        unsigned = validate_signed_payload(
            payload, schema=RUN_SIDECAR_SCHEMA, label="run-contract sidecar"
        )
    except TrainingProofError as error:
        raise RuntimeError(str(error)) from error
    if unsigned.get("run_contract_sha256") != expected_hash:
        raise RuntimeError("run-contract SHA-256 differs from the command line")
    run_contract = unsigned.get("run_contract")
    if not isinstance(run_contract, dict) or canonical_sha256(run_contract) != expected_hash:
        raise RuntimeError("signed run-contract payload does not match its canonical SHA-256")
    if unsigned.get("planned_cli_args") != actual_cli_args:
        raise RuntimeError("actual trainer argv differs from the complete planned argv")
    if Path(str(run_contract.get("trainer_path"))).resolve() != Path(
        sys.argv[0]
    ).resolve():
        raise RuntimeError("actual trainer executable differs from the planned process argv")
    if Path(str(run_contract.get("python_executable"))) != Path(
        os.path.abspath(sys.executable)
    ):
        raise RuntimeError("actual sys.executable differs from the planned interpreter")
    process_template = run_contract.get("planned_process_argv_template")
    if not isinstance(process_template, list) or process_template != [
        run_contract.get("python_executable"),
        "-u",
        run_contract.get("trainer_path"),
        *run_contract.get("planned_cli_args_template", []),
    ]:
        raise RuntimeError("run contract lacks the complete process argv template")
    return payload


def _strict_manifest_entries(path: Path, label: str) -> list[dict]:
    payload = _load_json_object(path, label)
    helper_hash = sha256_file(REPO_ROOT / "src/fov_latent_mask.py")
    if (
        payload.get("latent_loss_mask_contract_sha256")
        != mask_contract_sha256()
        or payload.get("latent_loss_mask_helper_sha256") != helper_hash
        or payload.get("masked_loss_contract") != MASKED_LOSS_CONTRACT
        or payload.get("masked_loss_contract_sha256")
        != MASKED_LOSS_CONTRACT_SHA256
    ):
        raise RuntimeError(
            f"{label} does not bind the active RF73 latent-mask contract/helper"
        )
    entries = payload.get("training")
    if not isinstance(entries, list) or not entries or any(
        not isinstance(entry, dict) for entry in entries
    ):
        raise RuntimeError(f"{label} training entries are missing, empty, or malformed")
    required = {
        "image",
        "label",
        "coverage_mask",
        "latent_loss_mask",
        "latent_loss_mask_nifti_file_sha256",
        "latent_loss_mask_content_sha256",
        "latent_loss_mask_affine_sha256",
        "latent_loss_mask_affine_float64_le",
        "latent_loss_mask_shape",
        "latent_loss_mask_eligible_voxels",
        "latent_loss_mask_algorithm_id",
        "latent_loss_mask_contract_sha256",
        "latent_loss_mask_helper_sha256",
        "latent_loss_mask_source_coverage_nifti_file_sha256",
        "latent_loss_mask_source_coverage_canonical_array_sha256",
        "sample_id",
        "examination_id",
        "spacing",
    }
    for index, entry in enumerate(entries):
        if not required <= set(entry):
            raise RuntimeError(f"{label} entry {index} is missing required fields")
        hashes = (
            "latent_loss_mask_nifti_file_sha256",
            "latent_loss_mask_content_sha256",
            "latent_loss_mask_affine_sha256",
            "latent_loss_mask_contract_sha256",
            "latent_loss_mask_helper_sha256",
            "latent_loss_mask_source_coverage_nifti_file_sha256",
            "latent_loss_mask_source_coverage_canonical_array_sha256",
        )
        if any(
            not isinstance(entry.get(field), str)
            or len(entry[field]) != 64
            or any(character not in "0123456789abcdef" for character in entry[field])
            for field in hashes
        ):
            raise RuntimeError(f"{label} entry {index} has an invalid SHA-256 field")
        if (
            entry["latent_loss_mask_algorithm_id"] != MASK_ALGORITHM_ID
            or entry["latent_loss_mask_contract_sha256"]
            != mask_contract_sha256()
            or entry["latent_loss_mask_helper_sha256"] != helper_hash
            or not isinstance(entry["latent_loss_mask_shape"], list)
            or len(entry["latent_loss_mask_shape"]) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in entry["latent_loss_mask_shape"]
            )
            or isinstance(entry["latent_loss_mask_eligible_voxels"], bool)
            or not isinstance(entry["latent_loss_mask_eligible_voxels"], int)
            or entry["latent_loss_mask_eligible_voxels"] <= 0
        ):
            raise RuntimeError(f"{label} entry {index} latent-mask contract drifted")
    sample_ids = [entry["sample_id"] for entry in entries]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError(f"{label} contains duplicate sample IDs")
    return entries


def _structured_importance_payload(
    *, accumulator: dict, named_parameters: dict, steps: int, provenance: dict
) -> dict:
    tensors = {
        name: (accumulator[name] / steps).detach().cpu().float().contiguous()
        for name in sorted(named_parameters)
    }
    metadata = [
        {
            "name": name,
            "shape": list(tensors[name].shape),
            "sum": float(tensors[name].double().sum().item()),
        }
        for name in sorted(tensors)
    ]
    unsigned = {
        "schema": IMPORTANCE_SCHEMA,
        "schema_version": IMPORTANCE_SCHEMA_VERSION,
        "status": "complete",
        "estimator": IMPORTANCE_ESTIMATOR,
        "likelihood_fisher_claim": False,
        "optimizer_steps": steps,
        "parameter_count": len(tensors),
        "tensor_inventory_sha256": tensor_mapping_sha256(tensors),
        "tensor_metadata_sha256": canonical_sha256(metadata),
        "provenance": provenance,
    }
    return {
        **unsigned,
        "metadata_artifact_sha256": canonical_sha256(unsigned),
        "tensors": tensors,
    }


def load_latent(path: Path, device: torch.device) -> torch.Tensor:
    """NIfTI channel-last (X, Y, Z, C) -> tensor (1, C, X, Y, Z)."""
    arr = nib.load(str(path)).get_fdata().astype(np.float32)
    return torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).to(device)


def load_latent_and_mask(
    latent_path: Path,
    mask_path: Path,
    *,
    entry: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one latent/mask pair and enforce exact mask provenance."""

    if sha256_file(mask_path) != entry["latent_loss_mask_nifti_file_sha256"]:
        raise RuntimeError(f"latent loss-mask file hash drifted: {mask_path}")
    latent_image = nib.load(str(latent_path))
    mask_image = nib.load(str(mask_path))
    if len(latent_image.shape) != 4 or len(mask_image.shape) != 3:
        raise RuntimeError("latent/mask dimensionality is invalid")
    if np.dtype(mask_image.get_data_dtype()) != np.dtype("u1"):
        raise RuntimeError("latent loss mask must have native uint8 storage")
    if (
        tuple(mask_image.shape) != tuple(latent_image.shape[:3])
        or entry["latent_loss_mask_shape"] != list(mask_image.shape)
        or not np.array_equal(mask_image.affine, latent_image.affine)
    ):
        raise RuntimeError("latent loss-mask shape/affine differs from its latent")
    try:
        exact_affine = decode_exact_affine_record(
            entry["latent_loss_mask_affine_float64_le"],
            expected_sha256=entry["latent_loss_mask_affine_sha256"],
        )
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    serialized_exact = np.asarray(exact_affine, dtype="<f4").astype(np.float64)
    if (
        not np.array_equal(
            np.asarray(mask_image.affine, dtype=np.float64), serialized_exact
        )
        or not np.all(np.isfinite(latent_image.affine))
    ):
        raise RuntimeError("latent loss-mask exact affine drifted")
    for image, name in ((latent_image, "latent"), (mask_image, "latent loss mask")):
        qform, qcode = image.get_qform(coded=True)
        sform, scode = image.get_sform(coded=True)
        if (
            int(qcode) != 1
            or int(scode) != 1
            or qform is None
            or sform is None
            or not np.allclose(qform, image.affine, rtol=0.0, atol=1e-5)
            or not np.array_equal(np.asarray(sform, dtype=np.float64), image.affine)
        ):
            raise RuntimeError(f"{name} qform/sform drifted")
    try:
        mask = validate_binary_mask(
            np.asanyarray(mask_image.dataobj),
            expected_shape=latent_image.shape[:3],
            require_nonempty=True,
        )
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    if (
        int(np.count_nonzero(mask))
        != entry["latent_loss_mask_eligible_voxels"]
        or latent_mask_content_sha256(mask, affine=exact_affine)
        != entry["latent_loss_mask_content_sha256"]
    ):
        raise RuntimeError("latent loss-mask content/count hash drifted")
    latent = np.ascontiguousarray(np.asanyarray(latent_image.dataobj), dtype=np.float32)
    if not np.all(np.isfinite(latent)):
        raise RuntimeError("latent contains non-finite values")
    latent_tensor = (
        torch.from_numpy(latent).permute(3, 0, 1, 2).unsqueeze(0).to(device)
    )
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
    return latent_tensor, mask_tensor


def load_training_entry(
    entry: dict, *, data_root: Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load latent, label, and mask atomically so batch lists cannot misalign."""

    coverage_path = data_root / entry["coverage_mask"]
    if (
        sha256_file(coverage_path)
        != entry["latent_loss_mask_source_coverage_nifti_file_sha256"]
    ):
        raise RuntimeError(f"source coverage file hash drifted: {coverage_path}")
    latent, mask = load_latent_and_mask(
        data_root / entry["image"],
        data_root / entry["latent_loss_mask"],
        entry=entry,
        device=device,
    )
    label = load_label(data_root / entry["label"], device)
    return latent, label, mask


def masked_output_l1(
    output: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Shared training, validation, LwF, and importance data objective."""

    return equal_examination_masked_l1(output.float() - reference.float(), mask)


def load_label(path: Path, device: torch.device) -> torch.Tensor:
    """NIfTI int volume -> tensor (1, 1, X, Y, Z)."""
    arr = nib.load(str(path)).get_fdata().astype(np.int64)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-config", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--network-config", required=True)
    ap.add_argument("--seed", type=int, default=1234,
                    help="Run seed. Paired experts must use the same seed to prove identical initialization.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Per-step batch (real batching, stacks N cases).")
    ap.add_argument("--grad-accumulate", type=int, default=1,
                    help="Gradient accumulation factor. Effective batch = batch_size * grad_accumulate.")
    ap.add_argument("--phase", choices=["1", "2", "mmwhs_b", "amos22_a"], default="1",
                    help="Dataset selector. All phases use 8-ch binarize_labels encoding.")
    ap.add_argument("--expert", choices=["A", "B", "oracle"], default="oracle",
                    help="Informational label only (no channel masking in binarize_labels mode).")
    ap.add_argument("--vis-every", type=int, default=0,
                    help="Save inference PNG every N epochs (0 = disabled).")
    ap.add_argument("--vis-case-idx", type=int, default=0,
                    help="Dataset index of fixed case used for vis inference.")
    ap.add_argument("--vis-steps", type=int, default=30,
                    help="Denoising steps for vis inference.")
    ap.add_argument("--cfg-dropout", type=float, default=0.1,
                    help="Classifier-free guidance dropout probability (0 = disabled).")
    ap.add_argument("--lr-schedule", choices=["poly2", "constant", "warmup-plateau"],
                    default="poly2",
                    help="poly2 = original PolynomialLR(power=2); constant = no decay; "
                         "warmup-plateau = linear warmup 5% steps then constant.")
    ap.add_argument("--warmup-frac", type=float, default=0.05,
                    help="Fraction of total steps used for warmup (warmup-plateau only).")
    ap.add_argument("--lr-override", type=float, default=None,
                    help="Override lr from config (e.g. 5e-5).")
    ap.add_argument("--epochs-override", type=int, default=None,
                    help="Override n_epochs from config.")
    ap.add_argument("--val-json", type=str, default=None,
                    help="JSON file with held-out validation entries (same format as training). "
                         "If provided, compute val_loss every --val-every epochs and save checkpoint_best.pt.")
    ap.add_argument("--val-every", type=int, default=1,
                    help="Run validation every N epochs (only when --val-json is set).")
    ap.add_argument("--val-n-samples", type=int, default=0,
                    help="Limit validation to first N entries (0 = use all). For speed during dev.")
    ap.add_argument("--strict-data-loading", action="store_true",
                    help="Abort on any train/validation sample load failure instead of silently skipping it. "
                         "Required for locked confirmatory experiments.")
    ap.add_argument("--n-bins", type=int, default=5,
                    help="Number of t bins for binned training+val loss tracking (0 = disabled). "
                         "For flow models, global mean loss blends signal across t; per-bin is informative.")
    ap.add_argument("--t-alpha", type=float, default=1.0,
                    help="Beta-distribution alpha for t sampling. alpha=beta=1 = uniform. "
                         "alpha=beta=2 = concentrate mid-t (Beta(2,2)). "
                         "alpha=beta=4 = stronger mid-t concentration.")
    ap.add_argument("--t-beta", type=float, default=1.0,
                    help="Beta-distribution beta for t sampling. See --t-alpha.")
    ap.add_argument("--z-jitter", type=int, default=0,
                    help="Maximum random Z-shift in latent voxels per training step (each = 4 label voxels). "
                         "0 = disabled. Typical: 2 or 3. Used to make experts FOV-shift robust for cross-dataset composition.")
    ap.add_argument("--cond-encoding", choices=["binarize", "onehot"], default="binarize",
                    help="Conditioning encoding. 'binarize' = native MAISI 8-bit (REQUIRED when loading "
                         "pretrained ControlNet). 'onehot' = 43-channel per-organ one-hot (only valid "
                         "when training from scratch; network config must set conditioning_embedding_in_channels=43).")
    ap.add_argument("--continual-method", choices=["none", "ewc", "lwf"], default="none",
                    help="Continual-learning anti-forgetting baseline: ewc (diagonal RF73-masked equal-examination L1-gradient importance) / lwf (output "
                         "distillation from the task-1 anchor model). Anchored to task 1 (preserves abd).")
    ap.add_argument("--continual-lambda", type=float, default=0.0, help="strength of the EWC/LwF penalty")
    ap.add_argument("--continual-anchor-ckpt", type=str, default=None,
                    help="task-1 anchor ControlNet ckpt (EWC old weights / LwF teacher)")
    ap.add_argument("--ewc-fisher", type=str, default=None,
                    help="Legacy option name: structured diagonal RF73-masked equal-examination L1-gradient importance artifact.")
    ap.add_argument("--compute-fisher", type=str, default=None,
                    help="Legacy option name: compute diagonal RF73-masked equal-examination L1-gradient importance.")
    ap.add_argument("--fisher-steps", type=int, default=210,
                    help="Exact optimizer steps for diagonal RF73-masked equal-examination L1-gradient importance.")
    ap.add_argument("--locked-confirmatory", action="store_true",
                    help="Enable the fail-closed preflight, exact-budget, and signed-receipt protocol.")
    ap.add_argument("--run-contract-file", type=str, default=None)
    ap.add_argument("--run-contract-sha256", type=str, default=None)
    ap.add_argument("--validation-draw-contract", type=str, default=None)
    ap.add_argument("--expected-terminal-epoch", type=int, default=None)
    ap.add_argument("--expected-optimizer-steps", type=int, default=None)
    ap.add_argument("--expected-sample-exposures", type=int, default=None)
    ap.add_argument("--expected-validation-evaluations", type=int, default=None)
    args = ap.parse_args()

    locked_sidecar = None
    locked_sidecar_file_sha256 = None
    if args.locked_confirmatory:
        if args.resume:
            raise RuntimeError("locked confirmatory runs never resume or reuse stale output")
        required_locked = {
            "run_contract_file": args.run_contract_file,
            "run_contract_sha256": args.run_contract_sha256,
            "validation_draw_contract": args.validation_draw_contract,
            "expected_terminal_epoch": args.expected_terminal_epoch,
            "expected_optimizer_steps": args.expected_optimizer_steps,
            "expected_sample_exposures": args.expected_sample_exposures,
            "expected_validation_evaluations": args.expected_validation_evaluations,
        }
        missing = [name for name, value in required_locked.items() if value is None]
        if missing:
            raise RuntimeError(f"locked confirmatory arguments are incomplete: {missing}")
        if not args.strict_data_loading:
            raise RuntimeError("locked confirmatory runs require --strict-data-loading")
        locked_sidecar = _validate_locked_run_sidecar(
            Path(args.run_contract_file),
            expected_hash=args.run_contract_sha256,
            actual_cli_args=sys.argv[1:],
        )
        locked_sidecar_file_sha256 = sha256_file(Path(args.run_contract_file))
        _locked_production_preflight(locked_sidecar["run_contract"])

    # Seed before constructing either network: the conditioning branch contains
    # randomly initialized parameters that are not copied from the shared U-Net.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    log(f"reproducibility seed={args.seed} (cuDNN deterministic, benchmark off)")

    if args.cond_encoding == "onehot":
        def make_cond(lbl):
            return _to_one_hot(lbl.long())
        log(f"phase={args.phase}  expert={args.expert}  conditioning=one-hot({N_COND_CHANNELS}-ch)")
    else:
        def make_cond(lbl):
            return binarize_labels(lbl.long()).float()
        log(f"phase={args.phase}  expert={args.expert}  conditioning=binarize_labels(8-bit)")

    cfg = load_config(args.env_config, args.model_config, args.network_config)

    device = torch.device("cuda:0")
    data_root = Path(cfg.data_base_dir[0]) if isinstance(cfg.data_base_dir, list) else Path(cfg.data_base_dir)
    json_path = cfg.json_data_list[0] if isinstance(cfg.json_data_list, list) else cfg.json_data_list
    if args.locked_confirmatory and data_root.resolve() != REPO_ROOT.resolve():
        raise RuntimeError("locked data_base_dir must be the repository root")
    entries = _strict_manifest_entries(Path(json_path), "training manifest")
    log(f"loaded {len(entries)} entries from {json_path}")

    run_fingerprints = {
        "env_config_sha256": sha256_file(Path(args.env_config)),
        "model_config_sha256": sha256_file(Path(args.model_config)),
        "network_config_sha256": sha256_file(Path(args.network_config)),
        "train_manifest_sha256": sha256_file(Path(json_path)),
        "trainer_sha256": sha256_file(Path(__file__)),
        "latent_mask_helper_sha256": sha256_file(
            REPO_ROOT / "src/fov_latent_mask.py"
        ),
        "latent_mask_contract_sha256": mask_contract_sha256(),
        "masked_loss_contract_sha256": MASKED_LOSS_CONTRACT_SHA256,
    }
    if args.locked_confirmatory:
        run_fingerprints["run_contract_sha256"] = args.run_contract_sha256
        run_fingerprints["run_contract_sidecar_sha256"] = (
            locked_sidecar_file_sha256
        )

    # --- Load validation entries (held-out, never in training)
    val_entries = []
    validation_draws = {}
    validation_draw_contract_sha256 = None
    if args.val_json:
        val_path = Path(args.val_json)
        if not val_path.is_file():
            if args.locked_confirmatory or args.strict_data_loading:
                raise RuntimeError(f"validation manifest is missing: {val_path}")
            log(f"WARN: --val-json {args.val_json} not found; validation disabled")
        else:
            val_entries = _strict_manifest_entries(val_path, "validation manifest")
            run_fingerprints["val_manifest_sha256"] = sha256_file(val_path)
            if args.val_n_samples > 0:
                if args.locked_confirmatory:
                    raise RuntimeError("locked validation may not be truncated")
                val_entries = val_entries[:args.val_n_samples]
            log(f"loaded {len(val_entries)} validation entries from {val_path}")
    if args.locked_confirmatory:
        if not args.val_json or not val_entries:
            raise RuntimeError("locked validation manifest is required and non-empty")
        draw_path = Path(args.validation_draw_contract)
        draw_payload = _load_json_object(draw_path, "validation draw contract")
        validation_draws = validate_validation_draw_contract(
            draw_payload,
            expected_sample_ids=[entry["sample_id"] for entry in val_entries],
        )
        validation_draw_contract_sha256 = sha256_file(draw_path)
        run_fingerprints["validation_draw_contract_sha256"] = (
            validation_draw_contract_sha256
        )
        if args.val_every != 1 or args.val_n_samples != 0:
            raise RuntimeError("locked validation requires --val-every 1 and all cases")
    error_event_counts = {
        "training_load_errors": 0,
        "validation_load_errors": 0,
        "nonfinite_losses": 0,
        "nonfinite_gradients": 0,
    }
    if args.locked_confirmatory:
        locked_run = locked_sidecar["run_contract"]
        if (
            locked_run.get("masked_loss_contract") != MASKED_LOSS_CONTRACT
            or locked_run.get("masked_loss_contract_sha256")
            != MASKED_LOSS_CONTRACT_SHA256
        ):
            raise RuntimeError("run contract lacks the active RF73 masked objective")
        if Path(str(locked_run.get("train_manifest"))).resolve() != Path(
            json_path
        ).resolve():
            raise RuntimeError("environment config selects the wrong training manifest")
        if Path(str(locked_run.get("validation_manifest"))).resolve() != Path(
            args.val_json
        ).resolve():
            raise RuntimeError("actual validation manifest differs from the run contract")
        if Path(str(locked_run.get("output_dir"))) != Path(cfg.output_dir):
            raise RuntimeError("environment output directory differs from the run contract")
        if locked_run.get("production_layout") is True:
            output_path = Path(cfg.output_dir).resolve(strict=False)
            nas_namespace = Path(
                str(locked_run.get("shared_nas_namespace"))
            ).resolve(strict=False)
            checkpoint_root = Path(
                str(locked_run.get("checkpoint_storage_root"))
            ).resolve(strict=False)
            try:
                output_path.relative_to(nas_namespace)
                output_path.relative_to(checkpoint_root)
            except ValueError as error:
                raise RuntimeError(
                    "production checkpoints must remain under the bound NAS root"
                ) from error
        test_gate = locked_run.get("test_release_gate")
        if not isinstance(test_gate, dict) or test_gate.get("must_be_absent") is not True:
            raise RuntimeError("locked run lacks a closed physical test-release gate")
        if test_gate.get("continual_success_contract_sha256") != locked_run.get(
            "continual_success_contract_sha256"
        ):
            raise RuntimeError("test-release gate does not bind the continual success contract")
        required_downstream = locked_run.get("required_downstream_method_outputs")
        if set(required_downstream or []) != {
            "old_only",
            "new_only",
            "hard_concat",
            "raised_cosine_ct_blend",
            "sync_overlap_add_lambda0",
            "guided_poe_lambda1",
            "seq_ft",
            "ewc",
            "pooled_slabs",
            "full_fov_reference",
        }:
            raise RuntimeError(
                "run contract lacks exact downstream success-method outputs"
            )
        required_contrasts = locked_run.get(
            "required_downstream_contrast_outputs"
        )
        if set(required_contrasts or []) != {
            "system_retention_vs_old_only",
            "system_acquisition_vs_new_only",
            "integration_vs_synchronized_overlap_add",
            "integration_vs_posthoc_ct_blend",
            "seam_vs_synchronized_overlap_add",
            "seam_vs_posthoc_ct_blend",
            "continuity_vs_synchronized_overlap_add",
            "continuity_vs_posthoc_ct_blend",
        }:
            raise RuntimeError(
                "run contract lacks exact downstream success contrasts"
            )
        if os.path.lexists(Path(str(test_gate.get("path")))):
            raise RuntimeError("physical test-release gate opened before training")

        planned_files = locked_run.get("input_files")
        planned_fingerprints = locked_run.get("input_fingerprints")
        if (
            not isinstance(planned_files, dict)
            or not isinstance(planned_fingerprints, dict)
            or set(planned_files) != set(planned_fingerprints)
        ):
            raise RuntimeError("run contract lacks a complete input-file inventory")
        for name, record in planned_files.items():
            if not isinstance(record, dict):
                raise RuntimeError(f"malformed run input record at {name}")
            path = Path(str(record.get("path")))
            expected = record.get("sha256")
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"runtime input file drifted at {name}")
            if name in run_fingerprints and run_fingerprints[name] != expected:
                raise RuntimeError(f"runtime input fingerprint disagrees at {name}")
            run_fingerprints[name] = expected

        dependencies = locked_run.get("dependencies")
        if not isinstance(dependencies, list):
            raise RuntimeError("run dependency inventory is malformed")
        for dependency in dependencies:
            if not isinstance(dependency, dict):
                raise RuntimeError("malformed run dependency")
            receipt_path = Path(str(dependency.get("receipt_path")))
            if not receipt_path.is_file():
                raise RuntimeError(
                    f"dependency lacks a signed completion receipt: {receipt_path}"
                )
            receipt_payload = _load_json_object(
                receipt_path, "dependency completion receipt"
            )
            try:
                receipt = validate_completion_receipt(
                    receipt_payload,
                    expected_run_contract_sha256=str(
                        dependency.get("run_contract_sha256")
                    ),
                )
            except TrainingProofError as error:
                raise RuntimeError(str(error)) from error
            checkpoint_path = Path(str(dependency.get("checkpoint_path")))
            if (
                dependency.get("checkpoint_role") != "best"
                or Path(str(receipt.get("best_checkpoint_path"))) != checkpoint_path
                or not checkpoint_path.is_file()
                or sha256_file(checkpoint_path)
                != receipt.get("best_checkpoint_sha256")
            ):
                raise RuntimeError("dependency best checkpoint is not receipt-bound")
            dependency_key = str(dependency.get("run_id"))
            run_fingerprints[f"dependency_{dependency_key}_receipt_sha256"] = (
                sha256_file(receipt_path)
            )
            run_fingerprints[f"dependency_{dependency_key}_best_sha256"] = (
                receipt["best_checkpoint_sha256"]
            )
            if "importance_artifact_path" in dependency:
                importance_path = Path(str(dependency["importance_artifact_path"]))
                importance_record = receipt.get("importance_artifact")
                if (
                    not isinstance(importance_record, dict)
                    or Path(str(importance_record.get("path"))) != importance_path
                    or not importance_path.is_file()
                    or sha256_file(importance_path) != importance_record.get("sha256")
                ):
                    raise RuntimeError(
                        "dependency importance artifact is not receipt-bound"
                    )
                run_fingerprints[
                    f"dependency_{dependency_key}_importance_sha256"
                ] = importance_record["sha256"]

        if not isinstance(planned_fingerprints, dict):
            raise RuntimeError("run contract lacks input fingerprints")
        for name, expected in planned_fingerprints.items():
            if run_fingerprints.get(name) != expected:
                raise RuntimeError(f"runtime input fingerprint drifted at {name}")

    # --- Build models
    log("building UNet ...")
    unet = define_instance(cfg, "diffusion_unet_def").to(device)
    sd = torch.load(cfg.trained_diffusion_path, map_location=device, weights_only=False)
    unet.load_state_dict(sd["unet_state_dict"], strict=False)
    scale_factor = sd["scale_factor"].to(device) if hasattr(sd["scale_factor"], "to") else torch.tensor(sd["scale_factor"]).to(device)
    unet.eval()
    for p in unet.parameters():
        p.requires_grad_(False)
    log(f"UNet loaded; scale_factor={scale_factor.item():.4f}; GPU {torch.cuda.memory_allocated()/1e9:.2f}GB")

    log("building ControlNet ...")
    controlnet = define_instance(cfg, "controlnet_def").to(device)
    copy_model_state(controlnet, unet.state_dict())
    if (
        args.locked_confirmatory
        and cfg.existing_ckpt_filepath
        and not Path(cfg.existing_ckpt_filepath).is_file()
    ):
        raise RuntimeError(
            f"locked warm-start checkpoint is missing: {cfg.existing_ckpt_filepath}"
        )
    if cfg.existing_ckpt_filepath and Path(cfg.existing_ckpt_filepath).exists():
        ctrl_sd = torch.load(cfg.existing_ckpt_filepath, map_location=device, weights_only=False)
        controlnet.load_state_dict(ctrl_sd["controlnet_state_dict"], strict=True)
        log(f"ControlNet loaded from {cfg.existing_ckpt_filepath}")
    else:
        log("ControlNet trained from scratch (no pretrained ckpt)")
    initialization_sha256 = state_dict_sha256(controlnet.state_dict())
    run_fingerprints["controlnet_initialization_sha256"] = initialization_sha256
    named_parameter_contract = [
        {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
        for name, parameter in controlnet.named_parameters()
    ]
    run_fingerprints["named_parameter_contract_sha256"] = canonical_sha256(
        named_parameter_contract
    )
    if cfg.existing_ckpt_filepath and Path(cfg.existing_ckpt_filepath).exists():
        run_fingerprints["existing_controlnet_sha256"] = sha256_file(Path(cfg.existing_ckpt_filepath))
    log(f"ControlNet initialization SHA256={initialization_sha256}")
    controlnet.train()
    log(f"ControlNet ready; GPU {torch.cuda.memory_allocated()/1e9:.2f}GB")

    def resolved_importance_provenance(template: dict) -> dict:
        """Resolve runtime-only hashes without weakening the signed template."""

        if not isinstance(template, dict):
            raise RuntimeError("importance provenance template must be an object")
        anchor_value = args.continual_anchor_ckpt or cfg.existing_ckpt_filepath
        if not anchor_value or not Path(anchor_value).is_file():
            raise RuntimeError(
                "importance provenance requires the exact warm-start/anchor checkpoint"
            )
        replacements = {
            "<CURRENT_RUN_CONTRACT_SHA256>": args.run_contract_sha256,
            "<ANCHOR_CHECKPOINT_SHA256>": sha256_file(Path(anchor_value)),
            "<NAMED_PARAMETER_CONTRACT_SHA256>": run_fingerprints[
                "named_parameter_contract_sha256"
            ],
        }
        resolved = {
            key: replacements.get(value, value) for key, value in template.items()
        }
        resolved["anchor_checkpoint_sha256"] = replacements[
            "<ANCHOR_CHECKPOINT_SHA256>"
        ]
        resolved["named_parameter_contract_sha256"] = replacements[
            "<NAMED_PARAMETER_CONTRACT_SHA256>"
        ]
        return resolved

    # --- Continual-learning anti-forgetting baselines (anchored to task 1) ---
    anchor_controlnet = None
    ewc_fisher = None
    ewc_old = None
    if args.continual_method == "lwf":
        anchor_controlnet = define_instance(cfg, "controlnet_def").to(device)
        a_sd = torch.load(args.continual_anchor_ckpt, map_location=device, weights_only=False)
        anchor_controlnet.load_state_dict(a_sd["controlnet_state_dict"], strict=True)
        anchor_controlnet.eval()
        for p in anchor_controlnet.parameters():
            p.requires_grad_(False)
        log(f"LwF: teacher (anchor) ControlNet loaded from {args.continual_anchor_ckpt}, lambda={args.continual_lambda}")
    elif args.continual_method == "ewc":
        if not args.continual_anchor_ckpt or not Path(args.continual_anchor_ckpt).is_file():
            raise RuntimeError("EWC anchor checkpoint is missing")
        if not args.ewc_fisher or not Path(args.ewc_fisher).is_file():
            raise RuntimeError("EWC importance artifact is missing")
        a_sd = torch.load(args.continual_anchor_ckpt, map_location=device, weights_only=False)["controlnet_state_dict"]
        named_parameters = dict(controlnet.named_parameters())
        if set(a_sd) != set(controlnet.state_dict()):
            raise RuntimeError("EWC anchor state keys do not exactly match ControlNet")
        ewc_old = {n: a_sd[n].to(device).float() for n in named_parameters}
        importance_payload = torch.load(
            args.ewc_fisher, map_location="cpu", weights_only=False
        )
        expected_importance_provenance = None
        if args.locked_confirmatory:
            expected_importance_provenance = locked_sidecar.get(
                "run_contract", {}
            ).get("importance_dependency")
            if not isinstance(expected_importance_provenance, dict):
                raise RuntimeError("EWC run contract lacks importance dependency")
            expected_importance_provenance = resolved_importance_provenance(
                expected_importance_provenance
            )
        validate_importance_payload(
            importance_payload,
            named_parameters=named_parameters,
            expected_provenance=expected_importance_provenance,
        )
        ewc_fisher = {
            n: importance_payload["tensors"][n].to(device).float()
            for n in named_parameters
        }
        log(
            "EWC: anchor weights + diagonal RF73-masked equal-examination L1-gradient importance "
            f"loaded ({len(ewc_fisher)} tensors), lambda={args.continual_lambda}"
        )
    main._fisher_acc = {
        name: torch.zeros_like(parameter, dtype=torch.float32, device=device)
        for name, parameter in controlnet.named_parameters()
    } if args.compute_fisher else {}
    main._fisher_count = 0

    noise_scheduler = define_instance(cfg, "noise_scheduler")

    # --- Optimizer
    n_epochs = args.epochs_override if args.epochs_override else cfg.controlnet_train["n_epochs"]
    lr = args.lr_override if args.lr_override else cfg.controlnet_train["lr"]
    _bsz_for_lr = max(1, int(args.batch_size))
    _gacc_for_lr = max(1, int(args.grad_accumulate))
    _microbatches_per_epoch = (len(entries) + _bsz_for_lr - 1) // _bsz_for_lr
    total_steps = max(1, n_epochs * ((_microbatches_per_epoch + _gacc_for_lr - 1) // _gacc_for_lr))
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=lr)
    if args.lr_schedule == "poly2":
        lr_sched = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)
    elif args.lr_schedule == "constant":
        lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    elif args.lr_schedule == "warmup-plateau":
        n_warm = max(1, int(args.warmup_frac * total_steps))
        def _wp(step):
            if step < n_warm:
                return (step + 1) / n_warm
            return 1.0
        lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_wp)
    else:
        raise ValueError(args.lr_schedule)
    log(f"lr_schedule={args.lr_schedule}  warmup_frac={args.warmup_frac}  total_steps={total_steps}")
    scaler = GradScaler("cuda")
    log(f"optimizer ready: lr={lr}, n_epochs={n_epochs}, eff_batch={_bsz_for_lr * _gacc_for_lr}, total_opt_steps={total_steps}")
    if args.compute_fisher:  # Importance pass must not change weights: lr=0 makes scaler.step a no-op
        for _g in optimizer.param_groups:
            _g["lr"] = 0.0
        log("importance mode: optimizer lr forced to 0 (weights frozen)")

    # Local shuffle RNG is checkpointed separately from the process-global RNGs.
    rng = random.Random(args.seed)

    # --- Resume
    output_dir = Path(cfg.output_dir)
    if args.locked_confirmatory and output_dir.exists():
        raise RuntimeError(
            f"locked output directory must not exist before launch: {output_dir}"
        )
    if args.locked_confirmatory:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            output_dir.mkdir(exist_ok=False)
        except FileExistsError as error:
            raise RuntimeError(
                f"locked output directory appeared during launch: {output_dir}"
            ) from error
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    resumed_step = 0
    if args.resume:
        ck_path = output_dir / "checkpoint_latest.pt"
        if ck_path.exists():
            ck = torch.load(ck_path, map_location=device, weights_only=False)
            validate_resume_checkpoint_bindings(
                ck, run_fingerprints=run_fingerprints
            )
            controlnet.load_state_dict(ck["controlnet_state_dict"])
            optimizer.load_state_dict(ck["optimizer_state_dict"])
            start_epoch = ck["epoch"] + 1
            resumed_step = ck.get("step", 0)
            if "python_rng_state" in ck:
                random.setstate(ck["python_rng_state"])
            if "shuffle_rng_state" in ck:
                rng.setstate(ck["shuffle_rng_state"])
            if "numpy_rng_state" in ck:
                np.random.set_state(ck["numpy_rng_state"])
            if "torch_rng_state" in ck:
                torch.set_rng_state(ck["torch_rng_state"])
            if torch.cuda.is_available() and "cuda_rng_state_all" in ck:
                torch.cuda.set_rng_state_all(ck["cuda_rng_state_all"])
            # Re-init LR scheduler so extending n_epochs works correctly, but keep
            # the saved scheduler epoch instead of double-decaying from resumed LR.
            # (Honor --lr-schedule for the fresh scheduler.)
            if args.lr_schedule == "poly2":
                lr_sched = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)
            elif args.lr_schedule == "constant":
                lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
            elif args.lr_schedule == "warmup-plateau":
                n_warm = max(1, int(args.warmup_frac * total_steps))
                def _wp_resume(step):
                    if step < n_warm:
                        return (step + 1) / n_warm
                    return 1.0
                lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_wp_resume)
            sched_state = ck.get("lr_scheduler_state_dict")
            if sched_state is not None:
                lr_sched.load_state_dict(sched_state)
                if getattr(lr_sched, "total_iters", total_steps) != total_steps:
                    lr_sched.total_iters = total_steps
            else:
                for _ in range(resumed_step):
                    lr_sched.step()
            log(f"resumed from {ck_path} at epoch {start_epoch} (step {resumed_step})")

    # --- Vis setup (periodic inference PNG)
    ae_vis = None
    vis_cond = vis_spacing = vis_modality = vis_dir = None
    if args.vis_every > 0:
        log("Loading AutoEncoder for vis (CPU, norm_float16=False) ...")
        ae_vis = define_instance(cfg, "autoencoder_def")  # keep on CPU
        ae_sd_vis = torch.load(cfg.trained_autoencoder_path, map_location="cpu", weights_only=False)
        if isinstance(ae_sd_vis, dict) and "unet_state_dict" in ae_sd_vis:
            ae_sd_vis = ae_sd_vis["unet_state_dict"]
        ae_vis.load_state_dict(ae_sd_vis)
        ae_vis.eval()
        for p in ae_vis.parameters():
            p.requires_grad_(False)
        for m in ae_vis.modules():
            if hasattr(m, "norm_float16"):
                m.norm_float16 = False
        log("AutoEncoder vis ready (CPU)")

        vis_idx = min(args.vis_case_idx, len(entries) - 1)
        vis_entry = entries[vis_idx]
        vis_lbl = load_label(data_root / vis_entry["label"], device)
        vis_cond = make_cond(vis_lbl)
        vis_spacing = torch.tensor([vis_entry["spacing"]], device=device, dtype=torch.float32) * 100.0
        vis_modality = torch.ones((1,), device=device, dtype=torch.long)
        vis_dir = output_dir / "vis"
        vis_dir.mkdir(exist_ok=True)
        log(f"vis case idx={vis_idx}: {vis_entry['label']}")

    # --- Train loop
    indices = list(range(len(entries)))
    t_start = time.time()
    step = resumed_step if args.resume else 0

    # Bin tracking for flow models — t-binned loss is more informative than global mean.
    NB = max(0, int(args.n_bins))
    T_max = noise_scheduler.num_train_timesteps
    bin_edges = np.linspace(0, 1, NB + 1) if NB > 0 else None

    # Auto-stop watchdog
    NAN_WINDOW = 100
    DIVERGE_THRESHOLD = 5.0
    SKIP_LIMIT = 50
    consecutive_skip = 0
    BSZ = max(1, int(args.batch_size))
    GACC = max(1, int(args.grad_accumulate))
    eff_bsz = BSZ * GACC
    log(f"batch_size={BSZ}  grad_accumulate={GACC}  effective_batch={eff_bsz}  cfg_dropout={args.cfg_dropout}")

    recent_losses = []
    sample_exposures = 0
    validation_evaluations = 0
    importance_complete = False
    for epoch in range(start_epoch, n_epochs):
        rng.shuffle(indices)
        epoch_losses = []
        # Per-bin accumulators for training loss
        bin_loss_sum = [0.0] * NB if NB > 0 else None
        bin_loss_count = [0] * NB if NB > 0 else None
        t_epoch = time.time()

        # Iterate batches of size BSZ; group GACC such batches into one optimizer step
        n = len(indices)
        i = 0
        accum_count = 0
        accum_target = GACC
        accum_examination_target = BSZ * GACC
        optimizer.zero_grad(set_to_none=True)

        while i < n:
            batch_start = i
            batch_indices = indices[i:i + BSZ]
            i += len(batch_indices)
            if accum_count == 0:
                remaining_microbatches = (n - batch_start + BSZ - 1) // BSZ
                accum_target = min(GACC, remaining_microbatches)
                accum_examination_target = min(
                    n - batch_start, accum_target * BSZ
                )

            # Load and stack batch
            img_list, lbl_list, mask_list, spacing_list, valid_entries = (
                [],
                [],
                [],
                [],
                [],
            )
            for bi in batch_indices:
                entry = entries[bi]
                try:
                    image, label, loss_mask = load_training_entry(
                        entry, data_root=data_root, device=device
                    )
                    img_list.append(image)
                    lbl_list.append(label)
                    mask_list.append(loss_mask)
                    spacing_list.append(entry["spacing"])
                    valid_entries.append(entry)
                except Exception as e:
                    error_event_counts["training_load_errors"] += 1
                    if args.strict_data_loading:
                        raise RuntimeError(
                            f"strict data loading failed for training entry {entry.get('image')}"
                        ) from e
                    print(f"  [skip {entry['image']}] {type(e).__name__}: {e}", flush=True)
            if not img_list:
                continue

            img_t = torch.cat(img_list, dim=0)         # (b, 4, D, H, W) — latent
            lbl_t = torch.cat(lbl_list, dim=0)         # (b, 1, D, H, W) — full-res label
            mask_t = torch.cat(mask_list, dim=0)       # (b, 1, D, H, W) — latent RF73
            b = img_t.shape[0]
            microbatch_weight = float(b) / float(accum_examination_target)
            sample_exposures += int(b)

            # Z-jitter augmentation: zero-fill latent, label, and mask on true Z.
            # Latent Z dim is 1/4 of label Z dim, so latent shift k → label shift k*4.
            if args.z_jitter > 0:
                z_shift_lat = int(np.random.randint(-args.z_jitter, args.z_jitter + 1))
                if z_shift_lat != 0:
                    img_t = zero_fill_shift(img_t, z_shift_lat, dim=-1)
                    lbl_t = zero_fill_shift(lbl_t, z_shift_lat * 4, dim=-1)
                    mask_t = zero_fill_shift(mask_t, z_shift_lat, dim=-1)
                    if torch.any(mask_t.sum(dim=(1, 2, 3, 4)) <= 0):
                        raise RuntimeError(
                            "Z-jitter removed every eligible latent voxel"
                        )

            images_scaled = img_t * scale_factor
            noise = torch.randn_like(images_scaled)
            # t sampling: beta(alpha, beta). alpha=beta=1 == uniform (scheduler default).
            if args.t_alpha == 1.0 and args.t_beta == 1.0:
                if isinstance(noise_scheduler, RFlowScheduler):
                    timesteps = noise_scheduler.sample_timesteps(images_scaled)
                else:
                    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (b,), device=device).long()
            else:
                t_frac = np.random.beta(args.t_alpha, args.t_beta, size=b).astype(np.float32)
                if isinstance(noise_scheduler, RFlowScheduler):
                    timesteps = torch.from_numpy(t_frac * float(T_max)).to(device).float()
                else:
                    timesteps = torch.from_numpy((t_frac * float(T_max)).astype(np.int64)).to(device).long()
            noisy_latent = noise_scheduler.add_noise(original_samples=images_scaled, noise=noise, timesteps=timesteps)

            controlnet_cond = make_cond(lbl_t)
            if args.cfg_dropout > 0:
                drop_mask = torch.rand(b, device=device) < args.cfg_dropout
                controlnet_cond[drop_mask] = 0.0
            spacing_tensor = torch.tensor(spacing_list, device=device, dtype=torch.float32) * 100.0  # (b, 3)
            modality_tensor = torch.ones((b,), device=device, dtype=torch.long)                     # (b,)

            with autocast("cuda"):
                down, mid = controlnet(
                    x=noisy_latent, timesteps=timesteps,
                    controlnet_cond=controlnet_cond, class_labels=modality_tensor,
                )
                model_output = unet(
                    x=noisy_latent, timesteps=timesteps, spacing_tensor=spacing_tensor,
                    down_block_additional_residuals=down,
                    mid_block_additional_residual=mid,
                    class_labels=modality_tensor,
                )
                if isinstance(noise_scheduler, RFlowScheduler):
                    target = images_scaled - noise
                else:
                    target = noise
                # Weight by examination count so accumulated gradients are the
                # exact examination mean, including a short final microbatch.
                loss = microbatch_weight * masked_output_l1(
                    model_output, target, mask_t
                )

                # --- Continual anti-forgetting penalty (anchored to task 1) ---
                if args.continual_method == "lwf" and anchor_controlnet is not None:
                    with torch.no_grad():
                        a_down, a_mid = anchor_controlnet(
                            x=noisy_latent, timesteps=timesteps,
                            controlnet_cond=controlnet_cond, class_labels=modality_tensor)
                        anchor_output = unet(
                            x=noisy_latent, timesteps=timesteps, spacing_tensor=spacing_tensor,
                            down_block_additional_residuals=a_down,
                            mid_block_additional_residual=a_mid, class_labels=modality_tensor)
                    loss = loss + args.continual_lambda * microbatch_weight * (
                        masked_output_l1(model_output, anchor_output, mask_t)
                    )
                elif args.continual_method == "ewc" and ewc_fisher is not None:
                    pen = sum((ewc_fisher[n] * (p.float() - ewc_old[n]) ** 2).sum()
                              for n, p in controlnet.named_parameters() if n in ewc_fisher)
                    loss = loss + args.continual_lambda * microbatch_weight * pen

            loss_val = float(loss.detach().cpu()) / microbatch_weight
            # Per-bin training loss tracking
            if NB > 0 and bin_loss_sum is not None:
                ts_cpu = timesteps.detach().float().cpu().numpy() / float(T_max)
                for tf in ts_cpu:
                    bi = min(NB - 1, max(0, int(tf * NB)))
                    bin_loss_sum[bi] += loss_val
                    bin_loss_count[bi] += 1
            if not (loss_val == loss_val) or loss_val == float("inf") or loss_val == float("-inf"):
                error_event_counts["nonfinite_losses"] += 1
                if args.locked_confirmatory:
                    raise RuntimeError("locked run encountered its first nonfinite loss")
                print(f"  [skip batch] loss is NaN/Inf -- entries: {[e['image'] for e in valid_entries]}", flush=True)
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                del loss
                consecutive_skip += 1
                if consecutive_skip >= SKIP_LIMIT:
                    log(f"ABORT: {consecutive_skip} consecutive batches skipped")
                    sys.exit(2)
                continue

            scaler.scale(loss).backward()
            accum_count += 1

            if accum_count >= accum_target:
                scaler.unscale_(optimizer)
                if args.compute_fisher:  # accumulate squared masked-L1 gradients
                    # Only accumulate when this step's grads are all finite. Under fp16 AMP the
                    # scaler must calibrate its scale down; we let it via a harmless lr=0 step.
                    finite = all(torch.isfinite(_p.grad).all() for _, _p in controlnet.named_parameters()
                                 if _p.grad is not None)
                    if not finite:
                        error_event_counts["nonfinite_gradients"] += 1
                        if args.locked_confirmatory:
                            raise RuntimeError(
                                "locked importance run encountered its first nonfinite gradient"
                            )
                    if finite:
                        for _n, _p in controlnet.named_parameters():
                            if _p.grad is not None:
                                main._fisher_acc[_n].add_(_p.grad.detach().float() ** 2)
                        main._fisher_count += 1
                    scaler.step(optimizer)   # lr==0 -> no weight change; lets scaler detect inf
                    scaler.update()          # decays scale on overflow until grads are finite
                    optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    step += 1
                    if main._fisher_count % 25 == 0 and main._fisher_count > 0:
                        log(f"importance {main._fisher_count}/{args.fisher_steps} (scale={scaler.get_scale():.0f})")
                    if main._fisher_count >= args.fisher_steps:
                        provenance = (
                            locked_sidecar.get("run_contract", {}).get(
                                "importance_output_provenance"
                            )
                            if args.locked_confirmatory
                            else {
                                "run_contract_sha256": args.run_contract_sha256,
                                **run_fingerprints,
                            }
                        )
                        if not isinstance(provenance, dict):
                            raise RuntimeError(
                                "importance run lacks structured output provenance"
                            )
                        if args.locked_confirmatory:
                            provenance = resolved_importance_provenance(provenance)
                        importance_payload = _structured_importance_payload(
                            accumulator=main._fisher_acc,
                            named_parameters=dict(controlnet.named_parameters()),
                            steps=main._fisher_count,
                            provenance=provenance,
                        )
                        _atomic_torch_save(importance_payload, Path(args.compute_fisher))
                        importance_complete = True
                        log(
                            "diagonal RF73-masked equal-examination L1-gradient importance saved to "
                            f"{args.compute_fisher} ({main._fisher_count} steps)"
                        )
                        break
                    continue
                grad_norm = torch.nn.utils.clip_grad_norm_(controlnet.parameters(), max_norm=1.0)
                if not torch.isfinite(grad_norm):
                    error_event_counts["nonfinite_gradients"] += 1
                    if args.locked_confirmatory:
                        raise RuntimeError("locked run encountered its first nonfinite gradient")
                    print("  [skip step] grad_norm is NaN/Inf", flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    accum_count = 0
                    consecutive_skip += 1
                    if consecutive_skip >= SKIP_LIMIT:
                        log(f"ABORT: {consecutive_skip} consecutive grad-skips")
                        sys.exit(2)
                    continue
                scaler.step(optimizer)
                scaler.update()
                lr_sched.step()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                step += 1
                consecutive_skip = 0
            epoch_losses.append(loss_val)
            recent_losses.append(loss_val)
            if len(recent_losses) > NAN_WINDOW:
                recent_losses.pop(0)
            if len(recent_losses) >= NAN_WINDOW:
                window_mean = sum(recent_losses) / len(recent_losses)
                if window_mean > DIVERGE_THRESHOLD:
                    log(f"ABORT: mean loss over last {NAN_WINDOW} steps = {window_mean:.3f} > {DIVERGE_THRESHOLD}")
                    sys.exit(3)

            # Free per-case tensors
            del img_t, lbl_t, controlnet_cond, noisy_latent, noise, model_output, down, mid

            if step > 0 and step % 5 == 0 and accum_count == 0:
                elapsed = time.time() - t_start
                rate = step / max(elapsed, 1e-6)
                mem = torch.cuda.memory_allocated() / 1e9
                log(f"epoch {epoch}/{n_epochs} step {step}/{total_steps} loss {loss_val:.4f} "
                    f"lr {lr_sched.get_last_lr()[0]:.2e} rate {rate:.2f} it/s GPU {mem:.2f}GB")

        if importance_complete and i < n and args.locked_confirmatory:
            raise RuntimeError(
                "importance step budget ended before the exact terminal dataset boundary"
            )
        epoch_mean = np.mean(epoch_losses) if epoch_losses else float("nan")
        if args.locked_confirmatory and not math.isfinite(float(epoch_mean)):
            raise RuntimeError("locked epoch has no finite complete training loss")
        log(f"=== epoch {epoch} done: mean_loss {epoch_mean:.4f} time {time.time()-t_epoch:.0f}s ===")

        # Per-bin training-loss breakdown
        train_bin_means = None
        if NB > 0 and bin_loss_count is not None:
            train_bin_means = [bin_loss_sum[i] / bin_loss_count[i] if bin_loss_count[i] > 0 else float("nan")
                               for i in range(NB)]
            bins_str = "  ".join([f"[{bin_edges[i]:.2f},{bin_edges[i+1]:.2f}]={train_bin_means[i]:.3f}"
                                  for i in range(NB)])
            log(f"  train_bins: {bins_str}")

        # Save checkpoint
        ck_path = output_dir / "checkpoint_latest.pt"
        _atomic_torch_save({
            "controlnet_state_dict": controlnet.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_sched.state_dict(),
            "epoch": epoch,
            "step": step,
            "mean_loss": epoch_mean,
            "seed": args.seed,
            "training_args": vars(args),
            "run_fingerprints": run_fingerprints,
            "latent_mask_helper_sha256": run_fingerprints[
                "latent_mask_helper_sha256"
            ],
            "latent_mask_contract_sha256": mask_contract_sha256(),
            "masked_loss_contract": MASKED_LOSS_CONTRACT,
            "masked_loss_contract_sha256": MASKED_LOSS_CONTRACT_SHA256,
            "run_contract_sha256": args.run_contract_sha256,
            "run_contract_sidecar_sha256": (
                locked_sidecar_file_sha256
                if args.run_contract_file
                else None
            ),
            "run_contract_hyperparameters": (
                locked_sidecar["run_contract"].get("hyperparameters")
                if locked_sidecar is not None
                else None
            ),
            "continual_success_contract_sha256": (
                locked_sidecar["run_contract"].get(
                    "continual_success_contract_sha256"
                )
                if locked_sidecar is not None
                else None
            ),
            "run_contract_input_files": (
                locked_sidecar["run_contract"].get("input_files")
                if locked_sidecar is not None
                else None
            ),
            "named_parameter_contract": named_parameter_contract,
            "named_parameter_contract_sha256": run_fingerprints[
                "named_parameter_contract_sha256"
            ],
            "planned_cli_args": (
                locked_sidecar.get("planned_cli_args")
                if locked_sidecar is not None
                else sys.argv[1:]
            ),
            "full_process_argv": _current_process_argv(),
            "validation_draw_contract_sha256": validation_draw_contract_sha256,
            "sample_exposures": sample_exposures,
            "validation_evaluations": validation_evaluations,
            "error_event_counts": dict(error_event_counts),
            "python_rng_state": random.getstate(),
            "shuffle_rng_state": rng.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        }, ck_path)
        log(f"saved {ck_path}")

        # --- Validation (val_loss) + best-checkpoint tracking
        if val_entries and (epoch + 1) % args.val_every == 0:
            controlnet.eval()
            val_losses = []
            val_bin_losses = [[] for _ in range(NB)] if NB > 0 else None
            t_val = time.time()
            cpu_rng_before = torch.get_rng_state().clone()
            cuda_rng_before = [state.clone() for state in torch.cuda.get_rng_state_all()]
            with torch.no_grad():
                for ei, ve in enumerate(val_entries):
                    try:
                        v_img, v_lbl, v_mask = load_training_entry(
                            ve, data_root=data_root, device=device
                        )
                    except Exception as e:
                        error_event_counts["validation_load_errors"] += 1
                        if args.strict_data_loading:
                            raise RuntimeError(
                                f"strict data loading failed for validation entry {ve.get('image')}"
                            ) from e
                        print(f"  [val skip {ve['image']}] {type(e).__name__}: {e}", flush=True)
                        continue
                    v_scaled = v_img * scale_factor
                    v_cond = make_cond(v_lbl)
                    v_spacing = torch.tensor([ve["spacing"]], device=device, dtype=torch.float32) * 100.0
                    v_mod = torch.ones((1,), device=device, dtype=torch.long)

                    if args.locked_confirmatory:
                        draw = validation_draws[ve["sample_id"]]
                        t_frac = float(draw["timestep_fraction"])
                        bi = int(draw["bin_index"])
                        if isinstance(noise_scheduler, RFlowScheduler):
                            v_t = torch.tensor(
                                [t_frac * T_max], device=device, dtype=torch.float32
                            )
                        else:
                            v_t = torch.tensor(
                                [int(t_frac * T_max)], device=device, dtype=torch.long
                            )
                        local_generator = torch.Generator(device=device)
                        local_generator.manual_seed(int(draw["noise_seed"]))
                        v_noise = torch.randn(
                            v_scaled.shape,
                            dtype=v_scaled.dtype,
                            device=device,
                            generator=local_generator,
                        )
                    else:
                        if NB > 0:
                            bi = ei % NB
                            lo, hi = bin_edges[bi], bin_edges[bi + 1]
                            t_frac = float(np.random.uniform(lo, hi))
                            if isinstance(noise_scheduler, RFlowScheduler):
                                v_t = torch.tensor(
                                    [t_frac * T_max],
                                    device=device,
                                    dtype=torch.float32,
                                )
                            else:
                                v_t = torch.tensor(
                                    [int(t_frac * T_max)],
                                    device=device,
                                    dtype=torch.long,
                                )
                        else:
                            if isinstance(noise_scheduler, RFlowScheduler):
                                v_t = noise_scheduler.sample_timesteps(v_scaled)
                            else:
                                v_t = torch.randint(
                                    0,
                                    noise_scheduler.num_train_timesteps,
                                    (1,),
                                    device=device,
                                ).long()
                        v_noise = torch.randn_like(v_scaled)

                    v_noisy = noise_scheduler.add_noise(original_samples=v_scaled, noise=v_noise, timesteps=v_t)
                    with autocast("cuda"):
                        v_down, v_mid = controlnet(
                            x=v_noisy, timesteps=v_t,
                            controlnet_cond=v_cond, class_labels=v_mod,
                        )
                        v_out = unet(
                            x=v_noisy, timesteps=v_t, spacing_tensor=v_spacing,
                            down_block_additional_residuals=v_down,
                            mid_block_additional_residual=v_mid,
                            class_labels=v_mod,
                        )
                        v_target = v_scaled - v_noise if isinstance(noise_scheduler, RFlowScheduler) else v_noise
                        v_loss_t = masked_output_l1(v_out, v_target, v_mask)
                    v_loss_f = float(v_loss_t.detach().cpu())
                    if not math.isfinite(v_loss_f):
                        error_event_counts["nonfinite_losses"] += 1
                        if args.locked_confirmatory:
                            raise RuntimeError(
                                "locked validation encountered its first nonfinite loss"
                            )
                    val_losses.append(v_loss_f)
                    validation_evaluations += 1
                    if NB > 0:
                        val_bin_losses[ei % NB].append(v_loss_f)
                    del (
                        v_img,
                        v_lbl,
                        v_mask,
                        v_cond,
                        v_noisy,
                        v_noise,
                        v_out,
                        v_down,
                        v_mid,
                    )
            if args.locked_confirmatory:
                if len(val_losses) != len(val_entries):
                    raise RuntimeError("locked validation was not exact and complete")
                if not torch.equal(torch.get_rng_state(), cpu_rng_before) or any(
                    not torch.equal(after, before)
                    for after, before in zip(
                        torch.cuda.get_rng_state_all(), cuda_rng_before
                    )
                ):
                    raise RuntimeError("fixed validation changed the training RNG state")
            val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
            if args.locked_confirmatory and not math.isfinite(val_loss):
                raise RuntimeError("locked validation mean is nonfinite")
            log(f"=== epoch {epoch} val_loss={val_loss:.4f} (n={len(val_losses)} samples, {time.time()-t_val:.0f}s) ===")
            if NB > 0 and val_bin_losses is not None:
                val_bin_means = [float(np.mean(b)) if b else float("nan") for b in val_bin_losses]
                bins_str = "  ".join([f"[{bin_edges[i]:.2f},{bin_edges[i+1]:.2f}]={val_bin_means[i]:.3f}(n={len(val_bin_losses[i])})"
                                      for i in range(NB)])
                log(f"  val_bins: {bins_str}")

            # Track best
            if not hasattr(main, "_best_val_loss"):
                main._best_val_loss = float("inf")
                main._best_val_epoch = -1
            if val_loss < main._best_val_loss:
                main._best_val_loss = val_loss
                main._best_val_epoch = epoch
                best_path = output_dir / "checkpoint_best.pt"
                _atomic_torch_save({
                    "controlnet_state_dict": controlnet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_sched.state_dict(),
                    "epoch": epoch,
                    "step": step,
                    "mean_loss": epoch_mean,
                    "val_loss": val_loss,
                    "seed": args.seed,
                    "training_args": vars(args),
                    "run_fingerprints": run_fingerprints,
                    "latent_mask_helper_sha256": run_fingerprints[
                        "latent_mask_helper_sha256"
                    ],
                    "latent_mask_contract_sha256": mask_contract_sha256(),
                    "masked_loss_contract": MASKED_LOSS_CONTRACT,
                    "masked_loss_contract_sha256": MASKED_LOSS_CONTRACT_SHA256,
                    "run_contract_sha256": args.run_contract_sha256,
                    "run_contract_sidecar_sha256": (
                        locked_sidecar_file_sha256
                        if args.run_contract_file
                        else None
                    ),
                    "run_contract_hyperparameters": (
                        locked_sidecar["run_contract"].get("hyperparameters")
                        if locked_sidecar is not None
                        else None
                    ),
                    "continual_success_contract_sha256": (
                        locked_sidecar["run_contract"].get(
                            "continual_success_contract_sha256"
                        )
                        if locked_sidecar is not None
                        else None
                    ),
                    "run_contract_input_files": (
                        locked_sidecar["run_contract"].get("input_files")
                        if locked_sidecar is not None
                        else None
                    ),
                    "named_parameter_contract": named_parameter_contract,
                    "named_parameter_contract_sha256": run_fingerprints[
                        "named_parameter_contract_sha256"
                    ],
                    "planned_cli_args": (
                        locked_sidecar.get("planned_cli_args")
                        if locked_sidecar is not None
                        else sys.argv[1:]
                    ),
                    "full_process_argv": _current_process_argv(),
                    "validation_draw_contract_sha256": validation_draw_contract_sha256,
                    "sample_exposures": sample_exposures,
                    "validation_evaluations": validation_evaluations,
                    "error_event_counts": dict(error_event_counts),
                }, best_path)
                log(f"  *** new best val_loss={val_loss:.4f} at epoch {epoch}; saved {best_path}")
            latest_payload = torch.load(
                ck_path, map_location="cpu", weights_only=False
            )
            latest_payload.update(
                {
                    "val_loss": val_loss,
                    "best_val_loss": main._best_val_loss,
                    "best_val_epoch": main._best_val_epoch,
                    "sample_exposures": sample_exposures,
                    "validation_evaluations": validation_evaluations,
                    "error_event_counts": dict(error_event_counts),
                }
            )
            _atomic_torch_save(latest_payload, ck_path)
            controlnet.train()

        # --- Periodic vis inference
        if ae_vis is not None and args.vis_every > 0 and (epoch + 1) % args.vis_every == 0:
            log(f"running vis inference (epoch {epoch + 1}) ...")
            unet.eval()
            controlnet.eval()
            try:
                with torch.no_grad():
                    sched_vis = deepcopy(noise_scheduler)
                    sched_vis.set_timesteps(
                        num_inference_steps=args.vis_steps,
                        input_img_size_numel=64 ** 3,
                    )
                    z_vis = torch.randn((1, 4, 64, 64, 64), device=device)
                    # RFlowScheduler.step needs next_timestep for correct ODE integration
                    vis_timesteps = sched_vis.timesteps.tolist()
                    vis_next_ts = vis_timesteps[1:] + [0]
                    for t_vis, next_t_vis in zip(vis_timesteps, vis_next_ts):
                        t_b = torch.tensor([t_vis], device=device)
                        with autocast("cuda"):
                            down_vis, mid_vis = controlnet(
                                x=z_vis, timesteps=t_b,
                                controlnet_cond=vis_cond,
                                class_labels=vis_modality,
                            )
                            v_vis = unet(
                                x=z_vis, timesteps=t_b,
                                spacing_tensor=vis_spacing,
                                down_block_additional_residuals=down_vis,
                                mid_block_additional_residual=mid_vis,
                                class_labels=vis_modality,
                            )
                        _step_out = sched_vis.step(v_vis, t_vis, z_vis, next_timestep=next_t_vis)
                        z_vis = _step_out[0] if isinstance(_step_out, (list, tuple)) else _step_out.prev_sample
                    ct_arr = ae_vis.decode_stage_2_outputs(
                        z_vis.cpu() / scale_factor.cpu()
                    ).squeeze().numpy()
                    # Denormalize [0,1] -> HU, then window for display
                    ct_arr = np.clip(ct_arr, 0.0, 1.0)
                    ct_arr = ct_arr * 2000.0 - 1000.0
                    _wl, _ww = 40.0, 400.0
                    ct_arr = np.clip((ct_arr - _wl + _ww / 2.0) / _ww, 0.0, 1.0)

                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                d, h, w = ct_arr.shape
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(ct_arr[d // 2], cmap="gray", vmin=0, vmax=1)
                axes[0].set_title(f"Axial z={d // 2}")
                axes[0].axis("off")
                axes[1].imshow(ct_arr[:, h // 2, :], cmap="gray", vmin=0, vmax=1)
                axes[1].set_title(f"Coronal y={h // 2}")
                axes[1].axis("off")
                axes[2].imshow(ct_arr[:, :, w // 2], cmap="gray", vmin=0, vmax=1)
                axes[2].set_title(f"Sagittal x={w // 2}")
                axes[2].axis("off")
                plt.suptitle(f"epoch {epoch + 1}  loss {epoch_mean:.4f}")
                plt.tight_layout()
                vis_path = vis_dir / f"epoch_{epoch + 1:04d}.png"
                plt.savefig(str(vis_path), dpi=80, bbox_inches="tight")
                plt.close(fig)
                log(f"vis saved: {vis_path}")
                del ct_arr, z_vis
            except Exception as vis_exc:
                log(f"vis FAILED (non-fatal): {vis_exc}")
            finally:
                unet.eval()
                controlnet.train()

    if args.locked_confirmatory:
        if sha256_file(Path(args.run_contract_file)) != locked_sidecar_file_sha256:
            raise RuntimeError("run-contract sidecar changed during execution")
        terminal_run_contract = locked_sidecar["run_contract"]
        for name, record in terminal_run_contract["input_files"].items():
            path = Path(str(record["path"]))
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"locked input changed during execution at {name}")
        # Re-open the recursive repository/data/environment roots before signing
        # completion so a mid-run mutation cannot inherit a valid receipt.
        _locked_production_preflight(terminal_run_contract)
        terminal_test_gate = terminal_run_contract["test_release_gate"]
        if os.path.lexists(Path(str(terminal_test_gate["path"]))):
            raise RuntimeError("physical test-release gate opened during training")
        for dependency in terminal_run_contract["dependencies"]:
            dependency_key = str(dependency["run_id"])
            receipt_path = Path(str(dependency["receipt_path"]))
            checkpoint_path = Path(str(dependency["checkpoint_path"]))
            if (
                not receipt_path.is_file()
                or sha256_file(receipt_path)
                != run_fingerprints[
                    f"dependency_{dependency_key}_receipt_sha256"
                ]
                or not checkpoint_path.is_file()
                or sha256_file(checkpoint_path)
                != run_fingerprints[f"dependency_{dependency_key}_best_sha256"]
            ):
                raise RuntimeError(
                    f"dependency changed during execution: {dependency_key}"
                )
            if "importance_artifact_path" in dependency:
                importance_dependency_path = Path(
                    str(dependency["importance_artifact_path"])
                )
                if (
                    not importance_dependency_path.is_file()
                    or sha256_file(importance_dependency_path)
                    != run_fingerprints[
                        f"dependency_{dependency_key}_importance_sha256"
                    ]
                ):
                    raise RuntimeError(
                        "importance dependency changed during execution"
                    )
        if args.compute_fisher and (
            not importance_complete or main._fisher_count != args.fisher_steps
        ):
            raise RuntimeError("importance run did not complete its exact step budget")
        expected_budget = {
            "terminal_epoch_zero_based": args.expected_terminal_epoch,
            "optimizer_steps": args.expected_optimizer_steps,
            "sample_exposures": args.expected_sample_exposures,
            "validation_evaluations": args.expected_validation_evaluations,
        }
        observed_budget = {
            "terminal_epoch_zero_based": n_epochs - 1,
            "optimizer_steps": step,
            "sample_exposures": sample_exposures,
            "validation_evaluations": validation_evaluations,
        }
        if observed_budget != expected_budget:
            raise RuntimeError(
                f"locked run budget mismatch: {observed_budget} != {expected_budget}"
            )
        if any(error_event_counts.values()):
            raise RuntimeError(
                f"locked run has nonzero error events: {error_event_counts}"
            )
        latest_path = output_dir / "checkpoint_latest.pt"
        best_path = output_dir / "checkpoint_best.pt"
        if not latest_path.is_file() or not best_path.is_file():
            raise RuntimeError("locked run lacks terminal latest/best checkpoints")
        latest_payload = torch.load(
            latest_path, map_location="cpu", weights_only=False
        )
        if (
            latest_payload.get("run_contract_sha256")
            != args.run_contract_sha256
            or latest_payload.get("epoch") != n_epochs - 1
            or latest_payload.get("step") != step
            or latest_payload.get("planned_cli_args") != sys.argv[1:]
            or latest_payload.get("full_process_argv")
            != _current_process_argv()
            or latest_payload.get("sample_exposures") != sample_exposures
            or latest_payload.get("validation_evaluations")
            != validation_evaluations
            or latest_payload.get("error_event_counts") != error_event_counts
            or latest_payload.get("run_fingerprints") != run_fingerprints
            or latest_payload.get("latent_mask_helper_sha256")
            != run_fingerprints["latent_mask_helper_sha256"]
            or latest_payload.get("latent_mask_contract_sha256")
            != mask_contract_sha256()
            or latest_payload.get("masked_loss_contract") != MASKED_LOSS_CONTRACT
            or latest_payload.get("masked_loss_contract_sha256")
            != MASKED_LOSS_CONTRACT_SHA256
            or latest_payload.get("named_parameter_contract_sha256")
            != run_fingerprints["named_parameter_contract_sha256"]
        ):
            raise RuntimeError("terminal checkpoint does not bind the exact run")
        importance_record = None
        if args.compute_fisher:
            importance_path = Path(args.compute_fisher)
            importance_payload = torch.load(
                importance_path, map_location="cpu", weights_only=False
            )
            importance_provenance = locked_sidecar["run_contract"][
                "importance_output_provenance"
            ]
            importance_provenance = resolved_importance_provenance(
                importance_provenance
            )
            importance_summary = validate_importance_payload(
                importance_payload,
                named_parameters=dict(controlnet.named_parameters()),
                expected_provenance=importance_provenance,
            )
            importance_record = {
                "path": str(importance_path.resolve()),
                "sha256": sha256_file(importance_path),
                **importance_summary,
            }
        receipt_unsigned = {
            "schema": COMPLETION_SCHEMA,
            "schema_version": 1,
            "status": "complete",
            "run_id": locked_sidecar["run_contract"]["run_id"],
            "run_contract_sha256": args.run_contract_sha256,
            "run_contract_sidecar_sha256": sha256_file(
                Path(args.run_contract_file)
            ),
            "planned_cli_args": locked_sidecar["planned_cli_args"],
            "planned_cli_args_sha256": canonical_sha256(
                locked_sidecar["planned_cli_args"]
            ),
            "full_process_argv": _current_process_argv(),
            "full_process_argv_sha256": canonical_sha256(
                _current_process_argv()
            ),
            "run_contract_hyperparameters": locked_sidecar["run_contract"].get(
                "hyperparameters"
            ),
            "continual_success_contract_sha256": locked_sidecar[
                "run_contract"
            ].get("continual_success_contract_sha256"),
            "required_downstream_method_outputs": locked_sidecar[
                "run_contract"
            ].get("required_downstream_method_outputs"),
            "required_downstream_contrast_outputs": locked_sidecar[
                "run_contract"
            ].get("required_downstream_contrast_outputs"),
            "run_contract_input_files": locked_sidecar["run_contract"].get(
                "input_files"
            ),
            "input_fingerprints": run_fingerprints,
            "latent_mask_helper_sha256": run_fingerprints[
                "latent_mask_helper_sha256"
            ],
            "latent_mask_contract_sha256": mask_contract_sha256(),
            "masked_loss_contract": MASKED_LOSS_CONTRACT,
            "masked_loss_contract_sha256": MASKED_LOSS_CONTRACT_SHA256,
            "named_parameter_contract": named_parameter_contract,
            "named_parameter_contract_sha256": run_fingerprints[
                "named_parameter_contract_sha256"
            ],
            "validation_draw_contract_sha256": validation_draw_contract_sha256,
            "expected_budget": expected_budget,
            "observed_budget": observed_budget,
            "error_event_counts": error_event_counts,
            "latest_checkpoint_path": str(latest_path.resolve()),
            "latest_checkpoint_sha256": sha256_file(latest_path),
            "best_checkpoint_path": str(best_path.resolve()),
            "best_checkpoint_sha256": sha256_file(best_path),
            "importance_artifact": importance_record,
        }
        receipt = signed_payload(receipt_unsigned)
        atomic_write_json(output_dir / "completion_receipt.json", receipt)
        log(f"signed completion receipt saved: {output_dir / 'completion_receipt.json'}")

    log(f"training done: total {time.time()-t_start:.0f}s, {step} steps")


if __name__ == "__main__":
    main()
