"""Pure proof utilities for locked controlled-FOV training runs.

The functions in this module deliberately avoid MONAI/CUDA imports so contract
builders and adversarial unit tests can validate run, validation-draw,
importance, and completion artifacts without constructing a model.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping, Sequence


SHA256_LENGTH = 64
VALIDATION_DRAW_SCHEMA = "medgen.fov_fixed_validation_draws"
RUN_SIDECAR_SCHEMA = "medgen.fov_controlnet_run_contract"
COMPLETION_SCHEMA = "medgen.fov_controlnet_completion_receipt"
IMPORTANCE_SCHEMA = "medgen.fov_diagonal_empirical_l1_gradient_importance"
IMPORTANCE_SCHEMA_VERSION = 2
IMPORTANCE_ESTIMATOR = (
    "mean_squared_parameter_gradient_of_equal_examination_rf73_masked_L1_flow_objective"
)
REPOSITORY_SNAPSHOT_SCHEMA = "medgen.fov_shared_repository_snapshot"
EXECUTION_ENVIRONMENT_SCHEMA = "medgen.fov_execution_environment"
FARM04_HOST = "FARM04"
ENVIRONMENT_VERSION_KEYS = (
    "python",
    "torch",
    "torch_cuda",
    "cudnn",
    "monai",
    "numpy",
    "scipy",
    "nibabel",
    "scikit_image",
)
MONAI_SOURCE_KEYS = (
    "monai.networks.utils.copy_model_state",
    "monai.networks.schedulers.RFlowScheduler",
    "monai.apps.generation.maisi.networks.autoencoderkl_maisi.AutoencoderKlMaisi",
    "monai.apps.generation.maisi.networks.diffusion_model_unet_maisi.DiffusionModelUNetMaisi",
    "monai.apps.generation.maisi.networks.controlnet_maisi.ControlNetMaisi",
)
REQUIRED_ENVIRONMENT_ARTIFACT_KEYS = ("conda_explicit", "pip_freeze")


class TrainingProofError(RuntimeError):
    """Raised when a locked runtime proof is malformed or inconsistent."""


def canonical_bytes(value: Any) -> bytes:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise TrainingProofError(f"Value is not canonical JSON: {error}") from error
    return (payload + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrainingProofError(f"{field} must be a lowercase SHA-256.")
    return value


def source_object_sha256(value: Any) -> str:
    """Hash the exact inspectable source of one imported semantic object."""

    try:
        source = inspect.getsource(value)
    except (OSError, TypeError) as error:
        raise TrainingProofError(
            f"Cannot inspect imported semantic source {value!r}: {error}"
        ) from error
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _safe_relative(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TrainingProofError(f"{field} must be a non-empty relative path.")
    relative = Path(value)
    if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
        raise TrainingProofError(f"{field} is not a safe relative path: {value!r}.")
    return relative


def _regular_tree_files(root: Path) -> set[str]:
    files: set[str] = set()
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in dirnames:
            path = directory_path / name
            if path.is_symlink():
                raise TrainingProofError(
                    f"Signed bundle contains a directory symlink: {path}."
                )
        for name in filenames:
            path = directory_path / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise TrainingProofError(
                    f"Signed bundle payload is not a regular file: {path}."
                )
            files.add(path.relative_to(root).as_posix())
    return files


def verify_recursive_sha256sums(
    root: Path,
    *,
    expected_sha256s_sha256: str,
    allowed_unchecksummed: Sequence[str] = (),
    require_sorted: bool = True,
) -> dict[str, str]:
    """Verify a safe, sorted, exhaustive recursive checksum manifest."""

    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise TrainingProofError(f"Signed bundle root is missing or a symlink: {root}.")
    sums = root / "SHA256SUMS"
    expected_sums = require_sha256(
        expected_sha256s_sha256, "expected SHA256SUMS SHA-256"
    )
    if sums.is_symlink() or not sums.is_file() or sha256_file(sums) != expected_sums:
        raise TrainingProofError(f"Signed bundle SHA256SUMS drifted: {sums}.")
    records: dict[str, str] = {}
    previous: str | None = None
    try:
        lines = sums.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise TrainingProofError(f"Cannot read signed SHA256SUMS {sums}: {error}") from error
    if not lines:
        raise TrainingProofError(f"Signed SHA256SUMS is empty: {sums}.")
    for line_number, line in enumerate(lines, 1):
        if len(line) < 67 or line[64:66] != "  ":
            raise TrainingProofError(f"Malformed SHA256SUMS line {line_number}.")
        digest = require_sha256(line[:64], f"SHA256SUMS line {line_number}")
        relative = _safe_relative(line[66:], f"SHA256SUMS line {line_number}")
        name = relative.as_posix()
        if name == "SHA256SUMS" or name in records:
            raise TrainingProofError(f"Unsafe or duplicate checksum path: {name}.")
        if require_sorted and previous is not None and name <= previous:
            raise TrainingProofError("SHA256SUMS paths must be strictly sorted.")
        path = root / relative
        cursor = root
        for part in relative.parts:
            cursor /= part
            if cursor.is_symlink():
                raise TrainingProofError(f"Checksummed path contains a symlink: {cursor}.")
        if not path.is_file() or sha256_file(path) != digest:
            raise TrainingProofError(f"Checksummed payload drifted: {path}.")
        records[name] = digest
        previous = name
    actual = _regular_tree_files(root) - {"SHA256SUMS"}
    allowed = set(allowed_unchecksummed)
    unlisted = actual - set(records)
    stale = set(records) - actual
    if stale or not unlisted <= allowed:
        raise TrainingProofError(
            "SHA256SUMS is not exhaustive: "
            f"unlisted={sorted(unlisted - allowed)[:5]}, "
            f"missing={sorted(stale)[:5]}."
        )
    return records


def validate_repository_snapshot_record(
    record: Mapping[str, Any],
    *,
    required_relative_paths: Sequence[str],
    target_host: str = FARM04_HOST,
) -> dict[str, Any]:
    """Verify the immutable code snapshot used by a remote training process."""

    if not isinstance(record, Mapping):
        raise TrainingProofError("Repository snapshot record is missing.")
    if record.get("target_host") != target_host:
        raise TrainingProofError("Repository snapshot record targets the wrong host.")
    root = Path(str(record.get("repository_root")))
    manifest_path = Path(str(record.get("manifest_path")))
    sums_path = Path(str(record.get("sha256s_path")))
    if (
        not root.is_absolute()
        or manifest_path.parent != root
        or sums_path != root / "SHA256SUMS"
    ):
        raise TrainingProofError("Repository snapshot root/manifest paths drifted.")
    expected_manifest = require_sha256(
        record.get("manifest_sha256"), "repository snapshot manifest SHA-256"
    )
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or sha256_file(manifest_path) != expected_manifest
    ):
        raise TrainingProofError("Repository snapshot manifest bytes drifted.")
    records = verify_recursive_sha256sums(
        root,
        expected_sha256s_sha256=require_sha256(
            record.get("sha256s_sha256"), "repository snapshot SHA256SUMS SHA-256"
        ),
    )
    manifest_relative = manifest_path.relative_to(root).as_posix()
    if records.get(manifest_relative) != expected_manifest:
        raise TrainingProofError("Repository SHA256SUMS does not bind its manifest.")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingProofError(f"Invalid repository snapshot manifest: {error}") from error
    if not isinstance(payload, Mapping):
        raise TrainingProofError("Repository snapshot manifest must be one object.")
    unsigned = validate_signed_payload(
        payload,
        schema=REPOSITORY_SNAPSHOT_SCHEMA,
        label="repository snapshot manifest",
    )
    hosts = unsigned.get("target_hosts")
    files = unsigned.get("files")
    if (
        unsigned.get("status") != "frozen"
        or unsigned.get("recursive_exhaustive") is not True
        or unsigned.get("repository_root") != str(root)
        or not isinstance(hosts, list)
        or target_host not in hosts
        or not isinstance(files, Mapping)
    ):
        raise TrainingProofError("Repository snapshot contract is incomplete or remote-ineligible.")
    normalized_files: dict[str, str] = {}
    for name, digest in files.items():
        relative = _safe_relative(name, "repository snapshot file")
        normalized_files[relative.as_posix()] = require_sha256(
            digest, f"repository snapshot file {name}"
        )
    expected_payloads = dict(records)
    expected_payloads.pop(manifest_relative, None)
    if normalized_files != expected_payloads:
        raise TrainingProofError("Repository snapshot file inventory is not exhaustive.")
    required = {_safe_relative(value, "required repository file").as_posix() for value in required_relative_paths}
    if not required <= set(normalized_files):
        raise TrainingProofError(
            f"Repository snapshot lacks required semantic files: {sorted(required - set(normalized_files))}."
        )
    return {**dict(unsigned), "manifest_sha256": expected_manifest, "files": normalized_files}


def validate_execution_environment_record(
    record: Mapping[str, Any],
    *,
    observed: Mapping[str, Any] | None = None,
    target_host: str = FARM04_HOST,
) -> dict[str, Any]:
    """Verify a signed environment manifest and optionally the live interpreter."""

    if not isinstance(record, Mapping):
        raise TrainingProofError("Execution-environment record is missing.")
    if record.get("target_host") != target_host:
        raise TrainingProofError("Execution-environment record targets the wrong host.")
    manifest_path = Path(str(record.get("manifest_path")))
    expected_manifest = require_sha256(
        record.get("manifest_sha256"), "execution environment manifest SHA-256"
    )
    if (
        not manifest_path.is_absolute()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
        or sha256_file(manifest_path) != expected_manifest
    ):
        raise TrainingProofError("Execution-environment manifest bytes drifted.")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingProofError(f"Invalid execution-environment manifest: {error}") from error
    if not isinstance(payload, Mapping):
        raise TrainingProofError("Execution-environment manifest must be one object.")
    unsigned = validate_signed_payload(
        payload,
        schema=EXECUTION_ENVIRONMENT_SCHEMA,
        label="execution environment manifest",
    )
    versions = unsigned.get("versions")
    sources = unsigned.get("monai_source_sha256")
    executable = unsigned.get("python_executable")
    artifacts = unsigned.get("artifacts")
    if (
        unsigned.get("status") != "frozen"
        or unsigned.get("target_host") != target_host
        or not isinstance(versions, Mapping)
        or set(versions) != set(ENVIRONMENT_VERSION_KEYS)
        or any(not isinstance(versions[key], str) or not versions[key] for key in versions)
        or not isinstance(sources, Mapping)
        or set(sources) != set(MONAI_SOURCE_KEYS)
        or not isinstance(executable, Mapping)
        or not isinstance(artifacts, Mapping)
        or set(artifacts) != set(REQUIRED_ENVIRONMENT_ARTIFACT_KEYS)
    ):
        raise TrainingProofError("Execution-environment contract is incomplete.")
    normalized_sources = {
        key: require_sha256(sources[key], f"MONAI semantic source {key}")
        for key in MONAI_SOURCE_KEYS
    }
    executable_path = Path(str(executable.get("path")))
    executable_hash = require_sha256(
        executable.get("sha256"), "Python executable SHA-256"
    )
    if (
        not executable_path.is_absolute()
        or not executable_path.is_file()
        or sha256_file(executable_path) != executable_hash
    ):
        raise TrainingProofError("Python executable path/hash drifted.")
    normalized_artifacts: dict[str, dict[str, str]] = {}
    for name, artifact in artifacts.items():
        if not isinstance(artifact, Mapping):
            raise TrainingProofError(f"Environment artifact {name} is malformed.")
        path = Path(str(artifact.get("path")))
        digest = require_sha256(artifact.get("sha256"), f"environment artifact {name}")
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
            or sha256_file(path) != digest
        ):
            raise TrainingProofError(f"Environment artifact {name} drifted.")
        normalized_artifacts[name] = {"path": str(path), "sha256": digest}
    normalized = {
        **dict(unsigned),
        "versions": dict(versions),
        "monai_source_sha256": normalized_sources,
        "python_executable": {"path": str(executable_path), "sha256": executable_hash},
        "artifacts": normalized_artifacts,
        "manifest_sha256": expected_manifest,
    }
    if observed is not None:
        for field in ("target_host", "versions", "monai_source_sha256"):
            if observed.get(field) != normalized.get(field):
                raise TrainingProofError(f"Live execution environment {field} drifted.")
        observed_executable = observed.get("python_executable")
        if not isinstance(observed_executable, Mapping) or (
            observed_executable.get("path") != str(executable_path)
            or observed_executable.get("sha256") != executable_hash
        ):
            raise TrainingProofError("Live sys.executable differs from the frozen contract.")
    return normalized


def validate_remote_path_contract(
    contract: Mapping[str, Any], *, require_existing: bool = False
) -> dict[str, str]:
    """Reject local-only paths from a production FARM04 run contract."""

    if not isinstance(contract, Mapping) or contract.get("target_host") != FARM04_HOST:
        raise TrainingProofError("Remote path contract does not target FARM04.")
    allowed_raw = contract.get("allowed_shared_roots")
    forbidden_raw = contract.get("forbidden_local_roots")
    paths = contract.get("paths")
    if (
        not isinstance(allowed_raw, list)
        or not allowed_raw
        or not isinstance(forbidden_raw, list)
        or not isinstance(paths, Mapping)
    ):
        raise TrainingProofError("Remote path contract is incomplete.")
    allowed = [Path(value) for value in allowed_raw]
    forbidden = [Path(value) for value in forbidden_raw]
    if any(not value.is_absolute() for value in (*allowed, *forbidden)):
        raise TrainingProofError("Remote path roots must be absolute.")
    normalized: dict[str, str] = {}
    for name, raw in paths.items():
        if not isinstance(name, str) or not isinstance(raw, str):
            raise TrainingProofError("Remote path inventory is malformed.")
        supplied_path = Path(raw)
        if not supplied_path.is_absolute():
            raise TrainingProofError(
                f"Production path is not absolute: {name}={supplied_path}."
            )
        path = Path(os.path.abspath(raw))
        if any(path == root or root in path.parents for root in forbidden):
            raise TrainingProofError(f"Production path is local-only: {name}={path}.")
        if not any(path == root or root in path.parents for root in allowed):
            raise TrainingProofError(f"Production path is not on shared storage: {name}={path}.")
        if require_existing and not path.exists():
            raise TrainingProofError(f"Remote production path is missing: {name}={path}.")
        normalized[name] = str(path)
    return normalized


def validate_semantic_source_files(
    repository_root: Path,
    records: Mapping[str, Any],
    *,
    required_names: Sequence[str],
) -> dict[str, dict[str, str]]:
    """Bind semantic repository helpers by relative location and exact bytes."""

    root = Path(repository_root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise TrainingProofError("Semantic repository root is missing, relative, or a symlink.")
    if not isinstance(records, Mapping) or set(records) != set(required_names):
        raise TrainingProofError("Semantic source-file inventory is incomplete.")
    normalized: dict[str, dict[str, str]] = {}
    for name in required_names:
        record = records[name]
        if not isinstance(record, Mapping):
            raise TrainingProofError(f"Semantic source record {name} is malformed.")
        relative = _safe_relative(
            record.get("repository_relative_path"), f"semantic source {name}"
        )
        expected_path = root / relative
        path = Path(str(record.get("path")))
        digest = require_sha256(record.get("sha256"), f"semantic source {name}")
        cursor = root
        has_symlink_component = False
        for part in relative.parts:
            cursor /= part
            if cursor.is_symlink():
                has_symlink_component = True
                break
        if (
            path != expected_path
            or has_symlink_component
            or not path.is_file()
            or sha256_file(path) != digest
        ):
            raise TrainingProofError(f"Semantic source {name} path/hash drifted.")
        normalized[name] = {
            "repository_relative_path": relative.as_posix(),
            "path": str(path),
            "sha256": digest,
        }
    return normalized


def signed_payload(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    if "artifact_sha256" in unsigned:
        raise TrainingProofError("Unsigned payload already contains artifact_sha256.")
    return {**dict(unsigned), "artifact_sha256": canonical_sha256(unsigned)}


def validate_signed_payload(
    payload: Mapping[str, Any], *, schema: str, label: str
) -> dict[str, Any]:
    if payload.get("schema") != schema or payload.get("schema_version") != 1:
        raise TrainingProofError(f"Unexpected {label} schema/version.")
    unsigned = dict(payload)
    declared = require_sha256(
        unsigned.pop("artifact_sha256", None), f"{label}.artifact_sha256"
    )
    if canonical_sha256(unsigned) != declared:
        raise TrainingProofError(f"{label} artifact signature failed.")
    return unsigned


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _hash_integer(material: Mapping[str, Any], domain: str) -> int:
    payload = {**dict(material), "domain": domain}
    return int(canonical_sha256(payload)[:16], 16)


def build_validation_draw_contract(
    *,
    sample_ids: Sequence[str],
    split_sha256: str,
    development_latent_manifest_sha256: str,
    master_seed: int,
    n_bins: int = 5,
) -> dict[str, Any]:
    """Build one method/epoch/order-independent stratified draw per sample."""

    ordered = sorted(sample_ids)
    if (
        not ordered
        or len(ordered) != len(set(ordered))
        or any(not isinstance(value, str) or not value for value in ordered)
    ):
        raise TrainingProofError("Validation sample IDs must be non-empty and unique.")
    if isinstance(master_seed, bool) or not isinstance(master_seed, int) or master_seed < 0:
        raise TrainingProofError("Validation master_seed must be non-negative.")
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise TrainingProofError("Validation n_bins must be positive.")
    roots = {
        "split_sha256": require_sha256(split_sha256, "split_sha256"),
        "development_latent_manifest_sha256": require_sha256(
            development_latent_manifest_sha256,
            "development_latent_manifest_sha256",
        ),
        "master_seed": master_seed,
        "n_bins": n_bins,
    }
    records: list[dict[str, Any]] = []
    for index, sample_id in enumerate(ordered):
        material = {**roots, "sample_id": sample_id}
        bin_index = index % n_bins
        unit = (_hash_integer(material, "timestep") + 0.5) / float(1 << 64)
        fraction = (bin_index + unit) / n_bins
        noise_seed = _hash_integer(material, "noise") & ((1 << 63) - 1)
        records.append(
            {
                "sample_id": sample_id,
                "bin_index": bin_index,
                "timestep_fraction": fraction,
                "noise_seed": noise_seed,
                "material_sha256": canonical_sha256(material),
            }
        )
    unsigned = {
        "schema": VALIDATION_DRAW_SCHEMA,
        "schema_version": 1,
        "status": "locked",
        "method_independent": True,
        "epoch_independent": True,
        "input_order_independent": True,
        "local_generators_only": True,
        "one_stratified_draw_per_case": True,
        "roots": roots,
        "sample_count": len(records),
        "records": records,
    }
    return signed_payload(unsigned)


def validate_validation_draw_contract(
    payload: Mapping[str, Any], *, expected_sample_ids: Sequence[str] | None = None
) -> dict[str, dict[str, Any]]:
    unsigned = validate_signed_payload(
        payload, schema=VALIDATION_DRAW_SCHEMA, label="validation draw contract"
    )
    if (
        unsigned.get("status") != "locked"
        or unsigned.get("method_independent") is not True
        or unsigned.get("epoch_independent") is not True
        or unsigned.get("input_order_independent") is not True
        or unsigned.get("local_generators_only") is not True
        or unsigned.get("one_stratified_draw_per_case") is not True
    ):
        raise TrainingProofError("Validation draw invariance contract failed.")
    records = unsigned.get("records")
    if not isinstance(records, list) or len(records) != unsigned.get("sample_count"):
        raise TrainingProofError("Validation draw records are incomplete.")
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise TrainingProofError("Malformed validation draw record.")
        sample_id = record.get("sample_id")
        fraction = record.get("timestep_fraction")
        noise_seed = record.get("noise_seed")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id in by_id
            or isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(float(fraction))
            or not 0.0 < float(fraction) < 1.0
            or isinstance(noise_seed, bool)
            or not isinstance(noise_seed, int)
            or not 0 <= noise_seed < (1 << 63)
        ):
            raise TrainingProofError("Invalid validation draw values.")
        by_id[sample_id] = dict(record)
    if expected_sample_ids is not None and set(by_id) != set(expected_sample_ids):
        raise TrainingProofError("Validation draw identities are not exact and complete.")
    return by_id


def tensor_mapping_sha256(tensors: Mapping[str, Any]) -> str:
    """Hash named tensor shapes, dtypes, and exact bytes in lexical order."""

    import torch

    if not isinstance(tensors, Mapping) or not tensors:
        raise TrainingProofError("Tensor mapping is empty.")
    digest = hashlib.sha256(b"medgen.fov.named-tensors.v1\0")
    for name in sorted(tensors):
        tensor = tensors[name]
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise TrainingProofError("Tensor mapping contains a non-tensor entry.")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_bytes(list(value.shape)))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def validate_importance_payload(
    payload: Mapping[str, Any],
    *,
    named_parameters: Mapping[str, Any],
    expected_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate RF73-masked equal-examination L1-gradient importance."""

    import torch

    if (
        payload.get("schema") != IMPORTANCE_SCHEMA
        or payload.get("schema_version") != IMPORTANCE_SCHEMA_VERSION
        or payload.get("status") != "complete"
        or payload.get("estimator")
        != IMPORTANCE_ESTIMATOR
        or payload.get("likelihood_fisher_claim") is not False
    ):
        raise TrainingProofError("Importance artifact schema/estimator is invalid.")
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping) or set(tensors) != set(named_parameters):
        raise TrainingProofError("Importance tensors do not match exact named parameters.")
    total_mass = 0.0
    tensor_metadata: list[dict[str, Any]] = []
    for name in sorted(named_parameters):
        parameter = named_parameters[name]
        tensor = tensors[name]
        if not isinstance(parameter, torch.Tensor) or not isinstance(tensor, torch.Tensor):
            raise TrainingProofError("Importance/parameter entry is not a tensor.")
        value = tensor.detach().cpu().float().contiguous()
        if tuple(value.shape) != tuple(parameter.shape):
            raise TrainingProofError(f"Importance shape mismatch for {name}.")
        if not torch.isfinite(value).all() or torch.any(value < 0):
            raise TrainingProofError(f"Importance tensor is nonfinite/negative: {name}.")
        mass = float(value.double().sum().item())
        if not math.isfinite(mass) or mass <= 0.0:
            raise TrainingProofError(
                f"Importance tensor must have finite nonzero mass: {name}."
            )
        total_mass += mass
        tensor_metadata.append(
            {"name": name, "shape": list(value.shape), "sum": mass}
        )
    if not math.isfinite(total_mass) or total_mass <= 0.0:
        raise TrainingProofError("Importance tensor mass must be finite and nonzero.")
    observed_tensor_hash = tensor_mapping_sha256(tensors)
    if payload.get("tensor_inventory_sha256") != observed_tensor_hash:
        raise TrainingProofError("Importance tensor inventory hash drifted.")
    if payload.get("tensor_metadata_sha256") != canonical_sha256(tensor_metadata):
        raise TrainingProofError("Importance tensor metadata hash drifted.")
    steps = payload.get("optimizer_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise TrainingProofError("Importance optimizer_steps is invalid.")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TrainingProofError("Importance provenance is missing.")
    if expected_provenance is not None:
        for key, value in expected_provenance.items():
            if provenance.get(key) != value:
                raise TrainingProofError(f"Importance provenance drifted at {key}.")
    unsigned_metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"tensors", "metadata_artifact_sha256"}
    }
    if payload.get("metadata_artifact_sha256") != canonical_sha256(unsigned_metadata):
        raise TrainingProofError("Importance metadata signature failed.")
    return {
        "optimizer_steps": steps,
        "parameter_count": len(tensors),
        "total_mass": total_mass,
        "tensor_inventory_sha256": observed_tensor_hash,
    }


def validate_completion_receipt(
    payload: Mapping[str, Any], *, expected_run_contract_sha256: str
) -> dict[str, Any]:
    from src.fov_latent_mask import (
        masked_loss_contract,
        masked_loss_contract_sha256,
    )

    unsigned = validate_signed_payload(
        payload, schema=COMPLETION_SCHEMA, label="completion receipt"
    )
    if (
        unsigned.get("status") != "complete"
        or unsigned.get("run_contract_sha256")
        != require_sha256(expected_run_contract_sha256, "run_contract_sha256")
    ):
        raise TrainingProofError("Completion receipt run/status mismatch.")
    if (
        unsigned.get("masked_loss_contract") != masked_loss_contract()
        or unsigned.get("masked_loss_contract_sha256")
        != masked_loss_contract_sha256()
    ):
        raise TrainingProofError("Completion receipt lacks the active masked loss.")
    counts = unsigned.get("error_event_counts")
    if not isinstance(counts, Mapping) or any(
        counts.get(name) != 0
        for name in (
            "training_load_errors",
            "validation_load_errors",
            "nonfinite_losses",
            "nonfinite_gradients",
        )
    ):
        raise TrainingProofError("Completion receipt contains nonzero error events.")
    budget = unsigned.get("observed_budget")
    expected = unsigned.get("expected_budget")
    if not isinstance(budget, Mapping) or budget != expected:
        raise TrainingProofError("Completion receipt did not meet its exact budget.")
    for role in ("latest", "best"):
        hash_field = f"{role}_checkpoint_sha256"
        path_field = f"{role}_checkpoint_path"
        require_sha256(unsigned.get(hash_field), f"completion.{hash_field}")
        if not isinstance(unsigned.get(path_field), str) or not unsigned[path_field]:
            raise TrainingProofError(
                f"Completion receipt lacks {role} checkpoint path."
            )
    return unsigned
