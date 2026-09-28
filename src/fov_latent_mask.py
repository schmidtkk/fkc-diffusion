"""Exact acquisition-coverage masks for the MAISI latent objective.

The frozen MAISI encoder geometry maps latent index ``i`` to source index
``4*i + 3``.  Its exact source receptive field has 73 samples per axis, so a
latent location is eligible for a data loss only when every acquired-source
voxel in the clipped Cartesian interval ``[4*i-33, 4*i+39]`` is covered.

Outside-tensor convolution padding is deliberately clipped away: it is an
encoder boundary condition, not missing acquisition.  In-tensor zero coverage
is never imputed and invalidates every latent receptive field that intersects
it.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


MASK_SCHEMA = "medgen.fov_rf73_latent_loss_mask"
MASK_SCHEMA_VERSION = 1
MASK_ALGORITHM_ID = "coverage_mask_latent_rf73_v1"
DOWNSAMPLE_FACTOR = 4
LATENT_CENTER_OFFSET = 3
RECEPTIVE_FIELD_SIZE = 73
RECEPTIVE_FIELD_LEFT_OFFSET = 33
RECEPTIVE_FIELD_RIGHT_OFFSET = 39


class LatentMaskError(ValueError):
    """Raised when a coverage mask or masked objective violates the contract."""


def mask_contract() -> dict[str, Any]:
    """Return the JSON-serializable, versioned RF73 algorithm contract."""

    return {
        "schema": MASK_SCHEMA,
        "schema_version": MASK_SCHEMA_VERSION,
        "algorithm_id": MASK_ALGORITHM_ID,
        "source_to_latent_downsample_factor": DOWNSAMPLE_FACTOR,
        "latent_center_source_index": "4*i+3",
        "source_receptive_field_size_per_axis": RECEPTIVE_FIELD_SIZE,
        "inclusive_source_interval_per_axis": [
            "max(0,4*i-33)",
            "min(N-1,4*i+39)",
        ],
        "eligibility": (
            "one iff every source coverage voxel in the clipped Cartesian "
            "receptive field is one"
        ),
        "outside_tensor_padding": "clipped_and_not_treated_as_missing_acquisition",
        "output_dtype": "uint8",
        "loss": "equal_examination_masked_L1",
        "loss_formula": "mean_b sum(M_b*abs(R_b))/(C*sum(M_b))",
    }


def mask_contract_sha256() -> str:
    """Hash :func:`mask_contract` with deterministic JSON serialization."""

    payload = json.dumps(
        mask_contract(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def masked_loss_contract() -> dict[str, Any]:
    """Return the shared equal-examination objective contract."""

    return {
        "schema": "medgen.fov_equal_examination_rf73_masked_l1",
        "schema_version": 1,
        "mask_field": "latent_loss_mask",
        "mask_algorithm_id": MASK_ALGORITHM_ID,
        "mask_contract_sha256": mask_contract_sha256(),
        "formula": "mean_b sum(M_b*abs(R_b))/(C*sum(M_b))",
        "required_consumers": [
            "diffusion_training_l1",
            "diffusion_validation_l1",
            "lwf_distillation_l1",
            "importance_gradient_l1",
        ],
        "ewc_parameter_penalty_masked": False,
        "z_jitter": "zero_fill_true_Z_and_move_latent_mask_identically",
    }


def masked_loss_contract_sha256() -> str:
    payload = json.dumps(
        masked_loss_contract(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _positive_divisible_shape(shape: Sequence[int], *, field: str) -> tuple[int, int, int]:
    if len(shape) != 3:
        raise LatentMaskError(f"{field} must have exactly three spatial axes")
    parsed: list[int] = []
    for value in shape:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise LatentMaskError(f"{field} must contain integer dimensions")
        integer = int(value)
        if integer <= 0 or integer % DOWNSAMPLE_FACTOR != 0:
            raise LatentMaskError(
                f"{field} dimensions must be positive and divisible by "
                f"{DOWNSAMPLE_FACTOR}: {tuple(shape)}"
            )
        parsed.append(integer)
    return tuple(parsed)  # type: ignore[return-value]


def validate_binary_mask(
    mask: np.ndarray,
    *,
    expected_shape: Sequence[int] | None = None,
    require_nonempty: bool = True,
    field: str = "latent_loss_mask",
) -> np.ndarray:
    """Return a contiguous uint8 binary mask or fail closed.

    Floating input is accepted only when every finite value is exactly zero or
    one.  This supports lossless NIfTI readers without weakening binary
    semantics.
    """

    value = np.asarray(mask)
    if value.ndim != 3:
        raise LatentMaskError(f"{field} must be a three-dimensional array")
    if expected_shape is not None:
        parsed_shape = tuple(int(v) for v in expected_shape)
        if len(parsed_shape) != 3 or tuple(value.shape) != parsed_shape:
            raise LatentMaskError(
                f"{field} shape {tuple(value.shape)} != expected {parsed_shape}"
            )
    if value.dtype.kind not in "buif" or not np.all(np.isfinite(value)):
        raise LatentMaskError(f"{field} must contain finite numeric binary values")
    if not np.all((value == 0) | (value == 1)):
        raise LatentMaskError(f"{field} contains a value outside {{0,1}}")
    result = np.ascontiguousarray(value, dtype=np.uint8)
    if require_nonempty and not np.any(result):
        raise LatentMaskError(f"{field} has no eligible latent voxel")
    return result


def _all_over_rf_intervals(value: np.ndarray, axis: int) -> np.ndarray:
    """Reduce one source axis to latent indices using clipped interval sums."""

    source_length = value.shape[axis]
    if source_length % DOWNSAMPLE_FACTOR != 0:
        raise LatentMaskError("source coverage shape is not divisible by four")
    latent_indices = np.arange(
        source_length // DOWNSAMPLE_FACTOR, dtype=np.int64
    )
    starts = np.maximum(
        0,
        DOWNSAMPLE_FACTOR * latent_indices
        - RECEPTIVE_FIELD_LEFT_OFFSET,
    )
    stops = np.minimum(
        source_length,
        DOWNSAMPLE_FACTOR * latent_indices
        + RECEPTIVE_FIELD_RIGHT_OFFSET
        + 1,
    )

    moved = np.moveaxis(value, axis, 0)
    prefix = np.empty((source_length + 1, *moved.shape[1:]), dtype=np.uint32)
    prefix[0] = 0
    np.cumsum(moved, axis=0, dtype=np.uint32, out=prefix[1:])
    counts = prefix[stops] - prefix[starts]
    interval_lengths = (stops - starts).reshape(
        (len(starts),) + (1,) * (moved.ndim - 1)
    )
    reduced = counts == interval_lengths
    return np.moveaxis(reduced, 0, axis)


def build_rf73_latent_loss_mask(
    source_coverage: np.ndarray,
    *,
    expected_latent_shape: Sequence[int] | None = None,
    require_nonempty: bool = True,
) -> np.ndarray:
    """Construct the exact RF73 latent eligibility mask from source coverage."""

    source = validate_binary_mask(
        source_coverage,
        require_nonempty=True,
        field="source_coverage_mask",
    )
    source_shape = _positive_divisible_shape(
        source.shape, field="source_coverage_mask.shape"
    )
    expected = tuple(length // DOWNSAMPLE_FACTOR for length in source_shape)
    if expected_latent_shape is not None:
        requested = tuple(int(v) for v in expected_latent_shape)
        if len(requested) != 3 or requested != expected:
            raise LatentMaskError(
                f"expected latent shape {requested} disagrees with source-derived {expected}"
            )

    eligible: np.ndarray = source.astype(bool, copy=False)
    for axis in range(3):
        eligible = _all_over_rf_intervals(eligible, axis)
    return validate_binary_mask(
        eligible,
        expected_shape=expected,
        require_nonempty=require_nonempty,
    )


def latent_affine_from_source(source_affine: np.ndarray) -> np.ndarray:
    """Map latent indices to source receptive-field centers ``4*j+3``."""

    source = np.asarray(source_affine, dtype=np.float64)
    if source.shape != (4, 4) or not np.all(np.isfinite(source)):
        raise LatentMaskError("source affine must be finite and 4x4")
    index_map = np.eye(4, dtype=np.float64)
    index_map[:3, :3] *= DOWNSAMPLE_FACTOR
    index_map[:3, 3] = LATENT_CENTER_OFFSET
    result = source @ index_map
    if not np.all(np.isfinite(result)):
        raise LatentMaskError("derived latent affine is non-finite")
    return result


def affine_sha256(affine: np.ndarray) -> str:
    """Hash one exact finite 4x4 affine as little-endian float64 bytes."""

    value = np.ascontiguousarray(np.asarray(affine), dtype="<f8")
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise LatentMaskError("affine must be finite and 4x4")
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def decode_exact_affine_record(
    record: Any, *, expected_sha256: str | None = None
) -> np.ndarray:
    """Decode the lossless float64-le affine record used by latent manifests."""

    keys = {"dtype", "shape", "order", "bytes_base64", "sha256"}
    if not isinstance(record, Mapping) or set(record) != keys:
        raise LatentMaskError("exact affine record has an invalid schema")
    if (
        record.get("dtype") != "<f8"
        or record.get("shape") != [4, 4]
        or record.get("order") != "C"
        or not isinstance(record.get("bytes_base64"), str)
    ):
        raise LatentMaskError("exact affine record contract drifted")
    try:
        raw = base64.b64decode(record["bytes_base64"].encode("ascii"), validate=True)
    except (binascii.Error, ValueError, UnicodeEncodeError) as error:
        raise LatentMaskError("exact affine bytes_base64 is invalid") from error
    if len(raw) != 128:
        raise LatentMaskError("exact affine record must encode 128 bytes")
    affine = np.frombuffer(raw, dtype="<f8").reshape((4, 4), order="C").copy()
    observed = affine_sha256(affine)
    if record.get("sha256") != observed or (
        expected_sha256 is not None and observed != expected_sha256
    ):
        raise LatentMaskError("exact affine record hash drifted")
    if (
        not np.isclose(affine[3, 3], 1.0, rtol=0.0, atol=0.0)
        or not np.allclose(affine[3, :3], 0.0, rtol=0.0, atol=0.0)
        or not np.isfinite(np.linalg.det(affine[:3, :3]))
        or np.isclose(np.linalg.det(affine[:3, :3]), 0.0)
    ):
        raise LatentMaskError("exact affine is not a nonsingular homogeneous matrix")
    return affine


def latent_mask_content_sha256(mask: np.ndarray, *, affine: np.ndarray) -> str:
    """Hash binary values, exact affine, shape, and algorithm semantics."""

    value = validate_binary_mask(mask)
    exact_affine = np.asarray(affine, dtype=np.float64)
    if exact_affine.shape != (4, 4) or not np.all(np.isfinite(exact_affine)):
        raise LatentMaskError("latent mask affine must be finite and 4x4")
    digest = hashlib.sha256()
    digest.update(MASK_ALGORITHM_ID.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype="<u8").tobytes(order="C"))
    digest.update(np.asarray(exact_affine, dtype="<f8").tobytes(order="C"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def equal_examination_masked_l1(residual: Any, mask: Any) -> Any:
    """Compute masked L1 with equal weight for every examination in a batch.

    ``residual`` must be ``(B,C,X,Y,Z)`` and ``mask`` must be
    ``(B,1,X,Y,Z)``.  Each examination is normalized by its own eligible voxel
    count and channel count before the arithmetic mean over examinations.
    """

    import torch

    if not isinstance(residual, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise LatentMaskError("residual and mask must be torch tensors")
    if residual.ndim != 5 or mask.ndim != 5:
        raise LatentMaskError("residual and mask must both be five-dimensional")
    if (
        mask.shape[0] != residual.shape[0]
        or mask.shape[1] != 1
        or tuple(mask.shape[2:]) != tuple(residual.shape[2:])
    ):
        raise LatentMaskError(
            f"mask shape {tuple(mask.shape)} is incompatible with residual "
            f"shape {tuple(residual.shape)}"
        )
    if not torch.isfinite(residual).all() or not torch.isfinite(mask).all():
        raise LatentMaskError("residual and mask must be finite")
    if not torch.all((mask == 0) | (mask == 1)):
        raise LatentMaskError("mask tensor contains a value outside {0,1}")
    mask_float = mask.to(device=residual.device, dtype=residual.dtype)
    eligible = mask_float.sum(dim=(1, 2, 3, 4))
    if torch.any(eligible <= 0):
        raise LatentMaskError("every examination must have an eligible latent voxel")
    numerator = (residual.abs() * mask_float).sum(dim=(1, 2, 3, 4))
    per_examination = numerator / (eligible * residual.shape[1])
    return per_examination.mean()


def zero_fill_shift(tensor: Any, shift: int, *, dim: int = -1) -> Any:
    """Shift a tensor without circular wraparound, filling exposed cells by zero."""

    import torch

    if not isinstance(tensor, torch.Tensor):
        raise LatentMaskError("zero_fill_shift expects a torch tensor")
    if isinstance(shift, bool) or not isinstance(shift, int):
        raise LatentMaskError("shift must be an integer")
    normalized_dim = dim if dim >= 0 else tensor.ndim + dim
    if normalized_dim < 0 or normalized_dim >= tensor.ndim:
        raise LatentMaskError("shift dimension is outside the tensor")
    length = tensor.shape[normalized_dim]
    if abs(shift) >= length:
        return torch.zeros_like(tensor)
    if shift == 0:
        return tensor
    result = torch.zeros_like(tensor)
    source = [slice(None)] * tensor.ndim
    destination = [slice(None)] * tensor.ndim
    if shift > 0:
        source[normalized_dim] = slice(0, length - shift)
        destination[normalized_dim] = slice(shift, length)
    else:
        source[normalized_dim] = slice(-shift, length)
        destination[normalized_dim] = slice(0, length + shift)
    result[tuple(destination)] = tensor[tuple(source)]
    return result
