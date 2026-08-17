# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build node-local host weight cache plans for diffusion DLO.

``build_host_weight_cache_plan`` is the module's single entry point. It
validates final ordinary-loader DiT tensors, joins or atomically publishes an
immutable content-addressed entry, validates the mmap result, and returns a
fail-closed host-weight plan. Reusable identity, hashing, locking, and file
primitives live in :mod:`host_weight_cache_utils`.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
import time
import uuid
from collections.abc import Mapping, Sequence
from itertools import chain
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.model_loader.host_weight_cache_utils import (
    HostWeightCacheError,
    IdentityNormalizationError,
    RuntimeTensor,
    canonical_json,
    canonicalize_existing_local_path,
    exclusive_lock,
    file_digest,
    fsync_directory,
    fsync_file,
    implementation_fingerprint,
    normalize_identity,
    remove_stale_temps,
    runtime_content_digest,
    split_shards,
    tensor_metadata,
)
from vllm_omni.diffusion.model_loader.host_weight_plan import (
    HostWeightPlan,
    HostWeightPlanResult,
    TensorBinding,
)

logger = init_logger(__name__)

# Internal cache-format and I/O defaults, not environment variables. Public
# configuration may override the root and lock timeout; chunk sizes stay private.
# Bump the schema to isolate incompatible manifest layouts.
HOST_WEIGHT_CACHE_SCHEMA_VERSION = 1
# Keep the default shared across local stage replicas.
DEFAULT_HOST_WEIGHT_CACHE_ROOT = "~/.cache/vllm-omni/dlo-host-weights"
# Bound waits for another publisher instead of hanging initialization forever.
DEFAULT_LOCK_TIMEOUT_SECONDS = 600.0
# Avoid one model-sized output file and bound hashing/read working sets.
DEFAULT_SHARD_SIZE_BYTES = 5 * 1024**3

_SUPPORTED_DTYPES = {
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}
for _dtype_name in ("float8_e4m3fn", "float8_e5m2"):
    if (_dtype := getattr(torch, _dtype_name, None)) is not None:
        _SUPPORTED_DTYPES.add(_dtype)


def default_host_weight_cache_root() -> str:
    """Return the default root, intentionally independent of VLLM_CACHE_ROOT."""
    return os.path.expanduser(DEFAULT_HOST_WEIGHT_CACHE_ROOT)


def _persistent_buffer(module: nn.Module, local_name: str) -> bool:
    parent_path, _, leaf_name = local_name.rpartition(".")
    owner = module.get_submodule(parent_path)
    return leaf_name not in owner._non_persistent_buffers_set


def _named_parameters_with_duplicates(module: nn.Module):
    try:
        return module.named_parameters(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with old torch
        return module.named_parameters()


def _named_buffers_with_duplicates(module: nn.Module):
    try:
        return module.named_buffers(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with old torch
        return module.named_buffers()


def _resolve_pipeline_tensor(pipeline: nn.Module, runtime_name: str) -> torch.Tensor | None:
    parent_path, _, leaf_name = runtime_name.rpartition(".")
    parent = pipeline.get_submodule(parent_path)
    tensor = parent._parameters.get(leaf_name)
    if tensor is None:
        tensor = parent._buffers.get(leaf_name)
    return tensor


def _collect_runtime_tensors(
    pipeline: nn.Module,
    dit_modules: Sequence[tuple[str, nn.Module]],
) -> list[RuntimeTensor]:
    """Collect a complete, safely remappable view of final CPU DiT tensors.

    Parameters and persistent buffers must resolve to the same objects through
    the owning pipeline, occupy complete contiguous CPU storages, use supported
    dtypes, and have no aliases inside or outside the discovered DiT modules.
    The stable name ordering is reused by content hashing and publication.
    """
    records: dict[str, RuntimeTensor] = {}
    for dit_name, dit_module in dit_modules:
        for local_name, tensor in _named_parameters_with_duplicates(dit_module):
            runtime_name = f"{dit_name}.{local_name}"
            candidate = RuntimeTensor(runtime_name, tensor, "parameter")
            existing = records.get(runtime_name)
            if existing is not None and existing.tensor is not tensor:
                raise HostWeightCacheError(
                    "ambiguous_ownership",
                    f"multiple DiT tensors resolve to {runtime_name!r}",
                )
            records[runtime_name] = candidate
        for local_name, tensor in _named_buffers_with_duplicates(dit_module):
            if not _persistent_buffer(dit_module, local_name):
                continue
            runtime_name = f"{dit_name}.{local_name}"
            candidate = RuntimeTensor(runtime_name, tensor, "buffer")
            existing = records.get(runtime_name)
            if existing is not None and existing.tensor is not tensor:
                raise HostWeightCacheError(
                    "ambiguous_ownership",
                    f"multiple DiT tensors resolve to {runtime_name!r}",
                )
            records[runtime_name] = candidate

    if not records:
        raise HostWeightCacheError(
            "ambiguous_ownership",
            "no DiT parameters or persistent buffers were discovered",
        )

    storage_owners: dict[tuple[int, int], str] = {}
    for record in records.values():
        tensor = record.tensor
        try:
            pipeline_tensor = _resolve_pipeline_tensor(pipeline, record.name)
        except AttributeError as exc:
            raise HostWeightCacheError(
                "ambiguous_ownership",
                f"DiT tensor {record.name!r} is not owned by the pipeline",
            ) from exc
        if pipeline_tensor is not tensor:
            raise HostWeightCacheError(
                "ambiguous_ownership",
                f"DiT tensor {record.name!r} does not resolve to the discovered object",
            )
        if tensor.device.type != "cpu" or tensor.is_meta:
            raise HostWeightCacheError(
                "unsupported_tensor",
                f"{record.name!r} must be a materialized CPU tensor, got {tensor.device}",
            )
        if tensor.layout != torch.strided:
            raise HostWeightCacheError(
                "unsupported_tensor",
                f"{record.name!r} uses unsupported layout {tensor.layout}",
            )
        if not tensor.is_contiguous():
            raise HostWeightCacheError(
                "unsupported_tensor",
                f"{record.name!r} is non-contiguous with stride {tensor.stride()}",
            )
        if tensor.dtype not in _SUPPORTED_DTYPES:
            raise HostWeightCacheError(
                "unsupported_tensor",
                f"{record.name!r} uses unsupported dtype {tensor.dtype}",
            )
        if hasattr(tensor, "to_local"):
            raise HostWeightCacheError(
                "unsupported_tensor",
                f"{record.name!r} is a distributed tensor",
            )

        tensor_nbytes = tensor.numel() * tensor.element_size()
        storage = tensor.untyped_storage()
        if tensor.storage_offset() != 0 or storage.nbytes() != tensor_nbytes:
            raise HostWeightCacheError(
                "unsupported_tensor",
                f"{record.name!r} is a view into a larger storage",
            )
        if tensor_nbytes:
            storage_id = (storage.data_ptr(), storage.nbytes())
            owner = storage_owners.setdefault(storage_id, record.name)
            if owner != record.name:
                raise HostWeightCacheError(
                    "unsupported_alias",
                    f"{record.name!r} shares storage with {owner!r}",
                )

    # The ownership boundary is the complete pipeline, not only the discovered
    # DiT traversal. Reject a DiT storage that is also registered by an encoder,
    # VAE, resident component, or non-persistent buffer.
    for pipeline_name, tensor in chain(
        _named_parameters_with_duplicates(pipeline),
        _named_buffers_with_duplicates(pipeline),
    ):
        if tensor.device.type != "cpu" or tensor.is_meta or tensor.numel() == 0:
            continue
        storage = tensor.untyped_storage()
        owner = storage_owners.get((storage.data_ptr(), storage.nbytes()))
        if owner is not None and owner != pipeline_name:
            raise HostWeightCacheError(
                "unsupported_alias",
                f"cached tensor {owner!r} shares storage with pipeline tensor {pipeline_name!r}",
            )

    return [records[name] for name in sorted(records)]


def _publish_entry(
    *,
    cache_root: Path,
    entry_dir: Path,
    cache_key: str,
    layout_identity: dict[str, Any],
    content_digest: str,
    records: Sequence[RuntimeTensor],
    max_shard_bytes: int,
) -> None:
    total_nbytes = sum(record.tensor.numel() * record.tensor.element_size() for record in records)
    free_bytes = shutil.disk_usage(cache_root).free
    required_bytes = total_nbytes + max(64 * 1024**2, total_nbytes // 100)
    if free_bytes < required_bytes:
        raise HostWeightCacheError(
            "insufficient_disk",
            f"host weight cache needs {required_bytes} bytes but only {free_bytes} are free under {cache_root}",
        )

    tmp_root = cache_root / ".tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = tmp_root / f"{cache_key}.{os.getpid()}.{uuid.uuid4().hex}"
    temp_dir.mkdir()
    try:
        shards = split_shards(records, max_shard_bytes)
        tensor_manifest: dict[str, dict[str, Any]] = {}
        file_manifest: dict[str, dict[str, Any]] = {}
        shard_count = len(shards)
        for index, shard in enumerate(shards, start=1):
            filename = f"model-{index:05d}-of-{shard_count:05d}.safetensors"
            path = temp_dir / filename
            save_file(
                {record.name: record.tensor.detach() for record in shard},
                str(path),
                metadata={"format": "pt", "dlo_host_weight_cache_schema": str(HOST_WEIGHT_CACHE_SCHEMA_VERSION)},
            )
            fsync_file(path)
            file_manifest[filename] = {
                "size": path.stat().st_size,
                "sha256": file_digest(path),
            }
            for record in shard:
                tensor_manifest[record.name] = {
                    **tensor_metadata(record),
                    "file": filename,
                    "storage_key": record.name,
                }

        manifest = {
            "schema_version": HOST_WEIGHT_CACHE_SCHEMA_VERSION,
            "cache_key": cache_key,
            "layout_identity": layout_identity,
            "runtime_content_sha256": content_digest,
            "total_tensor_bytes": total_nbytes,
            "created_at_unix_seconds": time.time(),
            "writer_pid": os.getpid(),
            "files": file_manifest,
            "tensors": tensor_manifest,
        }
        manifest_path = temp_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        fsync_directory(temp_dir)

        invalid_dir: Path | None = None
        if entry_dir.exists():
            invalid_dir = tmp_root / f"{cache_key}.invalid.{os.getpid()}.{uuid.uuid4().hex}"
            os.replace(entry_dir, invalid_dir)
        os.replace(temp_dir, entry_dir)
        fsync_directory(entry_dir.parent)
        if invalid_dir is not None:
            shutil.rmtree(invalid_dir, ignore_errors=True)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _validate_entry(
    *,
    entry_dir: Path,
    cache_key: str,
    layout_identity: dict[str, Any],
    content_digest: str,
    expected_records: Sequence[RuntimeTensor],
) -> HostWeightPlan:
    manifest_path = entry_dir / "manifest.json"
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as exc:
        raise HostWeightCacheError("entry_rejected", f"cannot read {manifest_path}: {exc}") from exc

    if manifest.get("schema_version") != HOST_WEIGHT_CACHE_SCHEMA_VERSION:
        raise HostWeightCacheError("entry_rejected", "host weight cache schema mismatch")
    if manifest.get("cache_key") != cache_key:
        raise HostWeightCacheError("entry_rejected", "host weight cache key mismatch")
    if manifest.get("layout_identity") != layout_identity:
        raise HostWeightCacheError("entry_rejected", "host weight cache layout identity mismatch")
    if manifest.get("runtime_content_sha256") != content_digest:
        raise HostWeightCacheError("entry_rejected", "host weight cache content identity mismatch")

    expected_metadata = {record.name: tensor_metadata(record) for record in expected_records}
    tensor_manifest = manifest.get("tensors")
    file_manifest = manifest.get("files")
    if not isinstance(tensor_manifest, dict) or not isinstance(file_manifest, dict):
        raise HostWeightCacheError("entry_rejected", "host weight cache manifest has invalid sections")
    if set(tensor_manifest) != set(expected_metadata):
        raise HostWeightCacheError("entry_rejected", "host weight cache tensor names differ from the loaded DiT")

    for filename, metadata in file_manifest.items():
        if Path(filename).name != filename or not isinstance(metadata, dict):
            raise HostWeightCacheError("entry_rejected", f"invalid cache filename {filename!r}")
        path = entry_dir / filename
        try:
            stat = path.stat()
        except OSError as exc:
            raise HostWeightCacheError("entry_rejected", f"missing cache shard {path}: {exc}") from exc
        if stat.st_size != metadata.get("size") or file_digest(path) != metadata.get("sha256"):
            raise HostWeightCacheError("entry_rejected", f"content digest mismatch for cache shard {path}")

    bindings: dict[str, TensorBinding] = {}
    mapped_records: list[RuntimeTensor] = []
    with contextlib.ExitStack() as stack:
        handles = {
            filename: stack.enter_context(safe_open(entry_dir / filename, framework="pt", device="cpu"))
            for filename in file_manifest
        }
        expected_keys_by_file: dict[str, set[str]] = {filename: set() for filename in file_manifest}
        for record in expected_records:
            stored = tensor_manifest.get(record.name)
            if not isinstance(stored, dict):
                raise HostWeightCacheError("entry_rejected", f"invalid tensor metadata for {record.name!r}")
            filename = stored.get("file")
            storage_key = stored.get("storage_key")
            if filename not in handles or not isinstance(storage_key, str):
                raise HostWeightCacheError("entry_rejected", f"invalid storage binding for {record.name!r}")
            actual_metadata = {key: stored.get(key) for key in expected_metadata[record.name]}
            if actual_metadata != expected_metadata[record.name]:
                raise HostWeightCacheError("entry_rejected", f"metadata mismatch for {record.name!r}")
            expected_keys_by_file[filename].add(storage_key)
            try:
                tensor = handles[filename].get_tensor(storage_key)
            except Exception as exc:
                raise HostWeightCacheError(
                    "entry_rejected",
                    f"cannot map {record.name!r} from {filename}: {exc}",
                ) from exc
            if tuple(tensor.shape) != tuple(record.tensor.shape) or tensor.dtype != record.tensor.dtype:
                raise HostWeightCacheError("entry_rejected", f"mapped metadata mismatch for {record.name!r}")
            mapped_records.append(RuntimeTensor(record.name, tensor, record.kind))
            bindings[record.name] = TensorBinding(storage_key=storage_key, file_path=str(entry_dir / filename))

        for filename, expected_keys in expected_keys_by_file.items():
            if set(handles[filename].keys()) != expected_keys:
                raise HostWeightCacheError("entry_rejected", f"unexpected tensor keys in cache shard {filename}")

        if runtime_content_digest(mapped_records) != content_digest:
            raise HostWeightCacheError("entry_rejected", "mapped runtime tensor content does not match the publisher")

    return HostWeightPlan(
        backing_kind="host_weight_cache",
        bindings=bindings,
        runtime_layout_key=cache_key,
        post_load_complete=True,
        expected_file_digests={
            str(entry_dir / filename): metadata["sha256"] for filename, metadata in file_manifest.items()
        },
    )


def build_host_weight_cache_plan(
    pipeline: nn.Module,
    *,
    dit_modules: Sequence[tuple[str, nn.Module]],
    loader_type: type,
    cache_root: str | os.PathLike[str] | None,
    lock_timeout_seconds: float,
    max_shard_bytes: int,
    model_identity: str | None,
    revision: str | None,
    runtime_dtype: Any,
    load_format: str,
    loader_inputs: Mapping[str, Any],
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    sequence_parallel_guard: Mapping[str, Any],
    use_hsdp: bool,
    enable_expert_parallel: bool,
    quantization_config: Any,
    cfg_parallel_size: int,
    pipeline_parallel_size: int,
) -> HostWeightPlanResult:
    """Build a host-weight plan from the host weight cache after ordinary loading.

    This is the module's main entry point. It first rejects unsupported runtime
    layouts, then discovers and validates the final CPU DiT tensors and derives
    a content-addressed identity from their metadata, bytes, loader inputs, and
    parallelism guards. A valid existing entry is joined immediately; otherwise
    one process publishes the entry under a per-identity file lock and peers
    validate it after the lock is released.

    Success returns bindings to immutable safetensors mmap files. Any expected
    incompatibility, lock, publication, or validation failure returns a
    fail-closed result with no plan and a stable fallback code, leaving the
    already-loaded pipeline tensors intact.
    """
    if load_format == "dummy":
        return HostWeightPlanResult(
            None,
            "dummy/random weights do not have a reusable runtime identity",
            "unsupported_load_format",
        )
    if use_hsdp:
        return HostWeightPlanResult(
            None, "HSDP/DTensor layouts are not host weight cache compatible", "unsupported_hsdp"
        )
    if enable_expert_parallel:
        return HostWeightPlanResult(
            None,
            "expert-parallel weight ownership is outside host weight cache v1",
            "unsupported_expert_parallel",
        )
    if quantization_config is not None:
        return HostWeightPlanResult(
            None,
            "quantized runtime layouts are not supported by the first host weight cache version",
            "unsupported_quantization",
        )
    if cfg_parallel_size != 1:
        return HostWeightPlanResult(None, "CFG-parallel component ownership is not proven", "unsupported_cfg_parallel")
    if pipeline_parallel_size != 1:
        return HostWeightPlanResult(
            None, "pipeline-parallel ownership is outside host weight cache v1", "unsupported_pp"
        )
    if not math.isfinite(lock_timeout_seconds) or lock_timeout_seconds <= 0:
        return HostWeightPlanResult(None, "host weight cache lock timeout must be positive", "invalid_configuration")
    if max_shard_bytes <= 0:
        return HostWeightPlanResult(None, "host weight cache shard size must be positive", "invalid_configuration")

    try:
        records = _collect_runtime_tensors(pipeline, dit_modules)
        content_digest = runtime_content_digest(records)
        layout_identity = {
            "schema_version": HOST_WEIGHT_CACHE_SCHEMA_VERSION,
            "model": (canonicalize_existing_local_path(model_identity) if model_identity is not None else None),
            "revision": revision,
            "runtime_dtype": str(runtime_dtype),
            "load_format": load_format,
            "loader_inputs": normalize_identity(loader_inputs),
            "loader_implementation_sha256": implementation_fingerprint(loader_type, pipeline, dit_modules),
            "components": [name for name, _ in dit_modules],
            "tensor_parallel_size": tensor_parallel_size,
            "tensor_parallel_rank": tensor_parallel_rank,
            "sequence_parallel_guard": normalize_identity(sequence_parallel_guard),
            "runtime_content_sha256": content_digest,
        }
        cache_key = hashlib.sha256(canonical_json(layout_identity)).hexdigest()
        root = Path(cache_root or default_host_weight_cache_root()).expanduser().resolve()
        entries_root = root / f"v{HOST_WEIGHT_CACHE_SCHEMA_VERSION}"
        entry_dir = entries_root / cache_key
        entries_root.mkdir(parents=True, exist_ok=True)

        if entry_dir.is_dir():
            try:
                plan = _validate_entry(
                    entry_dir=entry_dir,
                    cache_key=cache_key,
                    layout_identity=layout_identity,
                    content_digest=content_digest,
                    expected_records=records,
                )
                logger.info("Reusing DLO host weight cache entry %s from %s", cache_key[:12], entry_dir)
                return HostWeightPlanResult(plan)
            except HostWeightCacheError as exc:
                logger.warning("DLO host weight cache entry %s will be rebuilt: %s", cache_key[:12], exc)

        lock_path = root / ".locks" / f"{cache_key}.lock"
        with exclusive_lock(lock_path, lock_timeout_seconds):
            remove_stale_temps(root, cache_key)
            if entry_dir.is_dir():
                try:
                    plan = _validate_entry(
                        entry_dir=entry_dir,
                        cache_key=cache_key,
                        layout_identity=layout_identity,
                        content_digest=content_digest,
                        expected_records=records,
                    )
                    logger.info("Joined DLO host weight cache entry %s from %s", cache_key[:12], entry_dir)
                    return HostWeightPlanResult(plan)
                except HostWeightCacheError:
                    pass

            _publish_entry(
                cache_root=root,
                entry_dir=entry_dir,
                cache_key=cache_key,
                layout_identity=layout_identity,
                content_digest=content_digest,
                records=records,
                max_shard_bytes=max_shard_bytes,
            )
            plan = _validate_entry(
                entry_dir=entry_dir,
                cache_key=cache_key,
                layout_identity=layout_identity,
                content_digest=content_digest,
                expected_records=records,
            )
            logger.info(
                "Published DLO host weight cache entry %s with %d tensors at %s",
                cache_key[:12],
                len(records),
                entry_dir,
            )
            return HostWeightPlanResult(plan)
    except HostWeightCacheError as exc:
        return HostWeightPlanResult(None, str(exc), exc.code)
    except IdentityNormalizationError as exc:
        return HostWeightPlanResult(None, str(exc), "unstable_identity")
    except Exception as exc:
        return HostWeightPlanResult(None, f"host weight cache operation failed: {exc}", "cache_operation_failed")


__all__ = [
    "DEFAULT_LOCK_TIMEOUT_SECONDS",
    "DEFAULT_HOST_WEIGHT_CACHE_ROOT",
    "DEFAULT_SHARD_SIZE_BYTES",
    "HOST_WEIGHT_CACHE_SCHEMA_VERSION",
    "build_host_weight_cache_plan",
    "default_host_weight_cache_root",
]
