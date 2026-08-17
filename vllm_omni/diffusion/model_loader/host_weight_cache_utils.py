# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reusable identity, hashing, and file helpers for the DLO host weight cache."""

from __future__ import annotations

import contextlib
import dataclasses
import errno
import fcntl
import hashlib
import inspect
import json
import os
import shutil
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, TypeAlias, TypeVar

import torch
from torch import nn

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
PathIdentity: TypeAlias = str | os.PathLike[str] | os.PathLike[bytes]
_HASH_CHUNK_BYTES = 64 * 1024**2
_FILE_HASH_CHUNK_BYTES = 8 * 1024**2


class IdentityNormalizationError(ValueError):
    """Raised when a cache identity contains a process-unstable value."""


@dataclass(frozen=True)
class RuntimeTensor:
    """One final runtime tensor and its manifest identity."""

    name: str
    tensor: torch.Tensor
    kind: str


class HostWeightCacheError(RuntimeError):
    """Expected cache failure that should fall back with a stable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class _HashUpdater(Protocol):
    def update(self, data: bytes | bytearray | memoryview) -> object: ...


_RuntimeTensorT = TypeVar("_RuntimeTensorT", bound=RuntimeTensor)


def _type_identity(value: object) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def canonicalize_existing_local_path(value: PathIdentity) -> str:
    """Resolve equivalent local paths while leaving model repository IDs alone."""
    original = os.fsdecode(os.fspath(value))
    candidate = Path(original).expanduser()
    try:
        if candidate.exists():
            return str(candidate.resolve())
    except OSError:
        pass
    return original


def normalize_identity(value: object) -> JsonValue:
    """Convert supported loader inputs into deterministic JSON-compatible data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return {"enum": _type_identity(value), "name": value.name}
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, os.PathLike):
        return canonicalize_existing_local_path(value)
    if isinstance(value, type):
        return _type_identity(value)
    if dataclasses.is_dataclass(value):
        return normalize_identity(dataclasses.asdict(value))

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json")
        except (TypeError, ValueError):
            dumped = model_dump()
        return normalize_identity(dumped)

    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise IdentityNormalizationError("host weight cache identity mappings require string keys")
            normalized[key] = normalize_identity(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (set, frozenset)):
        return sorted((normalize_identity(item) for item in value), key=canonical_json)
    if isinstance(value, (list, tuple)):
        return [normalize_identity(item) for item in value]

    raise IdentityNormalizationError(
        f"host weight cache identity does not support values of type {_type_identity(value)}"
    )


def canonical_json(value: object) -> bytes:
    """Serialize a supported identity with stable ordering and separators."""
    return json.dumps(
        normalize_identity(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()


def implementation_fingerprint(
    loader_type: type,
    pipeline: nn.Module,
    dit_modules: Sequence[tuple[str, nn.Module]],
) -> str:
    """Hash relevant loader and module implementations for cache identity."""
    objects: list[object] = [loader_type, type(pipeline)]
    load_weights = getattr(type(pipeline), "load_weights", None)
    if load_weights is not None:
        objects.append(load_weights)
    for _, dit_module in dit_modules:
        objects.append(type(dit_module))
        post_load = getattr(type(dit_module), "post_load_weights", None)
        if post_load is not None:
            objects.append(post_load)

    digest = hashlib.sha256()
    identities: set[str] = set()
    for obj in objects:
        identity = f"{getattr(obj, '__module__', '')}.{getattr(obj, '__qualname__', type(obj).__qualname__)}"
        if identity in identities:
            continue
        identities.add(identity)
        digest.update(identity.encode())
        try:
            digest.update(inspect.getsource(obj).encode())
        except (OSError, TypeError):
            pass
    return digest.hexdigest()


def tensor_metadata(record: RuntimeTensor) -> dict[str, JsonValue]:
    """Return stable manifest metadata for one runtime tensor."""
    tensor = record.tensor
    return {
        "kind": record.kind,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "nbytes": tensor.numel() * tensor.element_size(),
        "layout": "contiguous",
    }


def _update_hash_with_tensor(digest: _HashUpdater, tensor: torch.Tensor) -> None:
    byte_view = tensor.detach().reshape(-1).view(torch.uint8)
    for offset in range(0, byte_view.numel(), _HASH_CHUNK_BYTES):
        chunk = byte_view[offset : offset + _HASH_CHUNK_BYTES]
        digest.update(memoryview(chunk.numpy()))


def runtime_content_digest(records: Sequence[RuntimeTensor]) -> str:
    """Hash ordered runtime tensor metadata and contents."""
    digest = hashlib.sha256()
    for record in records:
        digest.update(canonical_json({"name": record.name, **tensor_metadata(record)}))
        _update_hash_with_tensor(digest, record.tensor)
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    """Hash one cache file with a bounded read working set."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_FILE_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def exclusive_lock(path: Path, timeout_seconds: float) -> Iterator[None]:
    """Acquire one non-blocking file lock within a bounded wait."""
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    with path.open("a+b") as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise HostWeightCacheError("lock_failed", f"failed to lock {path}: {exc}") from exc
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise HostWeightCacheError(
                        "lock_timeout",
                        f"timed out after {timeout_seconds:g}s waiting for host weight cache writer {path.name}",
                    ) from exc
                time.sleep(min(0.2, remaining))
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def fsync_file(path: Path) -> None:
    """Persist one cache file before publication."""
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    """Persist directory entries needed for atomic publication."""
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def split_shards(
    records: Sequence[_RuntimeTensorT],
    max_shard_bytes: int,
) -> list[list[_RuntimeTensorT]]:
    """Split ordered tensors without exceeding the target when possible."""
    shards: list[list[_RuntimeTensorT]] = []
    current: list[_RuntimeTensorT] = []
    current_bytes = 0
    for record in records:
        nbytes = record.tensor.numel() * record.tensor.element_size()
        if current and current_bytes + nbytes > max_shard_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(record)
        current_bytes += nbytes
    if current:
        shards.append(current)
    return shards


def remove_stale_temps(cache_root: Path, cache_key: str) -> None:
    """Remove abandoned temporary entries for one cache identity."""
    temp_root = cache_root / ".tmp"
    if not temp_root.is_dir():
        return
    for path in temp_root.glob(f"{cache_key}.*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


__all__ = [
    "IdentityNormalizationError",
    "JsonValue",
    "HostWeightCacheError",
    "RuntimeTensor",
    "canonical_json",
    "canonicalize_existing_local_path",
    "exclusive_lock",
    "file_digest",
    "fsync_directory",
    "fsync_file",
    "implementation_fingerprint",
    "normalize_identity",
    "remove_stale_temps",
    "runtime_content_digest",
    "split_shards",
    "tensor_metadata",
]
