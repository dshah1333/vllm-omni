# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA implementation of read-only host-mapping registration."""

from __future__ import annotations

import ctypes
import mmap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from .host_registration import (
    HostRegistrationBudgetError,
    HostRegistrationCleanupError,
    HostRegistrationError,
)

_CUDA_HOST_REGISTER_READ_ONLY = 0x08
_CUDA_DEVICE_ATTRIBUTE_HOST_REGISTER_READ_ONLY_SUPPORTED = 113

CudaHostRegistrationError = HostRegistrationError
CudaHostRegistrationBudgetError = HostRegistrationBudgetError
CudaHostRegistrationCleanupError = HostRegistrationCleanupError


class _CudaRuntime(Protocol):
    def cudaHostRegister(self, address: int, size: int, flags: int) -> int: ...

    def cudaHostUnregister(self, address: int) -> int: ...

    def cudaGetErrorString(self, error: int) -> str | bytes: ...


@dataclass(frozen=True)
class _AddressRange:
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


def _coalesce_ranges(
    ranges: Sequence[tuple[int, int]],
    page_size: int = mmap.PAGESIZE,
) -> tuple[_AddressRange, ...]:
    """Page-align and merge overlapping ranges from one backing mapping."""
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")

    aligned: list[_AddressRange] = []
    for start, size in ranges:
        if start <= 0 or size < 0:
            raise ValueError(f"invalid host range start={start}, size={size}")
        if size == 0:
            continue
        aligned_start = start - start % page_size
        end = start + size
        aligned_end = ((end + page_size - 1) // page_size) * page_size
        aligned.append(_AddressRange(aligned_start, aligned_end))

    merged: list[_AddressRange] = []
    for region in sorted(aligned, key=lambda item: (item.start, item.end)):
        if merged and region.start <= merged[-1].end:
            previous = merged[-1]
            merged[-1] = _AddressRange(previous.start, max(previous.end, region.end))
        else:
            merged.append(region)
    return tuple(merged)


def _tensor_regions(
    sources_by_mapping: Mapping[str, Sequence[torch.Tensor]],
) -> tuple[_AddressRange, ...]:
    """Resolve storage spans without merging unrelated file mappings."""
    regions: list[_AddressRange] = []
    for mapping_name, tensors in sources_by_mapping.items():
        ranges: list[tuple[int, int]] = []
        for tensor in tensors:
            if tensor.device.type != "cpu":
                raise HostRegistrationError(
                    f"CUDA host registration requires CPU storage, but {mapping_name!r} contains {tensor.device}"
                )
            storage = tensor.untyped_storage()
            ranges.append((storage.data_ptr(), storage.nbytes()))
        regions.extend(_coalesce_ranges(ranges))
    return tuple(regions)


def _error_message(runtime: _CudaRuntime, error: int) -> str:
    try:
        message = runtime.cudaGetErrorString(error)
    except Exception:
        return str(error)
    if isinstance(message, bytes):
        return message.decode(errors="replace")
    return str(message)


def _consume_last_cuda_error(expected_error: int) -> None:
    """Clear a handled CUDA Runtime error before returning to PyTorch.

    PyTorch's ``torch.cuda.cudart()`` binding exposes host registration but not
    ``cudaGetLastError``. Resolve the already-loaded process symbol so this call
    clears the same CUDA Runtime instance that produced the registration error.
    A different pending error is not safe to hide behind the staging fallback.
    """
    try:
        get_last_error = ctypes.CDLL(None).cudaGetLastError
        get_last_error.argtypes = []
        get_last_error.restype = ctypes.c_int
        pending_error = int(get_last_error())
    except (AttributeError, OSError) as exc:
        raise HostRegistrationCleanupError("cannot clear CUDA's pending error after host-registration failure") from exc

    if pending_error not in (0, expected_error):
        raise HostRegistrationCleanupError(
            f"host registration returned CUDA error {expected_error}, but cudaGetLastError reported {pending_error}"
        )


def _handled_error_message(runtime: _CudaRuntime, error: int) -> str:
    """Format and consume one nonzero CUDA Runtime return code."""
    error_code = int(error)
    message = _error_message(runtime, error)
    try:
        _consume_last_cuda_error(error_code)
    except HostRegistrationCleanupError as exc:
        raise HostRegistrationCleanupError(f"{message}; {exc}") from exc
    return message


def _supports_read_only_host_registration(runtime: _CudaRuntime) -> bool:
    """Query support required by immutable file-backed cache mappings."""
    try:
        get_attribute = getattr(runtime, "cudaDeviceGetAttribute", None)
        if get_attribute is None:
            get_attribute = ctypes.CDLL(None).cudaDeviceGetAttribute
            get_attribute.argtypes = [
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
            ]
            get_attribute.restype = ctypes.c_int
    except (AttributeError, OSError) as exc:
        raise HostRegistrationError("cannot query CUDA read-only host-registration support") from exc

    supported = ctypes.c_int()
    try:
        error = get_attribute(
            ctypes.byref(supported),
            _CUDA_DEVICE_ATTRIBUTE_HOST_REGISTER_READ_ONLY_SUPPORTED,
            torch.accelerator.current_device_index(),
        )
    except Exception as exc:
        raise HostRegistrationError(f"cannot query CUDA read-only host-registration support: {exc}") from exc
    if int(error) != 0:
        raise HostRegistrationError(
            f"cudaDeviceGetAttribute(read-only host registration) failed: {_handled_error_message(runtime, error)}"
        )
    return bool(supported.value)


class CudaHostRegistration:
    """Own CUDA registrations for already-existing file-backed CPU tensors."""

    def __init__(
        self,
        runtime: _CudaRuntime,
        regions: tuple[_AddressRange, ...],
    ) -> None:
        self._runtime = runtime
        self._regions = regions
        self._closed = False

    @classmethod
    def create(
        cls,
        sources_by_mapping: Mapping[str, Sequence[torch.Tensor]],
        *,
        max_bytes: int,
    ) -> CudaHostRegistration:
        if max_bytes <= 0:
            raise HostRegistrationBudgetError("CUDA host-registration budget is disabled")
        regions = _tensor_regions(sources_by_mapping)
        total_bytes = sum(region.size for region in regions)
        if total_bytes > max_bytes:
            raise HostRegistrationBudgetError(
                f"mapped host ranges need {total_bytes} bytes, exceeding the {max_bytes}-byte registration budget"
            )
        if not regions:
            raise HostRegistrationError("no non-empty host ranges were available for registration")
        if not torch.cuda.is_available():
            raise HostRegistrationError("CUDA is not available")

        try:
            runtime = torch.cuda.cudart()
        except Exception as exc:
            raise HostRegistrationError(f"cannot access the CUDA runtime: {exc}") from exc
        if not _supports_read_only_host_registration(runtime):
            raise HostRegistrationError(
                "CUDA device does not support read-only host registration required by immutable "
                "host weight cache mappings"
            )

        registered: list[_AddressRange] = []
        try:
            for region in regions:
                error = runtime.cudaHostRegister(
                    region.start,
                    region.size,
                    _CUDA_HOST_REGISTER_READ_ONLY,
                )
                if int(error) != 0:
                    raise HostRegistrationError(
                        "cudaHostRegister(read-only) failed for "
                        f"[{region.start:#x}, {region.end:#x}): {_handled_error_message(runtime, error)}"
                    )
                registered.append(region)

            unpinned = [
                mapping_name
                for mapping_name, tensors in sources_by_mapping.items()
                if any(tensor.numel() and not tensor.is_pinned() for tensor in tensors)
            ]
            if unpinned:
                raise HostRegistrationError(
                    f"CUDA registration succeeded but PyTorch did not recognize pinned storage for {unpinned[:3]}"
                )
        except Exception as exc:
            rollback_errors: list[str] = []
            for region in reversed(registered):
                try:
                    error = runtime.cudaHostUnregister(region.start)
                    if int(error) != 0:
                        rollback_errors.append(
                            f"cudaHostUnregister({region.start:#x}) failed: {_handled_error_message(runtime, error)}"
                        )
                except Exception as rollback_exc:
                    rollback_errors.append(f"cudaHostUnregister({region.start:#x}) raised: {rollback_exc}")
            if rollback_errors:
                raise HostRegistrationCleanupError(
                    f"CUDA host registration failed ({exc}); rollback errors: {rollback_errors[:3]}"
                ) from exc
            if isinstance(exc, HostRegistrationError):
                raise
            raise HostRegistrationError(f"CUDA host registration raised: {exc}") from exc

        return cls(runtime, regions)

    @property
    def total_bytes(self) -> int:
        return sum(region.size for region in self._regions)

    @property
    def region_count(self) -> int:
        return len(self._regions)

    def close(self) -> list[str]:
        """Unregister every range, returning errors after best-effort cleanup."""
        if self._closed:
            return []
        errors: list[str] = []
        failed: list[_AddressRange] = []
        for region in reversed(self._regions):
            try:
                error = self._runtime.cudaHostUnregister(region.start)
                if int(error) != 0:
                    errors.append(
                        f"cudaHostUnregister({region.start:#x}) failed: {_handled_error_message(self._runtime, error)}"
                    )
                    failed.append(region)
            except Exception as exc:
                errors.append(f"cudaHostUnregister({region.start:#x}) raised: {exc}")
                failed.append(region)
        self._regions = tuple(reversed(failed))
        self._closed = not self._regions
        return errors
