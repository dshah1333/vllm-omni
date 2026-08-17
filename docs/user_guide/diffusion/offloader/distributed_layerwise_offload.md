# Distributed Layerwise Offloading

Distributed layerwise offloading (DLO) streams DiT blocks through two device
buffers instead of keeping the complete DiT in HBM. It supports sharded host
weights with AllGather or independently streamed rank-local weights. The
no-AllGather path can optionally share node-local host weights through mmap.

See the [shared DLO component architecture](../../../design/feature/offloader/distributed_layerwise_offload.md#architecture)
for ownership and lifecycle coordination. The mode-specific internals are
documented separately in the
[AllGather design](../../../design/feature/offloader/distributed_layerwise_offload.md#allgather-design)
and
[no-AllGather design](../../../design/feature/offloader/distributed_layerwise_offload.md#no-allgather-design).

## Choose a mode

| Mode | Use when | Host weights | Runtime synchronization |
| --- | --- | --- | --- |
| DLO AllGather (default) | DP ranks execute the same block path in lockstep | About `1 / dp_size` per rank | DLO weight AllGather |
| no-AllGather | Ranks or engines must schedule independently | Complete rank-local layout | No DLO weight collective |
| no-AllGather + host weight cache | Equivalent independent workers share one node | Shared final mmap layout per TP coordinate | No DLO weight collective |
| host weight cache + host registration | The platform supports registration and recurrent staging is too expensive | Shared registered mmap layout | No DLO weight collective |

AllGather is normally the best choice for synchronized DP. Host weight cache mode
targets independently scheduled replicas, especially repeated TP engines on
one node.

## Usage

```bash
# Sharded host weights with DLO AllGather
vllm serve /path/to/model --omni \
  --enable-distributed-layerwise-offload \
  --data-parallel-size 4

# Independently streamed rank-local weights
vllm serve /path/to/model --omni \
  --enable-distributed-layerwise-offload \
  --dlo-no-use-allgather

# Share final host weights and register up to 80 GiB per worker for direct H2D
vllm serve /path/to/model --omni \
  --enable-distributed-layerwise-offload \
  --tensor-parallel-size 2 \
  --dlo-no-use-allgather \
  --dlo-host-weight-cache-pin-limit-gib 80
```

## Relevant options

| Flag | Meaning | Default |
| --- | --- | --- |
| `--enable-distributed-layerwise-offload` | Enable DLO | `false` |
| `--dlo-no-use-allgather` | Stream complete rank-local blocks independently | `false` |
| `--dlo-host-weight-cache-pin-limit-gib GIB` | Per-worker registration budget; zero uses bounded host staging | `0` |
| `--dlo-resident-layers N` | Keep eligible leading DiT blocks resident in HBM | `0` |

No-AllGather DLO automatically uses a compatible checkpoint mmap plan, or
builds/joins the final-layout host weight cache after ordinary loading when a
checkpoint plan is unavailable. The cache uses
`~/.cache/vllm-omni/dlo-host-weights`. Programmatic configuration may override
`dlo_host_weight_cache_dir` and the writer lock timeout; these advanced storage
controls are intentionally not separate CLI flags.

## Operational notes

- All workers that should share must see the same local, disk-backed cache
  directory. Do not use tmpfs or a cross-node filesystem for this version.
- Cache entries are immutable and validated before use. A cache or registration
  failure keeps the ordinary loader weights or falls back to bounded staging.
- A positive registration budget must cover the complete page-aligned mapping
  reported in the worker log. Registration is all-or-nothing.
- CUDA is the first registration backend. Platforms without an equivalent
  implementation continue to use bounded staging.
- Registration is process-local but does not duplicate the underlying file
  pages. Each worker must still satisfy its platform and OS page-locking limits.
- Shutdown unregisters host ranges before closing their mmap handles.
- Host weight cache v1 has no automatic eviction. Stop all users of an entry
  before deleting it.

## Scheduling constraints

AllGather ranks must request blocks in the same collective order. Concurrent DP
requests may use different prompts, but they must follow the same denoising and
block-execution path and set the same explicit `num_inference_steps`.

No-AllGather workers do not have this DLO lockstep requirement. Ordinary TP or
SP model collectives still synchronize ranks within each engine.

## Limitations

- Direct checkpoint mmap currently requires TP1. The host weight cache can share
  final ordinary-loader layouts between matching TP coordinates at TP1 or TP>1.
- Host weight cache v1 rejects quantized, non-contiguous, aliased/tied,
  device-only, HSDP, expert-parallel, CFG-parallel, and PP layouts.
- HSDP plus DLO AllGather is unsupported, and host weight cache v1 rejects HSDP.
- The host weight cache changes steady-state host backing, not startup loading:
  each worker still performs ordinary loading and full content validation.
- Skip-load-on-hit, cache eviction, and cross-node sharing remain follow-up
  work in
  [RFC #6195](https://github.com/vllm-project/vllm-omni/issues/6195).

See the [Cosmos3 DistOffload recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/cosmos3/Cosmos3-DistOffload.md)
for an end-to-end example.
