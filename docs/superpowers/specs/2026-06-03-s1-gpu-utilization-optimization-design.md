# S1 (512x512) GPU Utilization Optimization Design

## Problem

When training at S1 resolution (512x512, token_count=1024) on an RTX 5090 32GB with default preset and `torch_compile=true`, GPU utilization is ~70%. Higher resolutions reach ~95%. The computation itself is correct and all caches are in place — the bottleneck is in the training pipeline's data scheduling, not the algorithm.

Root cause: S1's forward/backward completes in a few milliseconds, but the data loading pipeline (disk I/O → CPU preparation → H2D transfer) is serialized with GPU computation. The GPU idles while waiting for the next batch.

## Solution Overview

Two-phase optimization, both purely additive and fully compatible with the existing `_EpochPrefetch` mechanism:

- **Phase A**: Increase DataLoader `num_workers` and `prefetch_factor` — configuration change only, zero code risk.
- **Phase B**: CUDA Stream double-buffering in the training loop — overlap H2D transfer with GPU computation.

Target: ~95% GPU utilization at S1, matching high-resolution behavior.

## Phase A: DataLoader Configuration Tuning

### Changes

| Parameter | Current | New | Rationale |
|-----------|---------|-----|-----------|
| `max_data_loader_n_workers` | 1 | 4 | 4 workers parallelize disk I/O and deserialization of `.npz` / `.safetensors` files |
| `dataloader_prefetch_factor` | 1 | 4 | Each worker prefetches 4 batches; total prefetch depth = 4×4 = 16 batches |

### Configuration architecture

There are **four training pathways**, each with independent configuration management, all converging on `train.py`'s argparse namespace:

| Path | Config source | Notes |
|------|--------------|-------|
| **CLI** | `configs/base.toml` → TOML merge chain → argparse overrides | `cli_args.py` defaults: `n_workers=1`, `prefetch=1` |
| **GUI** | `configs/gui-methods/<variant>.toml` → daemon → same as CLI | Shares same TOML merge chain via `gui-methods/` dir |
| **Workflow** | `workflow/stages/train.py` `_AUTO_DEFAULTS` + YAML schema → CLI args via subprocess | **Missing** `max_data_loader_n_workers` and `dataloader_prefetch_factor` |
| **ComfyUI** | `custom_nodes/comfyui-anima-trainer/node_defaults.toml` → daemon | **Intentionally sets** `max_data_loader_n_workers=0` |

### Files to modify

1. **`configs/base.toml`** — add `max_data_loader_n_workers = 4` and `dataloader_prefetch_factor = 4`. This covers CLI and GUI paths (both use the TOML merge chain).

2. **`workflow/stages/train.py`** — add `max_data_loader_n_workers` and `dataloader_prefetch_factor` to `_AUTO_DEFAULTS` with values `4` and `4`. This ensures the workflow path passes these values to `train.py` via CLI args.

3. **`workflow/schemas/train_common.yaml`** — expose `max_data_loader_n_workers` and `dataloader_prefetch_factor` in the `performance` section so workflow users can override them from the Web UI.

4. **`custom_nodes/comfyui-anima-trainer/node_defaults.toml`** — **do NOT change**. ComfyUI intentionally sets `max_data_loader_n_workers=0` because multi-process DataLoader causes issues in the ComfyUI process environment. This path is excluded from the optimization.

5. **`library/training/cli_args.py`** — no changes needed. The existing argparse defaults (`1`) serve as fallback when no TOML config overrides them.

### Memory impact

Each prefetched S1 batch ≈ 8MB (latent) + TE/PE embeddings. 16 slots ≈ 200-400MB RAM. Trivial for a 5090-class system.

### Interaction with `_EpochPrefetch`

`_EpochPrefetch` constructs its temporary DataLoader by reading attributes from the main DataLoader ([loop.py:399-414](file:///o:/loratool/anima_lora_fork/library/training/loop.py#L399-L414)). It inherits `num_workers`, `prefetch_factor`, and `persistent_workers` automatically. During the ~20-step prefetch window at epoch boundaries, there are temporarily 8 worker processes (4 main + 4 prefetch). RAM impact: +400MB temporary. After the prefetch iterator is consumed, the extra workers shut down.

No code changes needed in `_EpochPrefetch`.

## Phase B: CUDA Stream Double-Buffering

### Mechanism

Introduce a `_StepPrefetch` helper that uses a dedicated CUDA stream to asynchronously transfer the next batch to GPU while the current batch is being computed.

```
Timeline:
  Step N:   GPU compute(batch_N)  ←→  Stream_B: H2D(batch_N+1)
  Step N+1: GPU compute(batch_N+1) ←→  Stream_B: H2D(batch_N+2)
  ...
```

### Implementation

#### `_StepPrefetch` class (new, in `library/training/loop.py`)

```python
class _StepPrefetch:
    """Overlap H2D transfer of batch N+1 with GPU computation of batch N."""

    def __init__(self, device: torch.device):
        self._stream = torch.cuda.Stream(device=device)
        self._batch: Optional[Any] = None

    def submit(self, batch_cpu, device: torch.device) -> None:
        """Launch async H2D transfer on the prefetch stream."""
        self._stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(self._stream):
            self._batch = send_to_device(batch_cpu, device)

    def consume(self) -> Any:
        """Wait for transfer completion and return the GPU-resident batch."""
        batch = self._batch
        self._batch = None
        if batch is not None:
            torch.cuda.current_stream(batch.device).wait_stream(self._stream)
        return batch

    def invalidate(self) -> None:
        """Discard any pending transfer (used on error / epoch boundary)."""
        self._batch = None
```

#### Training loop integration (modify `_run_epoch_steps` in `library/training/loop.py`)

The double-buffer is injected at the batch-fetch points in both the prefetch path and the standard path.

**Prefetch path** (when `_EpochPrefetch` is active, [loop.py:520-573](file:///o:/loratool/anima_lora_fork/library/training/loop.py#L520-L573)):

```
Step 0: batch = first_batch (already on GPU from _EpochPrefetch)
        step_prefetch.submit(next(iterator), device)  # start async transfer
Step 1+: batch = step_prefetch.consume()  # wait for transfer
         step_prefetch.submit(next(iterator), device)  # pipeline next
         ... (StopIteration: don't submit, just consume last)
```

**Standard path** (no epoch prefetch, [loop.py:574-636](file:///o:/loratool/anima_lora_fork/library/training/loop.py#L574-L636)):

```
for step, batch_cpu in enumerate(dataloader):
    batch = step_prefetch.consume() or send_to_device(batch_cpu, device)
    # ... run_step ...
    # Peek next batch for async transfer (if not last step)
```

### CUDAGraphs compatibility

`torch.compiler.cudagraph_mark_step_begin()` must be called after `consume()` returns (data is guaranteed on default stream). The ordering:

```
consume() → wait_stream() → data on default stream
cudagraph_mark_step_begin() → set CUDAGraph boundary
forward(batch) → uses ready data
submit(next_batch) → async on Stream B (independent of CUDAGraph)
```

### `accelerator.accumulate()` compatibility

`accumulate()` is a pure-Python context manager that toggles `sync_gradients`. It has no CUDA stream interaction. Double-buffering operates outside the accumulate scope (at the batch-fetch level), so gradient accumulation semantics are unaffected.

### Interaction with `_EpochPrefetch`

Three interaction points, all safe:

1. **Epoch boundary handoff**: `prefetch.result()` calls `join()` on the background thread, guaranteeing the `first_batch` H2D transfer completes before the main loop resumes. `_StepPrefetch` starts from a cold state (no pending batch) at each epoch boundary.

2. **Background thread H2D vs. main loop Stream B**: The background thread's `send_to_device` runs on the thread's default CUDA stream. The main loop's `_StepPrefetch` uses a separate stream. Since `join()` provides a happens-before guarantee, there is no race condition.

3. **Iterator reuse**: The iterator returned by `_EpochPrefetch` is used by the main loop's `next()` calls. `submit()` is only called after a successful `next()`, so `StopIteration` never leaves a dangling async transfer.

### Error handling

On exception or `KeyboardInterrupt`, the outer `try/except` in `run_training_loop` calls `pending_prefetch.cancel()`. `_StepPrefetch.invalidate()` should be called in the same handler to discard any in-flight transfer. The training loop's existing error paths remain unchanged.

## Expected Results

| Metric | Current (S1) | After Phase A | After Phase A+B |
|--------|-------------|---------------|-----------------|
| GPU utilization | ~70% | ~85-90% | ~95%+ |
| Per-step GPU idle | ~15-20ms | ~5-10ms | ~0-2ms |
| Extra RAM | — | ~400MB | ~400MB |
| Extra VRAM | — | 0 | ~8MB (1 batch) |

## Non-goals

- **No batch_size increase** — user preference.
- **No gradient accumulation change** — user preference.
- **No LoKr bf16 optimization** — user preference (separate concern).
- **No changes to `_EpochPrefetch`** — it works correctly and is fully compatible.
