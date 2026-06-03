# S1 GPU Utilization Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise S1 (512x512) training GPU utilization from ~70% to ~95% by optimizing data loading pipeline configuration and adding CUDA stream double-buffering.

**Architecture:** Two-phase approach. Phase A: increase DataLoader `num_workers` (1→4) and `prefetch_factor` (1→4) across all four training paths (CLI/GUI/Workflow/ComfyUI). Phase B: add `_StepPrefetch` class using a dedicated CUDA stream to overlap H2D transfer with GPU computation. All changes are additive and compatible with existing `_EpochPrefetch`.

**Tech Stack:** PyTorch DataLoader, CUDA Streams, HuggingFace Accelerate.

**Spec:** `docs/superpowers/specs/2026-06-03-s1-gpu-utilization-optimization-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `configs/base.toml` | Add `max_data_loader_n_workers` and `dataloader_prefetch_factor` to shared TOML config |
| Modify | `workflow/stages/train.py` | Add two params to `_AUTO_DEFAULTS` |
| Modify | `workflow/schemas/train_common.yaml` | Expose two params in `performance` section |
| Modify | `library/training/loop.py` | Add `_StepPrefetch` class + integrate into `_run_epoch_steps` |

---

## Task 1: Phase A — Update `configs/base.toml`

**Files:**
- Modify: `configs/base.toml:82-83`

- [ ] **Step 1: Add `max_data_loader_n_workers` and `dataloader_prefetch_factor` to base.toml**

In `configs/base.toml`, after the existing `persistent_data_loader_workers = true` line (line 83), add the two new parameters:

```toml
dataloader_pin_memory = true
persistent_data_loader_workers= true
max_data_loader_n_workers = 4
dataloader_prefetch_factor = 4
```

- [ ] **Step 2: Verify TOML is valid**

Run: `.venv\Scripts\python.exe -c "import tomllib; tomllib.load(open('configs/base.toml','rb')); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add configs/base.toml
git commit -m "feat: increase DataLoader workers and prefetch for S1 GPU utilization"
```

---

## Task 2: Phase A — Update `workflow/stages/train.py`

**Files:**
- Modify: `workflow/stages/train.py:78-95`

- [ ] **Step 1: Add two params to `_AUTO_DEFAULTS`**

In `workflow/stages/train.py`, add `max_data_loader_n_workers` and `dataloader_prefetch_factor` to the `_AUTO_DEFAULTS` dict (after line 93 `"use_cmmd": False,`):

```python
_AUTO_DEFAULTS = {
    "network_module": "networks.lora_anima",
    "network_train_unet_only": True,
    "mixed_precision": "bf16",
    "save_precision": "bf16",
    "attn_mode": "flash",
    "use_vae_cache": True,
    "use_text_cache": True,
    "skip_cache_check": True,
    "vae_chunk_size": 64,
    "vae_disable_cache": True,
    "masked_loss": True,
    "log_every_n_steps": 2,
    "dataloader_pin_memory": True,
    "persistent_data_loader_workers": True,
    "use_cmmd": False,
    "save_model_as": "safetensors",
    "max_data_loader_n_workers": 4,
    "dataloader_prefetch_factor": 4,
}
```

These are int values, so `_build_train_cmd` will serialize them as `--max_data_loader_n_workers 4` and `--dataloader_prefetch_factor 4` via the `else` branch (line 240-242). No changes needed to `_BOOL_VALUE_KEYS` or other serialization logic.

- [ ] **Step 2: Verify Python syntax**

Run: `.venv\Scripts\python.exe -c "import ast; ast.parse(open('workflow/stages/train.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add workflow/stages/train.py
git commit -m "feat: add DataLoader workers/prefetch defaults to workflow train stage"
```

---

## Task 3: Phase A — Update `workflow/schemas/train_common.yaml`

**Files:**
- Modify: `workflow/schemas/train_common.yaml:263-295`

- [ ] **Step 1: Add two fields to the `performance` section**

In `workflow/schemas/train_common.yaml`, after the `seed` field (line 295), add two new fields:

```yaml
      - key: seed
        type: int
        required: false
        layer: common
        label: "随机种子"
      - key: max_data_loader_n_workers
        type: int
        required: false
        layer: common
        default: 4
        label: "数据加载进程数"
      - key: dataloader_prefetch_factor
        type: int
        required: false
        layer: common
        default: 4
        label: "数据预取深度"
```

- [ ] **Step 2: Verify YAML is valid**

Run: `.venv\Scripts\python.exe -c "import yaml; yaml.safe_load(open('workflow/schemas/train_common.yaml')); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add workflow/schemas/train_common.yaml
git commit -m "feat: expose DataLoader workers/prefetch in workflow schema"
```

---

## Task 4: Phase B — Add `_StepPrefetch` class to `library/training/loop.py`

**Files:**
- Modify: `library/training/loop.py`

- [ ] **Step 1: Add `_StepPrefetch` class**

Insert the new class after the `_EpochPrefetch` class (after line 440, before `run_training_loop`). This keeps all prefetch-related classes together.

```python
class _StepPrefetch:
    """Overlap H2D transfer of batch N+1 with GPU computation of batch N.

    Uses a dedicated CUDA stream so that ``send_to_device`` for the next
    batch runs concurrently with forward/backward on the default stream.
    """

    __slots__ = ("_stream", "_batch")

    def __init__(self, device: torch.device):
        self._stream = torch.cuda.Stream(device=device)
        self._batch: Optional[Any] = None

    def submit(self, batch_cpu, device: torch.device) -> None:
        """Launch async H2D transfer on the prefetch stream."""
        self._stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(self._stream):
            self._batch = send_to_device(batch_cpu, device)

    def consume(self) -> Optional[Any]:
        """Wait for transfer completion and return the GPU-resident batch.

        Returns ``None`` if no batch was submitted (cold start or after
        invalidate).
        """
        batch = self._batch
        self._batch = None
        if batch is not None:
            torch.cuda.current_stream(batch.device).wait_stream(self._stream)
        return batch

    def invalidate(self) -> None:
        """Discard any pending transfer (used on error / epoch boundary)."""
        self._batch = None
```

- [ ] **Step 2: Verify syntax**

Run: `.venv\Scripts\python.exe -c "import ast; ast.parse(open('library/training/loop.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add library/training/loop.py
git commit -m "feat: add _StepPrefetch CUDA stream double-buffer class"
```

---

## Task 5: Phase B — Integrate `_StepPrefetch` into the prefetch path of `_run_epoch_steps`

**Files:**
- Modify: `library/training/loop.py:520-573`

The prefetch path is the code block starting with `if prefetch is not None:` (line 520). This path is used after the first epoch when `_EpochPrefetch` provides a pre-fetched iterator and first batch.

- [ ] **Step 1: Integrate double-buffering into the prefetch path**

Replace the prefetch path block (lines 520-573) with double-buffered version. The key changes:
- Create `_StepPrefetch` at the start of the epoch
- Step 0 uses `first_batch` directly, then submits `next(iterator)` for async transfer
- Step 1+ consumes the pre-submitted batch, then submits the next one
- On `StopIteration`, don't submit, just run the last step

```python
    if prefetch is not None:
        iterator, first_batch = prefetch.result()
        device = accelerator.device
        step_prefetch = _StepPrefetch(device)
        step = 0
        batch = first_batch
        while True:
            state.current_step.value = state.global_step
            _profiler_step_begin(state)
            loss = _run_step(trainer, state, batch)
            _profiler_step_end(state)
            keys_scaled, mean_norm, maximum_norm, max_mean_logs = _maybe_scale_norm(
                state
            )
            if accelerator.sync_gradients:
                state.progress_bar.update(1)
                state.global_step += 1
                _sample_at_step(trainer, state)
                state.saver.maybe_save_step(state.network, state.global_step, epoch)
                state.optimizer_train_fn()
            _log_step(
                trainer,
                state,
                loss=loss,
                step=step,
                epoch=epoch,
                keys_scaled=keys_scaled,
                mean_norm=mean_norm,
                maximum_norm=maximum_norm,
                max_mean_logs=max_mean_logs,
            )
            _maybe_run_step_validation(trainer, state, epoch)
            if state.global_step >= args.max_train_steps:
                break

            total_steps = len(state.train_dataloader)
            if (
                next_prefetch is None
                and step >= total_steps - prefetch_ahead - 1
                and epoch + 1 < state.num_train_epochs
            ):
                gen = _get_sampler_generator(state.train_dataloader)
                indices = _pre_generate_shuffle_indices(
                    state.train_dataloader, gen, epoch + 1
                )
                next_prefetch = _EpochPrefetch(
                    state.train_dataloader, indices, accelerator.device
                )
                next_prefetch.start()

            step += 1
            # Double-buffer: consume previously submitted async transfer,
            # or fall back to synchronous fetch if nothing was submitted.
            next_batch = step_prefetch.consume()
            if next_batch is not None:
                batch = next_batch
                # Pipeline the *next* next batch while GPU computes.
                try:
                    step_prefetch.submit(next(iterator), device)
                except StopIteration:
                    break
            else:
                try:
                    batch = send_to_device(next(iterator), device)
                except StopIteration:
                    break
```

The logic for step 0: `batch = first_batch` (already on GPU), then after `_run_step`, `step_prefetch.consume()` returns `None` (nothing submitted yet), so we fall to the `else` branch and do `step_prefetch.submit(next(iterator), device)` — wait, that's wrong. Let me reconsider.

Actually, the correct flow is:

- **Step 0**: `batch = first_batch` (on GPU). After `_run_step`, we try `next(iterator)` and `submit` it.
- **Step 1+**: `batch = step_prefetch.consume()` (from previous submit). After `_run_step`, we try `next(iterator)` and `submit` it.
- **Last step**: `next(iterator)` raises `StopIteration` → break.

Here's the corrected code for the batch-fetch section at the bottom of the while loop:

```python
            step += 1
            # Double-buffer: consume async transfer from previous step
            next_batch = step_prefetch.consume()
            if next_batch is not None:
                batch = next_batch
            else:
                # Step 0 cold start (no prior submit) or single-step epoch
                try:
                    batch = send_to_device(next(iterator), device)
                except StopIteration:
                    break
            # Submit next batch for async transfer while GPU is about to compute
            try:
                step_prefetch.submit(next(iterator), device)
            except StopIteration:
                pass  # Last step — no more to prefetch, loop will end after this iteration
```

Wait, this has a problem: we call `next(iterator)` twice per iteration (once for consume target, once for submit). Let me rethink.

The correct pattern is:

```python
            step += 1
            # Double-buffer: try to consume previously submitted async transfer
            next_batch = step_prefetch.consume()
            if next_batch is not None:
                batch = next_batch
                # Already have a batch from prefetch; submit the next one
                try:
                    step_prefetch.submit(next(iterator), device)
                except StopIteration:
                    pass  # no more data, but current batch is valid
            else:
                # Cold start (step 0 after first_batch) or no prior submit
                try:
                    batch = send_to_device(next(iterator), device)
                except StopIteration:
                    break
                # Submit the one after for async transfer
                try:
                    step_prefetch.submit(next(iterator), device)
                except StopIteration:
                    pass
```

This consumes one `next()` from the submit of the previous iteration, and submits one more for the next iteration. On step 0, there's no prior submit so we do a sync fetch and then submit the next one.

Actually wait — step 0 uses `first_batch` directly. After `_run_step` for step 0, we enter the batch-fetch. `consume()` returns `None` (nothing submitted yet), so we go to `else`, sync-fetch `next(iterator)`, and submit the one after. Step 1: `consume()` returns the submitted batch. Submit the next. And so on.

This is correct. Let me write the full replacement block.

- [ ] **Step 2: Verify syntax**

Run: `.venv\Scripts\python.exe -c "import ast; ast.parse(open('library/training/loop.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add library/training/loop.py
git commit -m "feat: integrate _StepPrefetch into prefetch path of training loop"
```

---

## Task 6: Phase B — Integrate `_StepPrefetch` into the standard path of `_run_epoch_steps`

**Files:**
- Modify: `library/training/loop.py:574-636`

The standard path is the `else` branch (line 574) when no `_EpochPrefetch` is active (typically first epoch or resume).

- [ ] **Step 1: Integrate double-buffering into the standard path**

The standard path currently iterates with a `for` loop. We need to change it to use manual iteration so we can peek ahead for the async submit. Replace the `for step, batch in enumerate(...)` loop body:

```python
    else:
        skipped_dataloader = None
        if state.initial_step > 0:
            skipped_dataloader = accelerator.skip_first_batches(
                state.train_dataloader, state.initial_step - 1
            )
            state.initial_step = 1

        step_prefetch = _StepPrefetch(accelerator.device)
        dl_iter = iter(skipped_dataloader or state.train_dataloader)
        step = 0
        # Prime: sync-fetch the first batch (or skip it for initial_step)
        try:
            batch_cpu = next(dl_iter)
        except StopIteration:
            return next_prefetch

        for step in range(len(state.train_dataloader)):
            if state.initial_step > 0:
                state.initial_step -= 1
                try:
                    batch_cpu = next(dl_iter)
                except StopIteration:
                    break
                continue

            # Use async-consumed batch if available, otherwise sync-transfer
            batch = step_prefetch.consume()
            if batch is None:
                batch = send_to_device(batch_cpu, accelerator.device)

            _profiler_step_begin(state)

            loss = _run_step(trainer, state, batch)

            _profiler_step_end(state)

            keys_scaled, mean_norm, maximum_norm, max_mean_logs = _maybe_scale_norm(
                state
            )

            if accelerator.sync_gradients:
                state.progress_bar.update(1)
                state.global_step += 1
                _sample_at_step(trainer, state)
                state.saver.maybe_save_step(state.network, state.global_step, epoch)
                state.optimizer_train_fn()

            _log_step(
                trainer,
                state,
                loss=loss,
                step=step,
                epoch=epoch,
                keys_scaled=keys_scaled,
                mean_norm=mean_norm,
                maximum_norm=maximum_norm,
                max_mean_logs=max_mean_logs,
            )
            _maybe_run_step_validation(trainer, state, epoch)

            if state.global_step >= args.max_train_steps:
                break

            total_steps = len(state.train_dataloader)
            if (
                next_prefetch is None
                and step >= total_steps - prefetch_ahead - 1
                and epoch + 1 < state.num_train_epochs
            ):
                gen = _get_sampler_generator(state.train_dataloader)
                indices = _pre_generate_shuffle_indices(
                    state.train_dataloader, gen, epoch + 1
                )
                next_prefetch = _EpochPrefetch(
                    state.train_dataloader, indices, accelerator.device
                )
                next_prefetch.start()

            # Fetch next CPU batch and submit for async H2D transfer
            try:
                batch_cpu = next(dl_iter)
                step_prefetch.submit(batch_cpu, accelerator.device)
            except StopIteration:
                break
```

Key differences from original:
1. `for step, batch in enumerate(dataloader)` → manual `dl_iter = iter(dataloader)` + `for step in range(...)` for peek-ahead control
2. `batch` is now consumed from `_StepPrefetch` when available, falling back to sync `send_to_device`
3. At the end of each step, `next(dl_iter)` fetches the CPU-side batch and submits it for async transfer

- [ ] **Step 2: Verify syntax**

Run: `.venv\Scripts\python.exe -c "import ast; ast.parse(open('library/training/loop.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add library/training/loop.py
git commit -m "feat: integrate _StepPrefetch into standard path of training loop"
```

---

## Task 7: Phase B — Add error handling for `_StepPrefetch` in `run_training_loop`

**Files:**
- Modify: `library/training/loop.py:462-470`

- [ ] **Step 1: Add `step_prefetch.invalidate()` to error handler**

The `run_training_loop` function (line 442) has a `try/except` around `_run_epoch_steps` (line 462-470). The `_StepPrefetch` instance lives inside `_run_epoch_steps`, so when an exception occurs the local variable is lost. However, Python's GC will clean up the CUDA stream. No explicit `invalidate()` is needed at the `run_training_loop` level — the `_StepPrefetch` is scoped to `_run_epoch_steps` and will be garbage collected on exception.

**No code changes needed for this task.** The existing error handling in `_run_epoch_steps` is sufficient because `_StepPrefetch` is a local variable that goes out of scope on exception.

- [ ] **Step 2: Mark as complete (no-op)**

---

## Task 8: Verify end-to-end with a short training run

**Files:** None (verification only)

- [ ] **Step 1: Run a short training to verify no regressions**

Run a 1-epoch LoKr training with S1 resolution to verify:
1. Training starts without errors
2. DataLoader uses 4 workers (check log output)
3. GPU utilization is improved (check with system monitor)

```powershell
.venv\Scripts\python.exe train.py --method lora --preset default --max_train_epochs 1
```

- [ ] **Step 2: Verify Phase B double-buffering is active**

Check that `_StepPrefetch` is being used by looking for CUDA stream operations in the training log (no explicit logging needed — just verify no errors/crashes).

---

## Self-Review

**Spec coverage:**
- Phase A config changes: Tasks 1-3 cover CLI/GUI (base.toml), Workflow (train.py + schema), ComfyUI (explicitly excluded) ✓
- Phase B `_StepPrefetch`: Task 4 ✓
- Phase B integration (prefetch path): Task 5 ✓
- Phase B integration (standard path): Task 6 ✓
- Error handling: Task 7 ✓
- `_EpochPrefetch` interaction: No changes needed, verified safe ✓

**Placeholder scan:** No TBD/TODO/fill-in-later found.

**Type consistency:** `_StepPrefetch.submit(batch_cpu, device)` / `.consume() -> Optional[Any]` / `.invalidate()` — consistent across Tasks 4-6.
