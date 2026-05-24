# Tooling architecture: a shared orchestration layer for `bench/` · `scripts/` · `preprocess/`

Status: **proposal** (2026-05-24). Follows the embedder-API work in
`examples/` (the `anima_lora` façade + the resolved `suggestions.md` friction
log) and the quick-win consolidation that shipped alongside it
(`anima_lora.ROOT`, `ensure_text_strategies` adoption, `resolve_dtype` →
`library.runtime.device.str_to_dtype`). Those fixed the *embedder-facing*
surface; this proposes the next layer down — the internal tooling.

## TL;DR

The repo has a clean **primitive** layer (`library/*`) and a new clean
**façade** for embedders (`anima_lora/`). What it does *not* have is an
**orchestration** layer: the reusable "drive the primitives end-to-end" logic —
*scan a directory → group images by shape → batched-encode → write sidecars
idempotently*, or *load a DiT with the adapter + compile applied in the right
order* — lives nowhere importable. It's smeared across nine `preprocess/*.py`
`main()` bodies and locked inside `bench/_anima.py` under a `bench`-private name.

So every tool reinvents the orchestration, and three tooling trees
(`bench/`, `scripts/`, `preprocess/`) can't share a harness. The proposal:

1. **Promote the orchestration into `library/`** — a new `library/preprocess/`
   package holds the dataset-caching functions; `preprocess/*.py` shrink to thin
   argparse wrappers that call them. (This is the "absorb `preprocess/` into
   `library/`" idea, done as *move the logic, keep the CLI*.)
2. **Promote the reusable half of `bench/_anima.py`** (model-build-with-ordering,
   bucket discovery) into `library/` so `bench`/`scripts`/`preprocess` share one
   harness; the bench-specific `result.json` envelope (`bench/_common.py`) stays
   in `bench/`.
3. **Write down the layering contract** so future tools land in the right place
   instead of growing a fourth parallel mini-library.
4. **Retire the `sys.path` divergence** by packaging the tooling trees (the one
   change the quick-win pass deliberately deferred, because the shims are
   load-bearing today).

No user-facing behavior changes. Every `make` target and CLI keeps working; the
cache *contents* stay equivalent.

## Background: three layers, one missing

| Layer | Role | Where it lives today | Health |
|---|---|---|---|
| **Primitives** | Load a model, encode one batch, resolve a cache path, decode a latent | `library/anima`, `library/models`, `library/vision`, `library/io/cache.py`, `library/runtime` | ✅ good, well-factored |
| **Façade** | "Read four exports" entry points for embedders | `anima_lora/__init__.py` | ✅ new, shipped |
| **Orchestration** | Drive the primitives over a *dataset* / *full run* — the loop, the grouping, the idempotent skip, the adapter+compile ordering | **nowhere reusable** — inlined in `preprocess/*.py` `main()` and `bench/_anima.py` | ❌ the gap |
| **Entry points** | argparse + dispatch | `preprocess/*.py`, `bench/**/run_bench.py`, `scripts/**`, `tasks.py` | ⚠️ carry orchestration they shouldn't |

The primitives are already where they belong. `library/io/cache.py` owns the
suffixes (`LATENT_CACHE_SUFFIX`, …) and `resolve_cache_path` /
`discover_cached_*`; `library/vision/encoder.py` owns `load_pe_encoder` +
`encode_pe_from_imageminus1to1`; `library/models/qwen_vae.py` owns the VAE. The
*scripts* import these correctly — the audit found **zero hand-rolled loaders**.

The problem is one level up. Compare the three cache scripts:

- `preprocess/cache_latents.py` (227 lines), `cache_text_embeddings.py` (435),
  `cache_pe_encoder.py` (442) each independently implement the **same shape**:
  walk `--dir` (with a `--recursive` per-subdir stem-collision check), pre-skip
  already-cached entries, group by resolution, run batched encoding (`DataLoader`
  in PE, manual batches in VAE/TE), write `{stem}_…safetensors`/`.npz`
  idempotently. The differences are the encoder and the suffix; the
  *orchestration is identical and copied three times* — see
  `cache_pe_encoder.py:325-419` vs the equivalent block in `cache_latents.py`.

- `bench/_anima.py:194 build_anima` encodes a genuinely load-bearing invariant —
  *`compile_blocks()` must run AFTER `apply_to` + `load_weights` so the LoRA
  monkey-patches are in the compiled graph* (`bench/_anima.py:148-155`). That
  ordering is exactly what an embedder or a new script needs and can't get
  without copying it out of `bench`. `discover_bucketed_samples`
  (`bench/_anima.py:342`) is similarly general but `bench`-locked.

Net effect: a new `scripts/` tool that wants "cache a dir of images" or "load
the DiT the way bench does" has nowhere to import it from, so it copies. The
2026-05-24 scan still finds 41 files under `bench/` + `scripts/` +
`preprocess/` using `sys.path` insertion or local root arithmetic; the quick-win
pass only removed a few narrow duplicates (strategy setup and dtype mapping).

## Target architecture (the layering contract)

```
                       ┌─────────────────────────────────────────┐
  entry points         │ tasks.py · preprocess/*.py · bench/**   │  ← argparse + dispatch ONLY
  (thin)               │ scripts/** · train.py · inference.py    │     (+ bench result envelope)
                       └───────────────────┬─────────────────────┘
                                           │ imports
  orchestration        ┌───────────────────▼─────────────────────┐
  (NEW, reusable)      │ library/preprocess/  (cache a dataset)   │
                       │ library/runtime/harness.py (build models) │
                       │ library/io/cache.py   (discover caches)   │
                       └───────────────────┬─────────────────────┘
                                           │ imports
  primitives           ┌───────────────────▼─────────────────────┐
                       │ library/{anima,models,vision,io,runtime} │
                       └───────────────────▲─────────────────────┘
                                           │ re-exports (lazy)
  façade (embedders)   ┌───────────────────┴─────────────────────┐
                       │ anima_lora/__init__.py                   │
                       └─────────────────────────────────────────┘
```

**Rule:** orchestration logic — any loop/grouping/ordering that an `if __name__
== "__main__"` block would otherwise own — lives in `library/`. Entry points
contain *only* argument parsing, defaults, and a call into `library/`. `bench/`
additionally owns its `result.json` envelope (a bench concept, not pipeline
infrastructure). The `anima_lora` façade re-exports the handful of orchestration
+ primitive entry points an *embedder* needs (it already does this for
generation/training).

## Proposed module homes (current → target)

### A. Dataset caching → `library/preprocess/` — **DONE** (2026-05-24)

A new subpackage mirroring the `preprocess/` dir name, holding the orchestration
the cache scripts shared. Implemented as below; every `preprocess/cache_*.py` is
now argparse + model-load + a call into `library/`, and the entire encode loop
(not just walk/group/skip) lives in the library:

```
library/preprocess/
  __init__.py          # re-exports the cache functions + dataset/progress helpers
  _dataset.py          # walk_images(dir, recursive) + per-subdir stem-collision
                       #   check; group_by_shape(); partition_cached(); PreprocessStats
  _progress.py         # tqdm_progress(desc) — optional progress callback (bars stay
                       #   out of the library core; wrappers opt in)
  latents.py           # cache_latents(dir, vae, *, cache_dir=…, recursive=…) -> stats
  text.py              # cache_text_embeddings(...) ; cache_pooled_text(...) ;
                       #   generate_caption_variants(...)
  pe.py                # cache_pe_features(...) + write_pe_centroid(...) + compute_pe_centroid(...)
  images.py            # resize_to_buckets(...) + process_image (picklable worker)
  masks.py             # NOT YET — SAM/MIT mask scripts only route through walk_images
                       #   (Phase 2b); full write-loop move still deferred
```

**API contract (as built):** each cache function takes the *already-loaded* model
(`vae` / `bundle` / strategies+encoder) plus explicit paths/dtypes, returns a
`PreprocessStats(seen, written, skipped, failed)`, and accepts an optional
`progress` callback. A `verbose` flag gates informational `print()`s so the
layer runs headless (daemon / tests / embedding code). Model loading, `--help`,
and the one-time uncond-sidecar staging stay in the wrappers.

Each `preprocess/<name>.py` becomes a thin wrapper:

```python
# preprocess/cache_pe_encoder.py  (was 442 lines → ~60)
def main() -> None:
    args = _build_parser().parse_args()
    from library.preprocess.pe import cache_pe_features, write_pe_centroid
    stats = cache_pe_features(
        data_dir=args.dir, cache_dir=args.cache_dir, encoder=args.encoder, ...)
    if args.centroid:
        write_pe_centroid(...)
```

The argparse stays in `preprocess/` (CLI is an entry-point concern); the loop
moves down. `make preprocess-*` and the daemon path in
`scripts/tasks/preprocess.py` are unchanged — they still `python
preprocess/<name>.py`.

Implementation detail: do not create a second image-scanning utility from
scratch. `library/datasets/image_utils.py` already has `IMAGE_EXTENSIONS`,
`glob_images_pathlib`, and the private `_assert_unique_stems`; phase 1 should
promote the unique-stem check to a public helper (or wrap it in
`library/preprocess/_dataset.py`) and reuse `resolve_cache_path` for all nested
cache paths. The same helper also covers `preprocess/generate_masks.py` and
`generate_masks_mit.py`, which currently duplicate `get_image_files` and the
same stem-collision check.

API shape: the library functions should return a small `PreprocessStats`
dataclass (`seen`, `written`, `skipped`, maybe `failed`) and accept explicit
paths/devices/dtypes. Progress bars and `print()` stay in the wrappers unless a
caller passes an optional progress callback. That keeps the new layer usable
from the daemon, tests, examples, and future embedding code without pretending a
CLI is always attached.

### B. Run harness → `library/runtime/harness.py` — **DONE** (2026-05-24)

Promoted the reusable half of `bench/_anima.py`:

| `bench/_anima.py` symbol | Target | Status |
|---|---|---|
| `build_anima` (+ `AnimaBundle`, the compile-after-adapter ordering) | `library/runtime/harness.py::build_anima` | ✅ moved |
| `discover_bucketed_samples` (+ `_RES_RE`) | `library/io/cache.py`, next to `discover_cached_*` | ✅ moved |
| `resolve_dtype` | delegates to `library.runtime.device.str_to_dtype` | ✅ already done |
| `add_common_args` | stays `bench`-side; `--device`/`--dtype` delegate to §C's `add_device_args` | ✅ |

`bench/_anima.py` now keeps `add_common_args` + `resolve_dtype` and **thin
re-exports** `build_anima` / `AnimaBundle` / `discover_bucketed_samples` from
their new homes, so `from bench._anima import build_anima` keeps working without
churning the bench scripts. `bench/_common.py` (the `result.json` writer)
**stays in `bench/`** — it's bench's output contract, not infrastructure. Model-
free coverage in `tests/test_runtime_harness_cli.py` (re-export identity,
`build_anima` no-DiT guard, fixture-backed `discover_bucketed_samples`).

### C. Shared CLI flags → `library/runtime/cli.py` — **DONE** (2026-05-24)

`preprocess/` had nine independent parsers re-declaring the same dataset-IO and
compute flags. `library/runtime/cli.py` now provides two parser-parent helpers:

- `add_io_args(parser, *, dir_required=, cache_noun=, include_batch_size=,
  batch_size_default=, include_num_workers=, …)` — the `--dir` / `--cache_dir` /
  `--recursive` trio (byte-identical across the three cache scripts) plus the
  parameterized `--batch_size` / `--num_workers`.
- `add_device_args(parser, *, include_device=, dtype_default=, dtype_choices=)`
  — `--device` / `--dtype` (the dtype strings round-trip through
  `str_to_dtype`).

Adopted in `cache_latents.py` / `cache_text_embeddings.py` /
`cache_pe_encoder.py` (each keeping its own `--batch_size` default: 4 / 16 / 8)
and `bench._anima.add_common_args` (device/dtype delegated). Behavior-preserving:
defaults, required-ness, and dtype choices are passed through per script;
`--help` + the full `pytest tests/` (412) stay green.

### D. Rename drift, folded in from the open `suggestions.md` items — **DONE** (2026-05-24)

These were catalogued during the `examples/` work and belong to the same
cleanup. All shipped:

- **#9 — done** `CachedDataset` (+ its `BucketBatchSampler`) moved from
  `library/datasets/distill.py` to a new `library/datasets/cache.py` (it was
  never distill-specific — it's the general latent+TE+pooled train-cache
  reader). `distill.py` re-exports both for back-compat; they're also re-exported
  from `library/datasets/__init__.py`.
- **#5 — done** Adapter-attach split out of `load_dit_model` into
  `attach_adapters(model, args, device, *, pgraft_mode, hydra_mode)`
  (`library/inference/models.py`) — the loader now does weights + compile, and
  `attach_adapters` does only the dynamic-hook P-GRAFT / HydraLoRA-moe attach.
  Exported via the `library.inference` lazy facade. `pgraft_mode`/`hydra_mode`
  are passed in (the caller already derives them to gate the static-merge skip).
- **#10 — done** Dropped the vestigial `GenerationSettings.dit_weight_dtype` (the
  model is forced to bf16 in `load_dit_model` regardless). Added a pure
  `resolve_seed(args)` helper; `generate()` no longer mutates `args.seed` in
  place — the CLI single-prompt / interactive paths assign
  `args.seed = resolve_seed(args)` before saving, and the frozen
  `GenerationRequest` contract / its docstring were updated to match.

### E. Packaging — retire the `sys.path` divergence — **E1 DONE** (2026-05-24)

The shims were **load-bearing**: `python bench/spd/foo.py` puts the script
dir on `sys.path[0]`, not the repo root, and `bench`/`scripts`/`preprocess` were
**not** installed packages (only `anima_lora*`/`library*`/`networks*` were). So
`from bench._common import …` only resolved because the shim inserted the repo
root. Three paths were considered:

- **(E1, recommended) — shipped** Added `bench*`, `scripts*`, `preprocess*` to
  `[tool.setuptools.packages.find].include`. After `uv sync` the editable
  finder maps them, so cross-tree imports (`from bench._common import …`,
  `from scripts.dcw.haar import …`) resolve from any cwd and every tool runs as
  `python -m bench.spd.foo`. **48 `sys.path.insert` lines deleted across 34
  files**; `ROOT`/`REPO_ROOT` defs that were only feeding the shim were removed,
  those still used for real default paths were kept (local `Path(__file__)`
  arithmetic — not yet unified onto `anima_lora.ROOT`; see open question 4).
  Added the missing `__init__.py` to `preprocess/` and to the 12 `bench/`
  subpackages that lacked one. Gate: full `pytest tests/` (412) green; ruff
  format clean on all touched files; import + `--help` + `-m` smoke over every
  touched wrapper. Surfaced + fixed two pre-existing broken bench imports
  (`bench/dcw/{covariance_ceiling,k_supervision_sweep}.py` referenced the old
  `ASPECT_NAMES`/`N_ASPECTS` names; repointed at
  `library.datasets.buckets.{DCW_ASPECT_NAMES,N_DCW_ASPECTS}`).
- **(E1b, publish-safe)** If this project is meant to be installed outside the
  repo, avoid exporting a generic top-level `scripts` package. Move the
  importable tooling into `anima_lora_tooling/` (or `anima_lora.tools`) and leave
  the legacy script files as thin `python path/to/file.py` wrappers. More churn
  than E1, but avoids package-name collisions.
- **(E2)** A single `tooling/_bootstrap.py` imported first-thing; uniform, but
  keeps a shim. Lower payoff.

E1 is the real fix for audit problem #1 and unblocks running any tool as `python
-m bench.spd.foo` without a shim.

Keep the ComfyUI node bootstraps out of this cleanup. Their `sys.path` and
`_vendor/` fallbacks are runtime deployment glue for nodes copied outside the
repo, not the internal tooling problem this proposal targets.

## Migration plan (incremental, low-risk, reversible)

Sequenced so each phase is independently shippable and leaves the tree green.
Mirrors the `CONTRIBUTING.md` tier ethos (mechanical first, behavior-preserving,
test-gated).

| Phase | Scope | Risk | Gate |
|---|---|---|---|
| **0 — done** | `anima_lora.ROOT`, `str_to_dtype` export, `ensure_text_strategies` adoption (6 sites), dtype dedup | — | shipped, `ruff` + import smoke green |
| **1 — done** | Created `library/preprocess/_dataset.py` (`walk_images` / `group_by_shape` / `partition_cached` + `PreprocessStats`) and migrated `cache_pe_encoder.py` onto it as the pilot | low | `tests/test_preprocess_dataset.py` (helper unit tests) green; `--help` + import smoke pass |
| **2 — done** | Migrated `cache_latents` / `cache_text_embeddings` / `resize_images` onto `walk_images` + `group_by_shape`; `cache_pooled_text` adopts `PreprocessStats` | low | import + `--help` smoke on all four; shared-helper unit tests green |
| **2b — done** | Routed `generate_masks` (SAM3) + `generate_masks_mit` through `walk_images`; deleted both `get_image_files` + inline collision blocks | low | import + `--help` smoke on both mask scripts |
| **A (full) — done** | Completed §A: moved the *entire* encode loops down — `cache_latents` / `cache_text_embeddings` / `cache_pooled_text` / `cache_pe_features` + `write_pe_centroid` / `resize_to_buckets` now live in `library/preprocess/`, returning `PreprocessStats` with an optional `tqdm_progress` callback. The five `preprocess/cache_*.py` + `resize_images.py` shrank to argparse + model-load + call | medium | full `pytest tests/` (406) green incl. new model-free `cache_pooled_text` / `resize_to_buckets` e2e tests; `--help` + import smoke on all wrappers. Encoder-backed parity (PE/latents/TE contents) still gated on `make preprocess-*` (needs weights) |
| **3 — done** | §B: moved `build_anima` (+ `AnimaBundle`) → `library/runtime/harness.py` and `discover_bucketed_samples` (+ `_RES_RE`) → `library/io/cache.py`; `bench/_anima.py` re-exports both. §C: added `library/runtime/cli.py` (`add_io_args` / `add_device_args`), adopted in the three cache wrappers + `add_common_args` | medium | full `pytest tests/` (412) green incl. new `tests/test_runtime_harness_cli.py`; `ruff` + `--help` + import smoke on touched wrappers |
| **4 — done** | Packaging (E1): `bench*`/`scripts*`/`preprocess*` added to `find.include` + `uv sync`; 48 `sys.path.insert` lines deleted across 34 files (shim-only `ROOT`/`REPO_ROOT` defs removed, real-path ones kept); `__init__.py` added to `preprocess/` + 12 `bench/` subpackages | medium | every tool imports as `-m` from outside the repo; full `pytest tests/` (412) green; ruff-format clean; `--help` + import smoke over all touched entry points. (Did *not* repoint surviving path-building onto `anima_lora.ROOT` — see open question 4) |
| **5 — done** | Rename drift (D): `CachedDataset` → `library/datasets/cache.py` (back-compat re-export kept), `attach_adapters` split out of `load_dit_model`, `GenerationSettings.dit_weight_dtype` dropped, `generate()` seed-mutation replaced with pure `resolve_seed(args)` | low | full `pytest tests/` (412) green incl. updated `test_generation_request.py`; import smoke |

**Content-parity gate (phases 1–2).** Caching is the riskiest to refactor
because of the bucketing/padding invariants (`CLAUDE.md` "Constant-token
bucketing", "Text encoder padding"). Each migration must produce
**content-equivalent sidecars** on a fixed sample dir. Do not require raw byte
identity for `.npz`: `np.savez` writes a zip container, so metadata such as file
timestamps can differ even when every array is identical. Add
`tests/test_preprocess_parity.py` that runs old vs new code on a tiny fixture and
compares keys, shapes, dtypes, tensor/array values, and safetensors metadata.
This makes the move provably behavior-preserving without false failures.

## What stays out of `library/` (non-goals)

- **`bench/_common.py`** — the `result.json` envelope (git SHA, metrics,
  artifacts) is bench's contract; promoting it would make `library/` import git
  plumbing for no reason.
- **`tasks.py` / `scripts/tasks/`** — the `make`-dispatch + subprocess + daemon
  layer is intentionally an entry-point concern; it orchestrates *processes*, not
  pipeline state.
- **Forcing bench probes onto `GenerationRequest`/`generate()`** — the SPD/DCW
  probes drive the sampler at a low level (custom σ schedules, mid-rollout
  captures) on purpose; they want the *primitives* (`build_anima`, strategies,
  decode), not the high-level orchestration. The harness in §B is the right
  shared seam for them, not the façade.
- **A new base-model abstraction** — out of scope (Tier 3 per `CONTRIBUTING.md`).
- **Deleting example/test path shims in the same pass** — examples intentionally
  support `python examples/foo.py`, and tests sometimes import script-local
  modules to exercise direct CLI behavior. Clean those only after the internal
  tooling packages are stable.

## Open questions

1. **`library/preprocess/` vs `library/datasets/caching.py`** — a new subpackage
   (mirrors the `preprocess/` name, room to grow) or a module under the existing
   `library/datasets/`? Leaning subpackage; `library/io/cache.py` already owns the
   path/suffix primitives it would sit on top of.
2. **Phase 4 ordering** — package the tooling trees *before* or *after* the
   `library/preprocess/` move? Packaging first makes the imports uniform but
   touches 41 files in one go; doing it last lets phases 1–3 land risk-free.
   Leaning *last*.
3. **Should the façade re-export the caching entry points** (`anima_lora.cache_pe_features`)?
   Embedders building their own dataset pipeline would want them; defer until a
   real caller asks.
4. **Which root helper is canonical?** Inside `library/`, prefer
   `library.env.project_root()` so low-level modules do not import the façade.
   In wrappers and external examples, `anima_lora.ROOT` is fine. Long term, make
   `anima_lora.ROOT` mirror the same implementation so there is one concept even
   if there are two import paths.
