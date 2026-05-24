# lycoris

Forward/backward numerical-correctness bench for the LyCORIS adapter modules
shipped in `networks/lora_modules/`: **LOCON**, **LOHA**, and **LOKR**.

## What it measures

`forward_equivalence.py` constructs small LOCON / LOHA / LOKR modules over a
plain `nn.Linear(in_features, out_features)`, plants **known** decomposed
weights, and runs a forward pass.  The module output is compared against the
same computation done by hand (matmul / Hadamard / Kronecker product as
appropriate).  The bench reports:

| metric | meaning |
|--------|---------|
| `max_abs_error` | max |module_output − reference| over all elements |
| `rel_l2` | ‖module_output − reference‖₂ / ‖reference‖₂ |

for each variant (locon, loha, lokr).  A secondary check sweeps dtype
(fp32 / bf16 / fp16) to confirm that low-precision paths stay within their
type's expected ULP range.

No pre-trained model or dataset is required — the bench is self-contained and
uses `torch.manual_seed` for reproducibility.

## Usage

```bash
# Full run (all variants, fp32)
uv run python bench/lycoris/forward_equivalence.py

# Quick smoke
uv run python bench/lycoris/forward_equivalence.py --n_trials 2

# Sweep dtype
uv run python bench/lycoris/forward_equivalence.py --dtype bf16

# Custom dimensions
uv run python bench/lycoris/forward_equivalence.py --in_features 32 --out_features 64 --rank 8
```

## Output

Standard bench envelope (`result.json`) via `bench/_common.write_result`.

```
bench/lycoris/results/<YYYYMMDD-HHMM>[-<label>]/
    result.json          # schema_version=1 envelope with per-variant metrics
```

`result.json` → `metrics` contains a key per variant (`locon`, `loha`,
`lokr`), each holding `max_abs_error`, `rel_l2`, and the test dimensions.

## Interpretation

| max_abs_error | verdict |
|---------------|---------|
| < 1e-5 (fp32) | **PASS** — module forward matches hand-computed reference within float32 ULP. |
| 1e-5 … 1e-3 | **MARGINAL** — likely a bf16 cast; re-run with `--dtype fp32` to confirm. |
| > 1e-3 | **FAIL** — a real bug in the decomposed weight assembly or the functional ops. |

This bench does not require a baseline run — it computes the expected output
inline and compares.  Run it after any change to `locon.py`, `loha.py`,
`lokr.py`, or `lycoris_functional.py`.
