# LyCORIS Variants: LOCON / LOHA / LOKR

Three low-rank adaptation architectures from the LyCORIS family, integrated as first-class `network_type` options. Each variant factorizes the weight update $\Delta W$ differently from standard LoRA, offering different parameter-efficiency / expressiveness trade-offs.

> **For the structural walkthrough** (forward pass formulas, dimension analysis, weight key naming, scalar baking behavior, comparison with standard LoRA), see **`docs/structure/lycoris-variants.md`**. This doc is the usage / ops / configuration reference.

## Overview

Standard LoRA decomposes $\Delta W = B A$ with a single rank-$r$ bottleneck. The LyCORIS family explores alternative factorizations:

| Variant | Core operation | Effective rank | Params relative to LoRA (same $r$) |
|---------|---------------|----------------|-------------------------------------|
| **LOCON** | $\Delta W = B A$ + Tucker core for Conv2d | $r$ | Same for Linear; +Tucker for Conv2d |
| **LOHA** | $\Delta W = (W_1^a W_1^b) \odot (W_2^a W_2^b)$ | $r^2$ | ~2× |
| **LOKR** | $\Delta W = \text{kron}(W_1,\, W_2)$ | $\text{rank}(W_1) \cdot \text{rank}(W_2)$ | Adaptive; can be < LoRA at same effective rank |

All three inherit `BaseLoRAModule` and share the same training pipeline (dropout, rank dropout, module dropout, max-norm scaling, channel scaling where applicable). They are selected via `network_type` and are **mutually exclusive** with each other and with OrthoLoRA / HydraLoRA.

**Unified reference:** Shih-Ying Yeh, Yu-Guan Hsieh, Zhidong Gao, Bernard B.W. Yang, Giyeong Oh, Yanmin Gong, "Navigating Text-To-Image Customization: From LyCORIS Fine-Tuning to Model Evaluation", ICLR 2024. [[Paper](https://openreview.net/forum?id=wfzXa8e783)] [[Code](https://github.com/KohakuBlueleaf/LyCORIS)]

---

## LOCON — Enhanced LoRA with Tucker Decomposition

Same algorithm as standard LoRA for Linear layers; adds a Tucker core tensor for Conv2d layers with kernel_size > 1, allowing the spatial convolution structure to be captured more efficiently than a flattened rank-$r$ approximation.

**Reference:** LoCon originated as a separate project for Tucker-decomposed Conv2d LoRA, now part of the LyCORIS project ([KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS)).

### Algorithm

- **Linear:** $\Delta W = B A \cdot s$, identical to standard LoRA.
- **Conv2d (Tucker mode):** Introduces a core tensor $T \in \mathbb{R}^{r \times r \times k_1 \times k_2}$ and reconstructs:

$$
\Delta W = \text{rebuild\_tucker}(T,\, W_\text{up}^\top,\, W_\text{down}) = \sum_{i,j} T_{ij} \cdot w_\text{up}^{(i)} \otimes w_\text{down}^{(j)}
$$

### File format

```
<prefix>.lora_down.weight    # (rank, in_dim)   or (rank, in_ch, k1, k2)
<prefix>.lora_up.weight      # (out_dim, rank)  or (out_ch, rank, 1, 1)
<prefix>.lora_mid.weight     # (rank, rank, k1, k2) — only in Tucker mode
<prefix>.alpha               # scalar
```

### Config parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `network_dim` | 32 | Rank for Linear layers |
| `network_alpha` | 32 | LoRA alpha |
| `conv_dim` | 1 | Rank for Conv2d layers |
| `conv_alpha` | 4 | Alpha for Conv2d layers |
| `use_tucker` | false | Enable Tucker decomposition for Conv2d with kernel_size > 1 |
| `scale_weight_norms` | 1.0 | Max-norm scaling target |

> **Anima 注意**：Anima-base-v1.0 DiT 的所有 LoRA 目标层都是 `nn.Linear`（无 `Conv2d`）。`conv_dim`、`conv_alpha`、`use_tucker` 参数在 Anima 训练中**不会生效**——设置它们不会产生错误，但也不会有任何效果。LOCON 在纯 Linear 模型上的行为与标准 LoRA **完全相同**。

---

## LOHA — Low-Rank Hadamard Product Adaptation

Uses the Hadamard (element-wise) product of two low-rank matrices to achieve effective rank $r^2$ with only $2\times$ the parameters of standard LoRA. Custom autograd functions provide exact gradients for the Hadamard product.

**Reference:** Part of the LyCORIS project ([KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS)). The Hadamard product decomposition was introduced in the LyCORIS ICLR 2024 paper.

### Algorithm

$$
\Delta W = \left( W_1^a W_1^b \right) \odot \left( W_2^a W_2^b \right) \cdot s
$$

where $W_1^a \in \mathbb{R}^{d_\text{out} \times r}$, $W_1^b \in \mathbb{R}^{r \times d_\text{in}}$, $W_2^a \in \mathbb{R}^{d_\text{out} \times r}$, $W_2^b \in \mathbb{R}^{r \times d_\text{in}}$.

The Hadamard product's gradient is $\nabla_A = \nabla \cdot B$ and $\nabla_B = \nabla \cdot A$, implemented via `HadaWeight` / `HadaWeightTucker` custom autograd functions.

### File format

```
<prefix>.hada_w1_a           # (rank, out_dim)   or (rank, out_dim) in Tucker
<prefix>.hada_w1_b           # (rank, in_dim)    or (rank, in_ch)
<prefix>.hada_w2_a           # (rank, out_dim)   or (rank, out_dim) in Tucker
<prefix>.hada_w2_b           # (rank, in_dim)    or (rank, in_ch)
<prefix>.hada_t1             # (rank, rank, k1, k2) — only in Tucker mode
<prefix>.hada_t2             # (rank, rank, k1, k2) — only in Tucker mode
<prefix>.alpha               # scalar
```

### Config parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `network_dim` | 16 | Rank $r$ (effective rank ≈ $r^2$) |
| `network_alpha` | 8 | LoRA alpha |
| `conv_dim` | 4 | Rank for Conv2d layers |
| `conv_alpha` | 1 | Alpha for Conv2d layers |
| `use_tucker` | false | Enable Tucker mode for Conv2d |
| `scale_weight_norms` | 1.0 | Max-norm scaling target |

> **Anima 注意**：`conv_dim`、`conv_alpha`、`use_tucker` 在 Anima 上不生效（纯 Linear 模型）。LOHA 的 Tucker 模式（`hada_t1`/`hada_t2`）也仅对 Conv2d 层有效，Anima 训练中不会触发。

---

## LOKR — Low-Rank Kronecker Product Adaptation

Factorizes the input and output dimensions into structured block matrices and composes them via the Kronecker product. This produces a structured high-rank approximation whose parameter count depends on the factorization shape rather than a single rank value.

**Reference:** Part of the LyCORIS project ([KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS)). The Kronecker product adaptation was introduced in the LyCORIS ICLR 2024 paper.

### Algorithm

Dimensions $d_\text{in}$ and $d_\text{out}$ are factored: $d_\text{in} = m \cdot n$, $d_\text{out} = l \cdot k$. Then:

$$
\Delta W = \text{kron}(W_1,\, W_2) \cdot s
$$

where $W_1 \in \mathbb{R}^{l \times m}$ (or its low-rank factors) and $W_2 \in \mathbb{R}^{k \times n}$ (or its low-rank factors). When `lora_dim` is small relative to the factored dimensions, both $W_1$ and $W_2$ are further decomposed into low-rank pairs. When `lora_dim` is large enough, either factor may become a full matrix.

### File format

```
<prefix>.lokr_w1              # (l, m) — full mode
<prefix>.lokr_w1_a            # (l, rank) — decomposed mode
<prefix>.lokr_w1_b            # (rank, m) — decomposed mode
<prefix>.lokr_w2              # (k, n) or (k, n, k1, k2) — full mode
<prefix>.lokr_w2_a            # (k, rank) — decomposed mode
<prefix>.lokr_w2_b            # (rank, n) or (rank, n*k1*k2) — decomposed mode
<prefix>.lokr_t2              # (rank, rank, k1, k2) — Tucker mode for Conv2d
<prefix>.alpha                # scalar
<prefix>.dora_scale           # optional — only when weight_decompose=true
```

### Config parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `network_dim` | 16 | Rank $r$ for low-rank factors |
| `network_alpha` | 8 | LoRA alpha |
| `conv_dim` | 1 | Rank for Conv2d layers |
| `conv_alpha` | 4 | Alpha for Conv2d layers |
| `decompose_both` | false | Decompose both $W_1$ and $W_2$ into low-rank |
| `use_tucker` | false | Tucker core for Conv2d $W_2$ |
| `lokr_factor` | -1 | Target factor size for dimension factorization; -1 = auto |
| `full_matrix` | false | Force full (non-decomposed) matrices |
| `use_scalar` | false | Learnable scalar (zero-init) instead of fixed scalar=1 |
| `weight_decompose` | false | Enable DoRA-style weight decomposition |
| `scale_weight_norms` | 1.0 | Max-norm scaling target |

> **Anima 注意**：`conv_dim`、`conv_alpha`、`use_tucker` 在 Anima 上不生效（纯 Linear 模型）。`lokr_factor` 是 LOKR 对 Anima 影响最大的参数——见下方 "LOKR lokr_factor divisibility constraint" 章节。`decompose_both=true` 配合较小 rank（如 4-8）可显著减少 LOKR 参数量。

---

## Training

### GUI

Select LOCON / LOHA / LOKR from the method dropdown. Each method ships a dedicated config in `configs/gui-methods/` (`locon.toml`, `loha.toml`, `lokr.toml`) that pre-fills all variant-specific parameters. The right-side explanation panel shows the corresponding HTML guide.

### CLI

Set `network_type` in your TOML config:

```toml
network_type = "loha"   # or "locon", "lokr"
network_dim  = 16
network_alpha = 8
```

All three variants share the same training loop as standard LoRA — no special loss terms or scheduler changes. They support the same dataset format, optimizer settings, and checkpointing schedule.

### Mutual exclusivity

LyCORIS variants are **mutually exclusive** with:
- **OrthoLoRA** (`use_ortho = true`) — orthogonal re-parameterization is not defined for non-standard factorizations
- **HydraLoRA** (`use_hydra = true`) — multi-head routing requires the standard $BA$ structure
- Each other — only one `network_type` can be active per training run

Setting `network_type` to any LyCORIS variant with `use_ortho = true` or `use_hydra = true` will raise a configuration error at startup.

---

## Inference

### CLI — static merge

`inference.py` auto-detects the variant by inspecting safetensors key prefixes (`hada_*` → LOHA, `lokr_*` → LOKR, standard `lora_up`/`lora_down` with LOCON metadata → LOCON). The weight delta is computed using the variant-specific formula and merged into the base model weights before denoising.

All three variants can coexist in the same `--lora_weight` list with regular LoRA files — each file is merged independently using its own formula.

### ComfyUI — LyCORIS adapter support

ComfyUI's LyCORIS loader node natively supports LOHA, LOKR, and LOCON weight formats. The safetensors files produced by this trainer use the same key naming convention as the sd-scripts / LyCORIS ecosystem, so they are directly loadable without conversion.

---

## Known limitations and compatibility notes

- **LOCON for Linear layers** is mathematically identical to standard LoRA. The only difference is Tucker decomposition support for Conv2d layers. If your target architecture has no Conv2d layers (e.g. DiT-based models), LOCON produces the same result as standard LoRA.
- **LOHA effective rank** is $r^2$, but the actual expressiveness is constrained by the Hadamard product structure — it is not equivalent to a rank-$r^2$ LoRA. The Hadamard product introduces element-wise multiplicative interactions that standard LoRA does not have.
- **LOKR factorization quality** depends on `lokr_factor`. A poor factorization (e.g. one factor = 1) degrades to a standard matrix and loses the Kronecker structure's efficiency. Use `lokr_factor = -1` (auto) or a value that evenly divides both $d_\text{in}$ and $d_\text{out}$.
- **LOKR `lokr_factor` divisibility constraint.** When `lokr_factor > 0`, the `factorization()` function requires `dimension % lokr_factor == 0` for **both** `d_in` and `d_out`. If this condition is not met for a given layer, the function **silently falls back to automatic search** — your explicit factor value is ignored for that layer.

  **Anima-base-v1.0 DiT layer dimensions** (the dimensions LyCORIS actually sees at training time):

  | Layer | Weight shape | d_in | d_out | Notes |
  |-------|-------------|------|-------|-------|
  | `self_attn.qkv_proj` | `[6144, 2048]` | 2048 | 6144 | Q/K/V fused: 2048×3 |
  | `self_attn.output_proj` | `[2048, 2048]` | 2048 | 2048 | Square |
  | `cross_attn.q_proj` | `[2048, 2048]` | 2048 | 2048 | Q not fused |
  | `cross_attn.kv_proj` | `[4096, 1024]` | 1024 | 4096 | K/V fused: 2048×2 |
  | `cross_attn.output_proj` | `[2048, 2048]` | 2048 | 2048 | Square |
  | `mlp.layer1` | `[8192, 2048]` | 2048 | 8192 | 4× expansion |
  | `mlp.layer2` | `[2048, 8192]` | 8192 | 2048 | Inverse |
  | `llm_adapter.*_proj` | `[1024, 1024]` | 1024 | 1024 | Text encoder adapter |

  **Recommended `lokr_factor` values for Anima:**

  | `lokr_factor` | 2048 | 6144 | 4096 | 8192 | 1024 | Notes |
  |---------------|------|------|------|------|------|-------|
  | **-1** (auto) | ✅ | ✅ | ✅ | ✅ | ✅ | Always works; finds most balanced pair |
  | **4** | ✅ 512 | ✅ 1536 | ✅ 1024 | ✅ 2048 | ✅ 256 | Divides all Anima dimensions |
  | **8** | ✅ 256 | ✅ 768 | ✅ 512 | ✅ 1024 | ✅ 128 | Good balance of granularity |
  | **16** | ✅ 128 | ✅ 384 | ✅ 256 | ✅ 512 | ✅ 64 | Recommended default |
  | **32** | ✅ 64 | ✅ 192 | ✅ 128 | ✅ 256 | ✅ 32 | Larger factors → fewer parameters |
  | **3** | ❌ | ✅ 2048 | ❌ | ✅ 2730 | ❌ | **Fails for 2048/4096/1024** — silently ignored |

  **Practical guidance:**
  - Use `lokr_factor = -1` (auto) unless you need precise control over factorization shape
  - `lokr_factor = 16` works for **all** Anima DiT layers and is a good starting point
  - `lokr_factor = 3` is **not recommended** — it only divides 6144 and 8192, so `self_attn.output_proj` (2048×2048), `cross_attn.q_proj` (2048×2048), and `llm_adapter` layers (1024×1024) will silently use auto factorization instead
- **No HydraLoRA / OrthoLoRA composition.** These variants cannot be combined with HydraLoRA's multi-head routing or OrthoLoRA's Cayley re-parameterization. The factorization structure is incompatible.
- **Checkpoint continuation.** Loading a LyCORIS safetensors file for continued training requires the correct `network_type` — the key prefix is auto-detected, but module dimensions must match.
- **DoRA (weight_decompose)** is only available for LOKR. LOCON and LOHA do not implement weight decomposition.

---

## Composition with other variants

| Stacks with | How it composes |
|-------------|----------------|
| **T-LoRA** | Timestep rank masking applies to the rank dimension. Works for LOCON (same as standard LoRA). For LOHA/LOKR the mask is applied after `make_weight` on the reconstructed weight. |
| **Spectrum** | Cached steps skip transformer blocks entirely — all LyCORIS modules just run fewer times. No interaction. |
| **Modulation guidance** | Orthogonal. Touches AdaLN only, outside the adapted Linears/Convs. |
| **ReFT** | Orthogonal side-channel. No interaction. |
| **HydraLoRA** | **Not supported.** HydraLoRA requires the standard $BA$ structure. |
| **OrthoLoRA** | **Not supported.** Cayley re-parameterization is defined for standard $BA$ only. |
| **P-GRAFT** | Cutoff step toggles `network.enabled` — all LyCORIS variants honor the flag. |
