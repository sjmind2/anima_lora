# LyCORIS Variants: Structural Walkthrough

Forward pass formulas, dimension analysis, weight key conventions, and scalar baking behavior for the three LyCORIS variants (LOCON, LOHA, LOKR). This is the companion to `docs/methods/lycoris-variants.md`, which covers usage, configuration, and composition.

Recap from `lora.md`: every target Linear $W_0 \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ is adapted by $y = W_0 x + m \cdot s \cdot \Delta W \cdot x$, with $m$ the multiplier, $s = \alpha / r$ the scale. The three LyCORIS variants differ in how $\Delta W$ is constructed.

> **Anima 适用性**：Anima-base-v1.0 DiT 的所有 LoRA 目标层均为 `nn.Linear`（无 `Conv2d`）。本文档中所有 "Conv2d" 和 "Tucker" 相关的公式和键名在 Anima 训练中**不会出现**。以下 Conv2d 描述保留作为算法参考，适用于其他包含卷积层的模型（如 Stable Diffusion UNet）。

---

## 1. LOCON — Enhanced LoRA with Tucker Decomposition

Implementation: `networks/lora_modules/locon.py:LoConModule`

### 1.1 Forward pass — Linear layers

Identical to standard LoRA:

$$
\Delta W = W_\text{up}\, W_\text{down}
$$

| Component | Shape | Init |
|-----------|-------|------|
| `lora_down.weight` | $(r,\ d_\text{in})$ | Kaiming uniform |
| `lora_up.weight` | $(d_\text{out},\ r)$ | Zeros |

At step 0: $\Delta W = \mathbf{0}$ because `lora_up` is zero-initialized.

### 1.2 Forward pass — Conv2d with Tucker decomposition

When the target is Conv2d with `kernel_size > 1` and `use_tucker = true`, a core tensor is inserted:

$$
\Delta W = \text{rebuild\_tucker}(T,\, W_\text{up}^\top,\, W_\text{down})
$$

Expanded via einsum:

$$
\Delta W[p][r][\ldots] = \sum_{i=1}^{r} \sum_{j=1}^{r} T[i][j][\ldots] \cdot W_\text{up}^\top[j][r] \cdot W_\text{down}[i][p]
$$

| Component | Shape | Init |
|-----------|-------|------|
| `lora_down.weight` | $(r,\ d_\text{in},\ 1,\ 1)$ — pointwise | Kaiming uniform |
| `lora_mid.weight` | $(r,\ r,\ k_1,\ k_2)$ — spatial core | Kaiming uniform |
| `lora_up.weight` | $(d_\text{out},\ r,\ 1,\ 1)$ — pointwise | Zeros |

The Tucker decomposition factorizes the 4D convolution kernel into two 1×1 pointwise projections (`lora_down`, `lora_up`) connected through the spatial core tensor (`lora_mid`), reducing parameter count from $d_\text{out} \times d_\text{in} \times k_1 \times k_2$ to $r \times d_\text{in} + r^2 \times k_1 \times k_2 + d_\text{out} \times r$.

### 1.3 Non-Tucker Conv2d

When `use_tucker = false` or `kernel_size == (1, 1)`, the Conv2d path uses standard LoRA structure: `lora_down` is a Conv2d with the original kernel size, `lora_up` is pointwise, and $\Delta W$ is computed as a matrix multiply after flattening spatial dims.

---

## 2. LOHA — Low-Rank Hadamard Product

Implementation: `networks/lora_modules/loha.py:LohaModule`
Custom autograd: `networks/lora_modules/lycoris_functional.py:HadaWeight`, `HadaWeightTucker`

### 2.1 Forward pass — Linear layers

$$
\Delta W = \left( W_1^a\, W_1^b \right) \odot \left( W_2^a\, W_2^b \right) \cdot s
$$

| Component | Shape | Init |
|-----------|-------|------|
| `hada_w1_a` | $(d_\text{out},\ r)$ | Normal(std=0.1) |
| `hada_w1_b` | $(r,\ d_\text{in})$ | Normal(std=1) |
| `hada_w2_a` | $(d_\text{out},\ r)$ | **Zeros** |
| `hada_w2_b` | $(r,\ d_\text{in})$ | Normal(std=1) |

At step 0: $\Delta W = (W_1^a W_1^b) \odot \mathbf{0} = \mathbf{0}$ because `hada_w2_a` is zero-initialized.

### 2.2 Custom autograd — HadaWeight

The Hadamard product's gradient cannot be expressed as a simple matrix operation. `HadaWeight` (`lycoris_functional.py:33–51`) implements:

$$
\nabla_{W_1^a} = (\nabla \cdot s \odot (W_2^a W_2^b))\, {W_1^b}^\top
$$

$$
\nabla_{W_1^b} = {W_1^a}^\top (\nabla \cdot s \odot (W_2^a W_2^b))
$$

$$
\nabla_{W_2^a} = (\nabla \cdot s \odot (W_1^a W_1^b))\, {W_2^b}^\top
$$

$$
\nabla_{W_2^b} = {W_2^a}^\top (\nabla \cdot s \odot (W_1^a W_1^b))
$$

This avoids forming the full $(d_\text{out} \times d_\text{in})$ gradient matrix — each factor receives its gradient by multiplying the element-wise product of the upstream gradient and the *other* pair's reconstructed matrix.

### 2.3 Forward pass — Conv2d Tucker mode

When `use_tucker = true` and kernel_size > 1, each pair uses a Tucker core tensor:

$$
R_1[p][r][\ldots] = \sum_{i,j} T_1[i][j][\ldots] \cdot W_{1b}[j][r] \cdot W_{1a}[i][p]
$$

$$
R_2[p][r][\ldots] = \sum_{i,j} T_2[i][j][\ldots] \cdot W_{2b}[j][r] \cdot W_{2a}[i][p]
$$

$$
\Delta W = R_1 \odot R_2 \cdot s
$$

| Component | Shape | Init |
|-----------|-------|------|
| `hada_t1` | $(r,\ r,\ k_1,\ k_2)$ | Normal(std=0.1) |
| `hada_w1_a` | $(r,\ d_\text{out})$ | Normal(std=0.1) |
| `hada_w1_b` | $(r,\ d_\text{in})$ | Normal(std=1) |
| `hada_t2` | $(r,\ r,\ k_1,\ k_2)$ | Normal(std=0.1) |
| `hada_w2_a` | $(r,\ d_\text{out})$ | **Zeros** |
| `hada_w2_b` | $(r,\ d_\text{in})$ | Normal(std=1) |

Note the transposed factor layout in Tucker mode: `hada_w1_a` is $(r,\ d_\text{out})$ rather than $(d_\text{out},\ r)$.

### 2.4 Effective rank analysis

Each pair $W_i^a W_i^b$ is rank-$r$. The Hadamard product of two rank-$r$ matrices has rank at most $r^2$ (each column of the product is an element-wise scaling of columns from the other factor). Thus LOHA at dimension $r$ achieves up to **rank $r^2$** expressiveness with $2 \times (d_\text{out} \cdot r + r \cdot d_\text{in})$ parameters — approximately $2\times$ standard LoRA's parameter count for an $r\times$ effective rank improvement.

### 2.5 Training-time forward (Linear)

At training time, LOHA uses `LohaLinearFn` (`loha.py:11–27`), a custom autograd function that computes `F.linear(x, diff_weight)` with explicit backward to avoid holding the full $\Delta W$ in memory. The `diff_weight` is already computed via `HadaWeight.apply`, so `LohaLinearFn` only handles the linear operation's gradient.

---

## 3. LOKR — Low-Rank Kronecker Product

Implementation: `networks/lora_modules/lokr.py:LokrModule`
Functional: `networks/lora_modules/lycoris_functional.py:make_kron`, `factorization`

### 3.1 Dimension factorization

Both $d_\text{in}$ and $d_\text{out}$ are factored into pairs:

$$
d_\text{in} = m \cdot n, \qquad d_\text{out} = l \cdot k
$$

The `factorization(dimension, factor)` function (`lycoris_functional.py:8–30`) finds the optimal $(m, n)$ such that $m \leq n$, $m \cdot n = d$, and $m + n$ is minimized (i.e. as close to a square as possible).

**Exact behavior by `factor` value:**

| `factor` | Condition | Result |
|----------|-----------|--------|
| `factor > 0` | `dimension % factor == 0` | `(factor, dimension // factor)` — exact, fast |
| `factor > 0` | `dimension % factor != 0` | **Silent fallback** to automatic search below — your `factor` is ignored |
| `factor < 0` (default) | always | Iterative search: start from $m=1$, increment $m$ while $m + n$ decreases and $m \leq |factor|$ |

The silent fallback when `dimension % factor != 0` is a common source of confusion: users set `lokr_factor` expecting a specific factorization shape, but for layers where the dimension is not divisible by that factor, the function produces a different (but still valid) decomposition without any warning.

### 3.2 Forward pass — full decomposition

$$
\Delta W = \text{kron}(W_1,\, W_2) \cdot s
$$

where $W_1 \in \mathbb{R}^{l \times m}$ and $W_2 \in \mathbb{R}^{k \times n}$, producing $\Delta W \in \mathbb{R}^{lk \times mn} = \mathbb{R}^{d_\text{out} \times d_\text{in}}$.

### 3.3 Adaptive parameterization

LOKR adaptively chooses between full matrices and low-rank decompositions based on `lora_dim`:

| Condition | $W_1$ form | $W_2$ form |
|-----------|-----------|-----------|
| `decompose_both=true` and $r < \max(l, m)/2$ | $W_{1a} W_{1b}$, shapes $(l, r) \times (r, m)$ | $W_{2a} W_{2b}$, shapes $(k, r) \times (r, n)$ |
| `decompose_both=false` or $r \geq \max(l, m)/2$ | Full $(l, m)$ | $W_{2a} W_{2b}$ or full $(k, n)$ |
| $r \geq \max(k, n)/2$ or `full_matrix=true` | (as above) | Full $(k, n)$ |

Flags tracked: `use_w1` (full $W_1$), `use_w2` (full $W_2$).

### 3.4 Initialization

| Component | Init |
|-----------|------|
| `lokr_w1` (full) | Kaiming uniform |
| `lokr_w1_a`, `lokr_w1_b` | Kaiming uniform |
| `lokr_w2` (full) | **Zeros** (unless `use_scalar=true`, then Kaiming) |
| `lokr_w2_a` | Kaiming uniform |
| `lokr_w2_b` | **Zeros** (unless `use_scalar=true`, then Kaiming) |
| `lokr_t2` (Tucker) | Kaiming uniform |

At step 0: $\Delta W = \text{kron}(W_1,\, \mathbf{0}) \cdot s = \mathbf{0}$ because `lokr_w2` (or `lokr_w2_b`) is zero-initialized.

When both $W_1$ and $W_2$ are full matrices (`use_w1=true` and `use_w2=true`), the effective rank is $l \cdot k = d_\text{out}$ — a full-rank update. In this case `alpha` is set to `lora_dim` and `scale = 1.0` automatically.

### 3.5 Training-time forward (Linear) — Kronecker factored

The Kronecker product $\text{kron}(W_1, W_2) \cdot x$ can be computed without materializing the full $d_\text{out} \times d_\text{in}$ matrix. `KronLinearFn` (`lokr.py:11–52`) implements:

$$
X = \text{reshape}(x,\ [-1,\ m,\ n]), \quad \text{temp} = X\, W_2^\top, \quad \text{out} = \text{einsum}("pr,\text{brk} \to \text{bpk}",\, W_1,\, \text{temp})
$$

This computes $\text{kron}(W_1, W_2) \cdot x$ in $O(B \cdot l \cdot k \cdot (m + n))$ instead of $O(B \cdot lk \cdot mn)$.

When both $W_1$ and $W_2$ are further decomposed into low-rank pairs, `KronLinearTwoStageFn` (`lokr.py:55–131`) chains two Kronecker factored operations:

$$
\text{Stage 1:}\quad z = \text{kron\_factored}(W_{1b},\, W_{2b},\, x)
$$

$$
\text{Stage 2:}\quad \text{out} = \text{kron\_factored}(W_{1a},\, W_{2a},\, z)
$$

### 3.6 Conv2d — Tucker mode for $W_2$

When the target is Conv2d with kernel_size > 1 and `use_tucker = true`, $W_2$ is reconstructed via Tucker decomposition:

$$
W_2 = \text{rebuild\_tucker}(T_2,\, W_{2a},\, W_{2b})
$$

with `lokr_t2` of shape $(r,\ r,\ k_1,\ k_2)$.

---

## 4. Weight key naming conventions

| Variant | Key pattern | Notes |
|---------|-------------|-------|
| LOCON | `lora_down.weight`, `lora_up.weight`, `lora_mid.weight` (optional) | Same as standard LoRA for Linear; `lora_mid` is Tucker-specific |
| LOHA | `hada_w1_a`, `hada_w1_b`, `hada_w2_a`, `hada_w2_b`, `hada_t1`/`hada_t2` (optional) | `hada_` prefix distinguishes from standard LoRA keys |
| LOKR | `lokr_w1` or `lokr_w1_a`/`lokr_w1_b`, `lokr_w2` or `lokr_w2_a`/`lokr_w2_b`, `lokr_t2` (optional) | `lokr_` prefix; key presence depends on adaptive parameterization |
| All | `alpha` | Scalar, always present |

Auto-detection in `loading.py` uses key prefixes: `hada_` → LOHA, `lokr_` → LOKR, standard `lora_up`/`lora_down` with metadata → LOCON.

---

## 5. Scalar baking behavior

### LOCON

`custom_state_dict` bakes `scalar` into `lora_up.weight`:

$$
\text{saved\_up} = W_\text{up} \cdot \text{scalar}
$$

`scalar` starts at 1.0 and is only modified by `apply_max_norm` when the weight norm exceeds `scale_weight_norms`. After saving, the file contains standard LoRA keys with the baked-in norm correction.

### LOHA

`custom_state_dict` bakes `scalar` into `hada_w1_a`:

$$
\text{saved\_w1a} = W_1^a \cdot \text{scalar}
$$

`scalar` starts at 1.0 and is modified by `apply_max_norm`. The remaining factors (`w1_b`, `w2_a`, `w2_b`, Tucker cores) are saved unchanged.

### LOKR

`custom_state_dict` bakes `scalar` into $W_1$ (or $W_{1a}$):

$$
\text{saved\_lokr\_w1} = W_1 \cdot \text{scalar} \qquad \text{(full mode)}
$$

$$
\text{saved\_lokr\_w1\_a} = W_{1a} \cdot \text{scalar} \qquad \text{(decomposed mode)}
$$

When `use_scalar = true`, `scalar` is a learnable `nn.Parameter` (zero-init) instead of a buffer. In this case $W_2$ (or `lokr_w2_b`) is Kaiming-initialized (not zero), and the scalar provides the zero-at-init guarantee: $\Delta W = \text{kron}(W_1,\, W_2) \cdot 0 = \mathbf{0}$.

When `weight_decompose = true`, the additional `dora_scale` parameter is saved as a separate key.

---

## 6. Comparison with standard LoRA

| Property | LoRA | LOCON | LOHA | LOKR |
|----------|------|-------|------|------|
| $\Delta W$ formula | $BA$ | $BA$ (Linear); Tucker (Conv2d) | $(A_1 B_1) \odot (A_2 B_2)$ | $\text{kron}(W_1,\, W_2)$ |
| Params per Linear | $r(d_\text{in} + d_\text{out})$ | Same | $2r(d_\text{in} + d_\text{out})$ | Adaptive |
| Effective rank | $r$ | $r$ | $\leq r^2$ | $r_1 \cdot r_2$ (adaptive) |
| Conv2d support | Flattened | Tucker core | Tucker core | Tucker core on $W_2$ |
| Custom autograd | No | No | Yes (HadaWeight) | Yes (KronLinearFn) |
| Zero-at-init | $B = 0$ | $W_\text{up} = 0$ | $W_2^a = 0$ | $W_2$ or $W_{2b} = 0$ |
| Max-norm scalar baking | Into $W_\text{up}$ | Into $W_\text{up}$ | Into $W_1^a$ | Into $W_1$ or $W_{1a}$ |
| HydraLoRA compatible | Yes | No | No | No |
| OrthoLoRA compatible | Yes | No | No | No |
| DoRA (weight_decompose) | Yes | No | No | Yes |

### Parameter count comparison (Linear, $d_\text{in} = d_\text{out} = 2048$, $r = 16$)

| Variant | Parameters | Effective rank |
|---------|-----------|----------------|
| LoRA | $2 \times 2048 \times 16 = 65{,}536$ | 16 |
| LOCON | Same as LoRA | 16 |
| LOHA | $4 \times 2048 \times 16 = 131{,}072$ | $\leq 256$ |
| LOKR (decomposed) | depends on factorization; typically $< 65{,}536$ | depends on factorization |

---

## 7. Functional utilities

`networks/lora_modules/lycoris_functional.py` centralizes shared operations:

| Function | Used by | Purpose |
|----------|---------|---------|
| `rebuild_tucker(t, wa, wb)` | LOCON, LOKR | Tucker core reconstruction via `einsum("ij...,ir,jp->pr...", t, wb, wa)` |
| `factorization(dim, factor)` | LOKR | Find optimal $(m, n)$ with $m \cdot n = d$, $m \leq n$ |
| `make_kron(w1, w2, scale)` | LOKR | Kronecker product with broadcasting for dim mismatch |
| `HadaWeight` | LOHA | Custom autograd for Hadamard product of two matrix products |
| `HadaWeightTucker` | LOHA | Custom autograd for Tucker-reconstructed Hadamard product |
| `factored_kron_forward(x, w1, w2)` | — (reference) | Memory-efficient Kronecker matmul; used inline in `KronLinearFn` |
