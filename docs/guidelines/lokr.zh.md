[English](lokr.md) | **中文**

# LoKR — 低秩 Kronecker 积适配

LoKR 是 LyCORIS 系列适配器的一种，通过分解权重维度并使用 Kronecker 积进行组合，生成结构化的高秩近似，其参数量取决于分解形状而非单一的秩值。

关于数学推导（前向公式、维度分析、权重键命名、标量烘焙），请参阅 [docs/methods/lycoris-variants.md](../methods/lycoris-variants.md)。本指南涵盖使用方法与配置说明。

## 快速开始

### GUI

在 Anima GUI 的方法下拉菜单中选择 **LoKR**。专用配置 `configs/gui-methods/lokr.toml` 会预填充所有变体特定的参数。

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `network_dim` | 8 | 低秩分解的秩 |
| `network_alpha` | 8 | LoRA alpha（缩放比例 = alpha / dim） |
| `decompose_both` | `true` | 同时将 W1 和 W2 分解为低秩对。在较小的秩下可显著减少参数量。 |
| `lokr_factor` | -1 | 维度分解的目标因子大小。`-1` = 自动（推荐）。 |
| `scale_weight_norms` | 1.0 | 最大范数缩放目标 |
| `weight_decompose` | `false` | 启用 DoRA 风格的权重分解（仅限 LoKR — LoHA/LoCON 不可用） |
| `use_scalar` | `false` | 可学习标量（零初始化），替代固定的 scalar=1 |
| `full_matrix` | `false` | 强制使用完整（未分解的）矩阵 |
| `conv_dim` | 4 | Conv2d 层的秩（**在 Anima 上无效** — DiT 不含 Conv2d） |
| `conv_alpha` | 4 | Conv2d 层的 alpha（**在 Anima 上无效**） |
| `use_tucker` | `true` | Conv2d 的 Tucker 核分解（**在 Anima 上无效**） |

## Anima 专项说明

### QKV 融合与 `lokr_factor`

Anima-base-v1.0 DiT 将 Q/K/V 融合为单个 `qkv_proj` Linear 层，形状为 `[6144, 2048]`。在保存检查点以兼容 ComfyUI 时，此融合模块必须拆分回独立的 `q_proj`/`k_proj`/`v_proj`。拆分要求 `factorization(6144, factor)` 生成的 `out_l` 能被 3 整除。

**推荐的 `lokr_factor` 值：**

| `lokr_factor` | 原始 `out_l` | 调整后 `out_l` | 检查点大小 |
|---------------|-------------|----------------|-----------|
| **-1**（自动） | auto | auto | ~10 MB |
| **6** | 6 | 6（无变化） | ~13 MB |
| **12** | 12 | 12（无变化） | ~7 MB |
| **24** | 24 | 24（无变化） | ~4 MB |
| 4 | 4 | **3**（已调整） | ~10 MB |
| 8 | 8 | **6**（已调整） | ~13 MB |
| 16 | 16 | **12**（已调整） | ~7 MB |

所有值都能生成正确大小的检查点。使用 3 的倍数（6、12、24）作为因子可完全避免调整。推荐使用 `lokr_factor = -1`（自动）以获得最均衡的分解。

### Anima DiT 中无 Conv2d

Anima-base-v1.0 DiT 中所有 LoRA 目标层均为 `nn.Linear`。`conv_dim`、`conv_alpha` 和 `use_tucker` 参数虽然会被接受且不会报错，但不会产生任何效果。

## 推理

### 命令行 — 静态合并

`inference.py` 通过检查 safetensors 键前缀（`lokr_*`）自动检测 LoKR。权重增量使用 Kronecker 积公式计算，并在去噪前合并到基础模型权重中。

LoKR 检查点可以与常规 LoRA 文件共存于同一 `--lora_weight` 列表中 — 每个文件独立合并。

### ComfyUI

ComfyUI 的 LyCORIS 加载器节点原生支持 LoKR 权重格式。本训练器生成的 safetensors 文件使用与 sd-scripts / LyCORIS 生态系统相同的键命名约定，因此可直接加载，无需转换。

## 兼容性

| 可组合使用 | 备注 |
|-----------|------|
| **T-LoRA** | 时间步秩掩码在重建权重的 `make_weight` 之后应用 |
| **Spectrum** | 无交互 — 缓存步骤完全跳过块 |
| **Modulation guidance** | 正交 — 仅涉及 AdaLN |
| **ReFT** | 正交侧通道 |
| **P-GRAFT** | 截断步骤切换 `network.enabled` |
| **HydraLoRA** | **不支持** — 需要标准 BA 结构 |
| **OrthoLoRA** | **不支持** — Cayley 重参数化仅针对标准 BA 定义 |

## 推荐配置

### 小数据集（≤ 20 张图像）

```toml
network_type = "lokr"
network_dim = 8
network_alpha = 8
decompose_both = true
lokr_factor = -1
scale_weight_norms = 1.0
learning_rate = 2e-5
max_train_epochs = 4
```

### 大数据集（100+ 张图像）

```toml
network_type = "lokr"
network_dim = 16
network_alpha = 16
decompose_both = true
lokr_factor = 12
scale_weight_norms = 1.0
learning_rate = 1e-4
max_train_epochs = 8
```
