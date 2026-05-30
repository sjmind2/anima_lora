[English](loha.md) | **中文**

# LoHA — 低秩 Hadamard 积适配

LoHA 是 LyCORIS 系列适配器的一种，通过对两个低秩矩阵进行 Hadamard（逐元素）积运算，在仅使用标准 LoRA 约 2 倍参数量的情况下实现有效秩 r²。自定义 autograd 函数为 Hadamard 积提供了精确梯度。

关于数学推导（前向公式、维度分析、权重键命名），请参阅 [docs/methods/lycoris-variants.md](../methods/lycoris-variants.md)。本指南涵盖使用方法与配置。

## 快速开始

### GUI

在 Anima GUI 的方法下拉菜单中选择 **LoHA**。专用配置 `configs/gui-methods/loha.toml` 会预填充所有变体特定参数。

## 配置参数

| 参数 | 默认值 | 说明 |
|-----------|---------|-------------|
| `network_dim` | 32 | 秩 r（有效秩 ≈ r²） |
| `network_alpha` | 16 | LoRA alpha。**缩放比例 = alpha / dim** — 推荐范围 0.1–0.5。 |
| `scale_weight_norms` | 1.0 | 最大范数缩放目标 |
| `conv_dim` | 4 | Conv2d 层的秩（**在 Anima 上无效** — DiT 不含 Conv2d） |
| `conv_alpha` | 1 | Conv2d 层的 alpha（**在 Anima 上无效**） |
| `use_tucker` | `true` | Conv2d 的 Tucker 模式（**在 Anima 上无效**） |

### 理解缩放比例

缩放因子 `s = alpha / dim` 控制权重更新 ΔW 的大小。对于 LoHA：

- `dim=32, alpha=16` → scale = 0.5（良好的起点）
- `dim=32, alpha=8` → scale = 0.25（更保守）
- `dim=16, alpha=8` → scale = 0.5（相同缩放，更低的有效秩）

推荐缩放范围：**0.1–0.5**。缩放越高 = 适配越强，但存在不稳定风险。

## 与 LoKR 的对比

| 特性 | LoHA | LoKR |
|---------|------|------|
| 核心运算 | Hadamard（逐元素）积 | Kronecker 积 |
| 有效秩 | r² | rank(W1) × rank(W2)（自适应） |
| 参数量（相同 r） | ~2× LoRA | 自适应；可小于 LoRA |
| DoRA 支持 | 否 | 是（`weight_decompose=true`） |
| 自定义 autograd | `HadaWeight` / `HadaWeightTucker` | `KronLinearFn` / `KronLinearTwoStageFn` |
| 适用场景 | 在给定 r 下获得更高的有效秩 | 通过因子调节实现灵活的参数量控制 |

## 推理

### CLI — 静态合并

`inference.py` 通过检查 safetensors 键前缀（`hada_*`）自动检测 LoHA。权重增量使用 Hadamard 积公式计算，并合并到基础模型权重中。

LoHA 检查点可以与普通 LoRA 或 LoKR 文件共存于同一个 `--lora_weight` 列表中。

### ComfyUI

ComfyUI 的 LyCORIS 加载器节点原生支持 LoHA 权重格式，可直接加载，无需转换。

## 兼容性

与 LoKR 相同的互斥规则：

| 可组合 | 备注 |
|-------------|-------|
| **T-LoRA** | 时间步秩掩码在 `make_weight` 之后应用 |
| **Spectrum** | 无交互 |
| **Modulation guidance** | 正交 |
| **ReFT** | 正交侧通道 |
| **P-GRAFT** | 截断步骤切换 `network.enabled` |
| **HydraLoRA** | **不支持** |
| **OrthoLoRA** | **不支持** |

## 推荐配置

### 标准训练

```toml
network_type = "loha"
network_dim = 32
network_alpha = 16
scale_weight_norms = 1.0
learning_rate = 1e-4
max_train_epochs = 4
```

### 搭配 T-LoRA 与时间步掩码

```toml
network_type = "loha"
network_dim = 32
network_alpha = 16
use_timestep_mask = true
min_rank = 8
scale_weight_norms = 1.0
learning_rate = 1e-4
max_train_epochs = 6
```
